"""Train a source/class/video balanced candidate using validation alone.

One separately reported warmup pass exposes every train crop. Subsequent small
epochs contain the same number of draws from each source/class group, choose
videos uniformly, and cycle shuffled frame queues across epoch boundaries.
No test tensor is constructed and no test prediction or metric is computed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np

from scripts.train_retraining_comparison import (
    check_split, evaluate_groups, predictions, save_json, sha,
    source_of, source_summary, video_balanced_weights,
)


def canonical_sha(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def warmup_schedule(frame_count, seed):
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError('Warmup needs a positive frame count')
    return np.random.default_rng(np.random.SeedSequence([seed, 91])).permutation(frame_count)


def groups_and_videos(records, required_split):
    """Validate four source/class cells, with consistent video provenance."""
    records = list(records)
    if not records or any(r.get('split') != required_split for r in records):
        raise ValueError(f'All records must belong to {required_split}')
    sources = sorted({source_of(r) for r in records})
    if len(sources) != 2:
        raise ValueError('This protocol requires exactly two sources')
    groups = {(source, label): [] for source in sources for label in ('fake', 'real')}
    video_indices, video_metadata = {}, {}
    for index, record in enumerate(records):
        key = (source_of(record), record.get('label'))
        if key not in groups:
            raise ValueError('Frame labels must be fake or real')
        video = record.get('video_id')
        if not isinstance(video, str) or not video:
            raise ValueError('Frames need nonempty video_id')
        if video_metadata.setdefault(video, key) != key:
            raise ValueError('A video cannot belong to different source/class groups')
        video_indices.setdefault(video, []).append(index)
    for video, key in video_metadata.items():
        groups[key].append(video)
    if any(not videos for videos in groups.values()):
        raise ValueError('Each source must contain fake and real videos')
    return {key: tuple(sorted(videos)) for key, videos in sorted(groups.items())}, {
        video: np.asarray(indices, dtype=np.int64) for video, indices in video_indices.items()}


class BalancedVideoScheduler:
    """Pure NumPy stateful sampler; one draw is one augmented training example.

    Every batch has equal source/class mass. Each group consumes a shuffled
    permutation of its video IDs before starting another round. A video's frame
    queue is also shuffled and exhausted before reuse. Both queues persist
    between epochs, so the final partial cycle does not restart each epoch.
    """
    def __init__(self, records, batch_size=32, steps=None, seed=42):
        self.records = list(records)
        self.groups, self.video_indices = groups_and_videos(self.records, 'train')
        if not isinstance(batch_size, int) or batch_size < 4 or batch_size % 4:
            raise ValueError('batch_size must be a positive multiple of four')
        self.batch_size = batch_size
        self.per_group = batch_size // 4
        self.steps = math.ceil(len(records) / batch_size) if steps is None else steps
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        self.rng = np.random.default_rng(seed)
        self.video_orders = {key: [] for key in self.groups}
        self.video_cursors = {key: 0 for key in self.groups}
        self.frame_orders = {video: np.empty(0, dtype=np.int64) for video in self.video_indices}
        self.frame_cursors = {video: 0 for video in self.video_indices}
        self.epochs_generated = 0

    def _draw(self, key):
        if self.video_cursors[key] >= len(self.video_orders[key]):
            self.video_orders[key] = self.rng.permutation(self.groups[key]).tolist()
            self.video_cursors[key] = 0
        video = self.video_orders[key][self.video_cursors[key]]
        self.video_cursors[key] += 1
        if self.frame_cursors[video] >= len(self.frame_orders[video]):
            self.frame_orders[video] = self.rng.permutation(self.video_indices[video])
            self.frame_cursors[video] = 0
        result = int(self.frame_orders[video][self.frame_cursors[video]])
        self.frame_cursors[video] += 1
        return result

    def next_epoch(self):
        batches = np.empty((self.steps, self.batch_size), dtype=np.int64)
        for batch in batches:
            draws = [self._draw(key) for key in self.groups for _ in range(self.per_group)]
            batch[:] = self.rng.permutation(draws)
        self.epochs_generated += 1
        return batches


def source_class_video_weights(records):
    """Weights average frame CE within video, then class, then source."""
    groups, videos = groups_and_videos(records, 'validation')
    weights = np.empty(len(records), dtype=np.float64)
    for key, video_ids in groups.items():
        for video in video_ids:
            ids = videos[video]
            weights[ids] = len(records) / (len(groups) * len(video_ids) * len(ids))
    if not np.isclose(weights.sum(), len(records), rtol=0, atol=1e-9):
        raise ValueError('Validation weights do not sum to the validation frame count')
    return weights


def source_class_video_loss(probabilities, records):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(records), 2):
        raise ValueError('Expected one two-class probability row per validation frame')
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError('Validation probabilities must be finite normalized probabilities')
    targets = np.asarray([0 if r['label'] == 'fake' else 1 for r in records])
    losses = -np.log(np.clip(probabilities[np.arange(len(records)), targets], 1e-7, 1 - 1e-7))
    return float(np.dot(losses, source_class_video_weights(records)) / len(records))


def exposure_summary(records, counts):
    counts = np.asarray(counts)
    if (counts.shape != (len(records),) or counts.dtype.kind not in 'iu'
            or np.any(counts < 0)):
        raise ValueError('Exposure counts must be nonnegative integers for every train frame')
    groups, videos = groups_and_videos(records, 'train')
    result = {'frames': len(records), 'draws': int(counts.sum()),
              'unique_frames_exposed': int(np.count_nonzero(counts)),
              'fraction_exposed': float(np.count_nonzero(counts) / len(records)),
              'minimum_draws_per_frame': int(counts.min()),
              'maximum_draws_per_frame': int(counts.max()), 'groups': {}, 'videos': {}}
    for key, video_ids in groups.items():
        ids = np.concatenate([videos[video] for video in video_ids])
        result['groups']['|'.join(key)] = {
            'source': key[0], 'label': key[1], 'videos': len(video_ids), 'frames': len(ids),
            'draws': int(counts[ids].sum()), 'unique_frames_exposed': int(np.count_nonzero(counts[ids]))}
    for video, ids in sorted(videos.items()):
        record = records[int(ids[0])]
        result['videos'][video] = {
            'source': source_of(record), 'label': record['label'], 'frames': len(ids),
            'draws': int(counts[ids].sum()), 'unique_frames_exposed': int(np.count_nonzero(counts[ids])),
            'fraction_exposed': float(np.count_nonzero(counts[ids]) / len(ids)),
            'minimum_draws_per_frame': int(counts[ids].min()),
            'maximum_draws_per_frame': int(counts[ids].max())}
    return result


def load_train_validation_pixels(archive_path, records):
    """Stream the archive; only train/validation rows become NumPy tensors.

    The compressed archive contains all splits. Its other bytes must be
    decompressed to advance the stream, but test rows are discarded without
    conversion, retention, preprocessing, inference, or numerical inspection.
    """
    selected = {split: [i for i, r in enumerate(records) if r['split'] == split]
                for split in ('train', 'validation')}
    if any(not ids for ids in selected.values()):
        raise ValueError('Train and validation must both have frames')
    outputs = {split: np.empty((len(ids), 32, 32, 3), dtype=np.uint8)
               for split, ids in selected.items()}
    positions = {original: (split, offset) for split, ids in selected.items()
                 for offset, original in enumerate(ids)}
    row_bytes = 32 * 32 * 3
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open('x.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
            if shape != (len(records), 32, 32, 3) or fortran_order or dtype != np.dtype(np.uint8):
                raise ValueError('Expected aligned contiguous uint8 Nx32x32x3 BGR archive')
            for index in range(len(records)):
                raw = stream.read(row_bytes)
                if len(raw) != row_bytes:
                    raise ValueError('Truncated frame archive')
                if index in positions:
                    split, offset = positions[index]
                    outputs[split][offset] = np.frombuffer(raw, dtype=np.uint8).reshape(32, 32, 3)
            if stream.read(1):
                raise ValueError('Unexpected bytes after the declared frame array')
    return outputs, selected


def augmentation_parameters(rng, shape):
    return {'theta': float(rng.uniform(-12, 12)),
            'tx': float(rng.uniform(-.10, .10) * shape[0]),
            'ty': float(rng.uniform(-.10, .10) * shape[1]),
            'shear': 0, 'zx': float(rng.uniform(.90, 1.10)),
            'zy': float(rng.uniform(.90, 1.10)),
            'flip_horizontal': bool(rng.random() < .5), 'flip_vertical': False,
            'channel_shift_intensity': None, 'brightness': None}


def make_dataset(tf, x, y, schedules, augmentor, seed, batch_size, weights=None):
    class TrainingDataset(tf.keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
            self.epoch = 0
            self.schedule = schedules(0)

        def set_epoch(self, epoch):
            if epoch != self.epoch:
                if epoch != self.epoch + 1:
                    raise ValueError('Training epochs must advance consecutively')
                self.epoch = epoch
                self.schedule = schedules(epoch)

        def __len__(self):
            return math.ceil(len(self.schedule) / batch_size)

        def __getitem__(self, batch_index):
            if batch_index < 0 or batch_index >= len(self):
                raise IndexError(batch_index)
            ids = self.schedule[batch_index * batch_size:(batch_index + 1) * batch_size]
            rng = np.random.default_rng(np.random.SeedSequence([seed, self.epoch, batch_index]))
            batch = np.asarray([augmentor.apply_transform(x[int(index)], augmentation_parameters(rng, x.shape[1:]))
                                for index in ids], dtype=np.float32)
            if weights is not None:
                return batch, y[ids], weights[ids]
            return batch, y[ids]

        # Keras may inspect initial batches for its output signature. Scheduling
        # advances solely from the explicit epoch callback, not adapter probes.
        def on_epoch_end(self):
            pass

    return TrainingDataset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset/merged-20261001')
    parser.add_argument('--output', default='outputs/merged-balanced-20261001')
    parser.add_argument('--epochs', type=int, default=50, help='Balanced epochs, excluding the full warmup pass')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.cpu_threads < 1 or args.seed < 0:
        parser.error('epochs/threads must be positive and seed nonnegative')
    if args.batch_size < 4 or args.batch_size % 4:
        parser.error('batch-size must be a positive multiple of four')

    data_dir, out = ROOT / args.dataset, ROOT / args.output
    manifest_path, archive_path = data_dir / 'manifest.json', data_dir / 'frames.npz'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Only a complete prepared manifest may be trained')
    records = manifest['frames']
    check_split(records, manifest.get('split_policy'))  # Metadata-only leakage audit.
    if any(record['array_index'] != index for index, record in enumerate(records)):
        raise ValueError('Manifest array_index does not match archive row order')
    if sha(archive_path) != manifest['array']['sha256']:
        raise ValueError('Frame archive hash differs from its preparation manifest')
    pixels, indices = load_train_validation_pixels(archive_path, records)
    selected_records = {split: [records[index] for index in ids] for split, ids in indices.items()}
    scheduler = BalancedVideoScheduler(selected_records['train'], args.batch_size, seed=args.seed)
    source_class_video_weights(selected_records['validation'])
    warmup = warmup_schedule(len(selected_records['train']), args.seed)
    out.mkdir(parents=True, exist_ok=False)
    config = {'seed': args.seed, 'architecture': 'LivenessNet CNN 32x32 BGR; unchanged',
              'input': 'uint8 BGR / 255', 'optimizer': 'Adam', 'initial_learning_rate': 1e-4,
              'fresh_initialization': True, 'baseline_or_warmstart_loaded': False,
              'dataset_manifest_sha256': sha(manifest_path), 'dataset_archive_sha256': sha(archive_path),
              'training_script_sha256': sha(__file__),
              'architecture_source_sha256': sha(ROOT / 'model/livenessnet.py'),
              'batch_size': args.batch_size, 'cpu_threads': args.cpu_threads,
              'warmup': {'epochs': 1, 'frames': len(warmup),
                         'steps': math.ceil(len(warmup) / args.batch_size),
                         'order_sha256': hashlib.sha256(warmup.astype('<i8').tobytes()).hexdigest(),
                         'weighting': 'class then video; full shuffled pass; all frames exactly once',
                         'eligible_for_checkpoint': False},
              'balanced': {'max_epochs': args.epochs, 'steps_per_epoch': scheduler.steps,
                           'draws_per_epoch': scheduler.steps * args.batch_size,
                           'groups': [list(key) for key in scheduler.groups],
                           'draws_per_group_per_batch': scheduler.per_group,
                           'sample_weights': False,
                           'sampling': 'equal source/class cells, shuffled uniform video rounds; persistent shuffled frame cycles',
                           'coverage': 'Small balanced epochs need not expose all frames; queues persist; warmup exposes all'},
              'augmentation': {'rotation_degrees': 12, 'zoom_range': [.90, 1.10],
                               'width_shift_fraction': .10, 'height_shift_fraction': .10,
                               'horizontal_flip': True, 'fill_mode': 'nearest'},
              'early_stopping': {'monitor': 'val_source_class_video_loss', 'mode': 'min', 'patience': 12},
              'reduce_lr': {'monitor': 'val_source_class_video_loss', 'factor': .5, 'patience': 4,
                            'min_lr': 1e-6},
              'test_policy': 'No test tensor, prediction, or metric. Test metadata only for split integrity.',
              'checkpoint_policy': 'Only balanced-epoch validation source/class/video cross-entropy selects the model',
              'split_policy': manifest.get('split_policy', {}),
              'train_validation_summary': source_summary([r for r in records if r['split'] in indices])}
    config['protocol_hashed_fields'] = sorted(config)
    config['protocol_sha256'] = canonical_sha({key: config[key] for key in config['protocol_hashed_fields']})
    save_json(out / 'configuration.json', config)

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.preprocessing import LabelEncoder
    from model.livenessnet import LivenessNet
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    config['tensorflow_version'] = tf.__version__
    config['deterministic_ops_enabled'] = True
    save_json(out / 'configuration.json', config)
    started = time.time()
    x = {split: value.astype(np.float32) / 255.0 for split, value in pixels.items()}
    del pixels
    le = LabelEncoder().fit(['fake', 'real'])
    y = {split: tf.keras.utils.to_categorical(le.transform([r['label'] for r in values]), 2)
         for split, values in selected_records.items()}
    model = LivenessNet.build(32, 32, 3, 2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    augmentor = ImageDataGenerator(fill_mode='nearest')
    warmup_data = make_dataset(tf, x['train'], y['train'], lambda epoch: warmup,
                               augmentor, args.seed + 91, args.batch_size,
                               video_balanced_weights(selected_records['train']))

    class WarmupProgress(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            save_json(out / 'progress.json', {'phase': 'warmup', 'frames': len(warmup),
                                            'steps': len(warmup_data), 'eligible_for_checkpoint': False})
        def on_epoch_end(self, epoch, logs=None):
            print(json.dumps({'phase': 'warmup_complete', 'epoch': 1,
                              'seconds_elapsed': time.time() - started,
                              **{key: float(value) for key, value in (logs or {}).items()}}), flush=True)

    warmup_history = model.fit(warmup_data, epochs=1, shuffle=False,
                               callbacks=[WarmupProgress()], verbose=2).history
    save_json(out / 'warmup_history.json', {key: [float(value) for value in values]
                                          for key, values in warmup_history.items()})
    optimizer_steps_warmup = int(model.optimizer.iterations.numpy())
    if optimizer_steps_warmup != len(warmup_data):
        raise ValueError('Warmup optimizer steps differ from the complete train pass')
    epoch_draw_counts, epoch_order_hashes, epoch_orders = [], [], []
    np.save(out / 'warmup_order.npy', warmup.astype(np.int32))

    def schedules(epoch):
        if epoch != len(epoch_draw_counts):
            raise ValueError('Balanced schedule must be generated exactly once per epoch')
        schedule = scheduler.next_epoch().reshape(-1)
        epoch_draw_counts.append(np.bincount(schedule, minlength=len(x['train'])).astype(np.int32))
        epoch_order_hashes.append(hashlib.sha256(schedule.astype('<i8').tobytes()).hexdigest())
        epoch_orders.append(schedule.astype(np.int32))
        return schedule

    balanced_data = make_dataset(tf, x['train'], y['train'], schedules,
                                 augmentor, args.seed, args.batch_size)

    class ValidationBalance(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            balanced_data.set_epoch(epoch)

        def on_epoch_end(self, epoch, logs=None):
            probabilities = predictions(self.model, x['validation'])
            metrics = evaluate_groups(probabilities, selected_records['validation'], tuple(le.classes_))
            monitor = source_class_video_loss(probabilities, selected_records['validation'])
            logs['val_source_class_video_loss'] = monitor
            logs['val_loss'], logs['val_accuracy'] = metrics['loss'], metrics['accuracy']
            logs['val_macro_f1'] = metrics['classification_report']['macro avg']['f1-score']
            for source, group in metrics['per_source'].items():
                key = source.replace('-', '_')
                logs[f'val_{key}_loss'] = group['loss']
                logs[f'val_{key}_fake_recall'] = group['classification_report']['fake']['recall']
            save_json(out / f'validation_epoch_{epoch+1:03d}.json', {
                'balanced_epoch': epoch + 1, 'selection_loss': monitor, 'metrics': metrics,
                'schedule_sha256': epoch_order_hashes[epoch],
                'balanced_epoch_exposure': exposure_summary(selected_records['train'], epoch_draw_counts[epoch])})

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            result = {'phase': 'balanced_training', 'balanced_epoch': epoch + 1,
                      'warmup_epochs': 1, 'optimizer_steps': int(self.model.optimizer.iterations.numpy()),
                      'seconds_elapsed': time.time() - started,
                      **{key: float(value) for key, value in (logs or {}).items()}}
            save_json(out / 'progress.json', result)
            print(json.dumps(result), flush=True)

    monitor = 'val_source_class_video_loss'
    callbacks = [ValidationBalance(),
                 tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor=monitor,
                                                    mode='min', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='min', patience=12,
                                                 restore_best_weights=False),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode='min', factor=.5,
                                                     patience=4, min_lr=1e-6),
                 tf.keras.callbacks.CSVLogger(str(out / 'training.csv')), Progress()]
    history = model.fit(balanced_data, epochs=args.epochs, shuffle=False,
                        callbacks=callbacks, verbose=2).history
    history = {key: [float(value) for value in values] for key, values in history.items()}
    save_json(out / 'history.json', history)
    balanced_epochs = len(history['loss'])
    best_epoch = int(np.argmin(history[monitor])) + 1
    total_optimizer_steps = int(model.optimizer.iterations.numpy())
    if total_optimizer_steps != optimizer_steps_warmup + balanced_epochs * scheduler.steps:
        raise ValueError('Optimizer steps disagree with completed warmup/balanced schedules')
    if len(epoch_draw_counts) != balanced_epochs:
        raise ValueError('Scheduled epochs differ from completed epochs')
    best = tf.keras.models.load_model(out / 'best.keras', compile=False)
    best.save(out / 'liveness.h5')
    (out / 'le.pickle').write_bytes(pickle.dumps(le))
    exported = tf.keras.models.load_model(out / 'liveness.h5', compile=False)
    samples = np.concatenate((x['train'][:4], x['validation'][:4]))
    expected, actual = predictions(best, samples), predictions(exported, samples)
    if (exported.input_shape[1:] != (32, 32, 3) or exported.output_shape[-1] != 2
            or not np.allclose(expected, actual, rtol=1e-5, atol=1e-6)):
        raise ValueError('H5 export fails probability or API input/output compatibility')
    validation = {}
    for split in ('train', 'validation'):
        probabilities = predictions(exported, x[split])
        np.save(out / f'selected_{split}_probabilities.npy', probabilities)
        validation[split] = evaluate_groups(probabilities, selected_records[split], tuple(le.classes_))
    validation['validation']['source_class_video_loss'] = source_class_video_loss(
        np.load(out / 'selected_validation_probabilities.npy'), selected_records['validation'])
    if not np.isclose(validation['validation']['source_class_video_loss'], history[monitor][best_epoch - 1],
                      rtol=1e-5, atol=1e-6):
        raise ValueError('Reloaded selected checkpoint differs from its validation selection loss')
    warmup_counts = np.ones(len(x['train']), dtype=np.int32)
    draws = np.stack(epoch_draw_counts)
    np.savez_compressed(out / 'sample_exposure.npz', warmup=warmup_counts, balanced_by_epoch=draws)
    np.savez_compressed(out / 'balanced_schedules.npz', indices=np.stack(epoch_orders))
    config['export_compatibility'] = {'passed': True, 'labels': list(le.classes_),
                                      'frames_checked': len(samples),
                                      'max_probability_difference': float(np.abs(expected - actual).max()),
                                      'scope': 'H5 serialization/input/output; no HTTP or camera test'}
    save_json(out / 'configuration.json', config)
    report = {'configuration': config, 'epochs_run': balanced_epochs, 'best_epoch': best_epoch,
              'balanced_epochs_run': balanced_epochs, 'selected_balanced_epoch': best_epoch,
              'warmup_epochs': 1, 'total_passes_including_warmup': 1 + balanced_epochs,
              'optimizer_steps': {'warmup': optimizer_steps_warmup,
                                  'balanced': total_optimizer_steps - optimizer_steps_warmup,
                                  'total': total_optimizer_steps,
                                  'selected_checkpoint': optimizer_steps_warmup + best_epoch * scheduler.steps},
              'model_sha256': sha(out / 'liveness.h5'), 'best_keras_sha256': sha(out / 'best.keras'),
              'labels_sha256': sha(out / 'le.pickle'), 'configuration_sha256': sha(out / 'configuration.json'),
              'sample_exposure_sha256': sha(out / 'sample_exposure.npz'),
              'balanced_schedules_sha256': sha(out / 'balanced_schedules.npz'),
              'warmup_order_sha256': sha(out / 'warmup_order.npy'),
              'total_seconds': time.time() - started,
              'selection_monitor': monitor, 'selection_metric': monitor,
              'minimum_weighted_validation_ce': history[monitor][best_epoch - 1],
              'history_at_best_epoch': {key: values[best_epoch - 1] for key, values in history.items()},
              'history_at_last_epoch': {key: values[-1] for key, values in history.items()},
              'validation_only_trial': True,
              'test_predictions_or_metrics_created': False, 'selected': validation,
              'epoch_schedule_sha256': epoch_order_hashes,
              'exposure_at_selected_epoch': exposure_summary(selected_records['train'],
                                                            warmup_counts + draws[:best_epoch].sum(axis=0)),
              'balanced_exposure_at_selected_epoch': exposure_summary(selected_records['train'],
                                                                     draws[:best_epoch].sum(axis=0)),
              'exposure_after_training': exposure_summary(selected_records['train'], warmup_counts + draws.sum(axis=0)),
              'balanced_exposure_after_training': exposure_summary(selected_records['train'], draws.sum(axis=0)),
              'limitations': [
                  'Source balancing changes the training objective; larger image count does not guarantee improvement.',
                  'A warmup uses every train frame once; small balanced epochs do not each cover every frame.',
                  'Repeated short-video draws receive different augmentation but are correlated observations.',
                  'Validation selects this candidate; validation is not an untouched generalization estimate.',
                  'No test performance conclusion or API deployment is made by this helper.',
                  'Old identities may overlap between splits; video boundaries remain disjoint.']}
    save_json(out / 'validation_selection.json', report)
    save_json(out / 'training_summary.json', report)
    result = {'phase': 'complete', 'warmup_epochs': 1, 'balanced_epochs_run': balanced_epochs,
              'selected_balanced_epoch': best_epoch,
              'selection_validation_loss': validation['validation']['source_class_video_loss'],
              'optimizer_steps': total_optimizer_steps, 'model_sha256': report['model_sha256'],
              'test_evaluated': False, 'seconds_elapsed': time.time() - started}
    save_json(out / 'progress.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
