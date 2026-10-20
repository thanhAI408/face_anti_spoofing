"""Validation-only controlled trial: balanced sampling plus Group/LayerNorm.

Sampling, optimizer, augmentation, warmup, selection metric and API shapes
match train_balanced_merged.py. Only normalization layers change. Five
convolution BatchNorm layers become GroupNorm with eight groups; the Dense
BatchNorm becomes LayerNorm. Training always starts from fresh random weights.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np

from scripts.train_balanced_merged import (
    BalancedVideoScheduler, canonical_sha, exposure_summary,
    load_train_validation_pixels, make_dataset, source_class_video_loss,
    source_class_video_weights, warmup_schedule,
)
from scripts.train_retraining_comparison import (
    check_split, evaluate_groups, predictions, save_json, sha,
    source_summary, video_balanced_weights,
)


def build_groupnorm_model(tf):
    """Preserve the CNN topology and replace only its normalization layers."""
    layers = tf.keras.layers
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.GroupNormalization(groups=8, axis=-1, epsilon=.001))
    model.add(layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(layers.Dropout(.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.GroupNormalization(groups=8, axis=-1, epsilon=.001))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.GroupNormalization(groups=8, axis=-1, epsilon=.001))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(.25))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.GroupNormalization(groups=8, axis=-1, epsilon=.001))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.GroupNormalization(groups=8, axis=-1, epsilon=.001))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.LayerNormalization(axis=-1, epsilon=.001))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(2, activation='softmax'))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='dataset/merged-20261001')
    parser.add_argument('--output', default='outputs/merged-groupnorm-20261001')
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
    config = {'seed': args.seed, 'architecture': 'LivenessNet CNN 32x32 BGR; Conv GroupNorm(8), Dense LayerNorm',
              'input': 'uint8 BGR / 255', 'optimizer': 'Adam', 'initial_learning_rate': 1e-4,
              'fresh_initialization': True, 'baseline_or_warmstart_loaded': False,
              'dataset_manifest_sha256': sha(manifest_path), 'dataset_archive_sha256': sha(archive_path),
              'training_script_sha256': sha(__file__),
              'training_script_path': 'scripts/train_groupnorm_merged.py',
              'architecture_source_sha256': sha(__file__),
              'architecture_source_path': 'scripts/train_groupnorm_merged.py',
              'norm_type': 'group-layer',
              'normalization': {'convolution': 'GroupNormalization', 'groups': 8,
                                'dense': 'LayerNormalization', 'axis': -1, 'epsilon': .001},
              'balanced_helper_source_sha256': sha(ROOT / 'scripts/train_balanced_merged.py'),
              'balanced_helper_source_path': 'scripts/train_balanced_merged.py',
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
    model = build_groupnorm_model(tf)
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


