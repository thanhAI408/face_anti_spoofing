"""Train a compatible CNN and compare it with a frozen pre-training baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scripts.retraining_metrics import evaluate_probabilities, paired_video_bootstrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def predictions(model, x, batch_size=128):
    return np.concatenate([model(x[start:start+batch_size], training=False).numpy()
                           for start in range(0, len(x), batch_size)])


def source_of(record):
    return str(record.get('source', 'unspecified'))


def baseline_training_metadata(model_path):
    """Load historical epochs from the selected baseline's own run directory.

    Missing metadata is reported explicitly; metadata identifying a different
    model or an impossible best epoch fails before the expensive training run.
    """
    model_path = Path(model_path)
    run_dir = model_path.parent
    evaluation_path, history_path = run_dir / 'evaluation.json', run_dir / 'history.json'
    metadata = {'run_directory': str(run_dir), 'model_sha256': sha(model_path),
                'comparison_note': 'Historical loss uses different data; compare before/after clean evaluation loss instead'}
    if not evaluation_path.is_file() or not history_path.is_file():
        metadata.update({'status': 'unavailable', 'epochs_run': None, 'best_epoch': None,
                         'reason': 'Baseline evaluation.json/history.json are not both present'})
        return metadata
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    history = json.loads(history_path.read_text(encoding='utf-8'))
    expected_sha = evaluation.get('new_model_sha256') or evaluation.get('model_sha256')
    if expected_sha and expected_sha != metadata['model_sha256']:
        raise ValueError('Baseline run metadata does not match the selected baseline model hash')
    epochs, best = evaluation.get('epochs_run'), evaluation.get('best_epoch')
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError('Baseline metadata needs a positive integer epochs_run')
    if not isinstance(best, int) or isinstance(best, bool) or not 1 <= best <= epochs:
        raise ValueError('Baseline best_epoch must lie within epochs_run')
    for key in ('loss', 'val_loss'):
        if not isinstance(history.get(key), list) or len(history[key]) != epochs:
            raise ValueError(f'Baseline {key} history does not match epochs_run')
    if int(np.argmin(history['val_loss'])) + 1 != best:
        raise ValueError('Baseline best_epoch differs from its validation-loss history')
    metadata.update({'status': 'available', 'epochs_run': epochs, 'best_epoch': best,
                     'evaluation_path': str(evaluation_path), 'evaluation_sha256': sha(evaluation_path),
                     'history_path': str(history_path), 'history_sha256': sha(history_path),
                     'model_hash_verified_by_metadata': bool(expected_sha),
                     'historical_augmented_train_loss_at_best': float(history['loss'][best-1]),
                     'historical_validation_loss_at_best': float(history['val_loss'][best-1])})
    return metadata


def check_split(records, split_policy=None):
    main = ('train', 'validation', 'test')
    disjoint_sources = (split_policy or {}).get('subject_disjoint_sources')
    if disjoint_sources is not None:
        if (not isinstance(disjoint_sources, list) or not disjoint_sources
                or any(not isinstance(source, str) or not source for source in disjoint_sources)):
            raise ValueError('subject_disjoint_sources must be a nonempty list of source names')
        if set(disjoint_sources) - {source_of(r) for r in records}:
            raise ValueError('subject_disjoint_sources refers to a source absent from frame records')
    for a in main:
        a_records = [r for r in records if r['split'] == a]
        if {r['label'] for r in a_records} != {'fake', 'real'}:
            raise ValueError(f'{a} needs both classes')
        for b in main:
            if a >= b:
                continue
            b_records = [r for r in records if r['split'] == b]
            if {r['video_id'] for r in a_records} & {r['video_id'] for r in b_records}:
                raise ValueError(f'video_id leakage between {a}/{b}')
            if disjoint_sources is None:
                groups = [(a_records, b_records)]
            else:
                groups = [([r for r in a_records if source_of(r) == source],
                           [r for r in b_records if source_of(r) == source])
                          for source in disjoint_sources]
            for group_a, group_b in groups:
                if {r['subject_id'] for r in group_a} & {r['subject_id'] for r in group_b}:
                    raise ValueError(f'subject_id leakage between {a}/{b}')
    video_splits, video_groups, crop_splits, crop_labels = {}, {}, {}, {}
    for record in records:
        prior = video_splits.setdefault(record['video_id'], record['split'])
        if prior != record['split']:
            raise ValueError('Video leakage across dataset splits')
        video_metadata = (record['label'], record['subject_id'], source_of(record))
        if video_groups.setdefault(record['video_id'], video_metadata) != video_metadata:
            raise ValueError('Each video must have consistent label, subject, and source')
        crop = record.get('crop_pixel_sha256')
        if crop and crop_splits.setdefault(crop, record['split']) != record['split']:
            raise ValueError('Identical crop pixels occur in different dataset splits')
        if crop and crop_labels.setdefault(crop, record['label']) != record['label']:
            raise ValueError('Identical crop pixels have conflicting labels')


def source_summary(records):
    """Count provenance independently of the supplied manifest summary."""
    summary = {}
    for source in sorted({source_of(r) for r in records}):
        source_records = [r for r in records if source_of(r) == source]
        summary[source] = {}
        for split in sorted({r['split'] for r in source_records}):
            group = [r for r in source_records if r['split'] == split]
            summary[source][split] = {
                'frames': len(group), 'videos': len({r['video_id'] for r in group}),
                'subjects': sorted({r['subject_id'] for r in group}),
                'identity_unknown_frames': sum(bool(r.get('identity_unknown')) for r in group),
                'frames_by_label': dict(Counter(r['label'] for r in group))}
    return summary


def evaluate_groups(probs, records, labels):
    """Evaluate identical observations overall and by their source provenance."""
    result = evaluate_probabilities(probs, records, labels=labels)
    result['per_source'] = {}
    for source in sorted({source_of(record) for record in records}):
        ids = [i for i, record in enumerate(records) if source_of(record) == source]
        result['per_source'][source] = evaluate_probabilities(probs[ids], [records[i] for i in ids], labels=labels)
    return result


def evaluate_splits(model, labels, x, records, out, prefix):
    result = {}
    splits = ['train', 'validation', 'test'] + sorted({r['split'] for r in records} - {'train', 'validation', 'test'})
    for split in splits:
        ids = [i for i, r in enumerate(records) if r['split'] == split]
        if not ids:
            continue
        probs = predictions(model, x[ids])
        np.save(out / f'{prefix}_{split}_probabilities.npy', probs)
        result[split] = evaluate_groups(probs, [records[i] for i in ids], labels)
        print(json.dumps({'phase': prefix, 'split': split, 'frames': len(ids),
                          'loss': result[split]['loss'], 'accuracy': result[split]['accuracy'],
                          'macro_f1': result[split]['classification_report']['macro avg']['f1-score']}), flush=True)
    return result


def video_balanced_weights(records):
    counts = Counter(r['video_id'] for r in records)
    videos_by_class = {label: {r['video_id'] for r in records if r['label'] == label}
                       for label in ('fake', 'real')}
    return np.asarray([len(records) / (2 * len(videos_by_class[r['label']]) * counts[r['video_id']])
                       for r in records], dtype=np.float32)


def plot_history(history, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = np.arange(1, len(history['loss'])+1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history['loss'], label='Train: augmented, weighted')
    axes[0].plot(epochs, history['val_loss'], label='Validation: original frames')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[1].plot(epochs, history['accuracy'], label='Train accuracy')
    axes[1].plot(epochs, history['val_accuracy'], label='Validation accuracy')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.01))
    for axis in axes:
        axis.legend(); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from model.livenessnet import LivenessNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/retrain-20260930')
    parser.add_argument('--output', default='outputs/retrain-20260930')
    parser.add_argument('--baseline-model', default='outputs/retrain-20260915/liveness.h5')
    parser.add_argument('--baseline-labels', default='outputs/retrain-20260915/le.pickle')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.cpu_threads < 1 or args.batch_size < 1:
        parser.error('epochs, batch-size and cpu-threads must be positive')
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    data_dir = ROOT / args.dataset
    manifest_path = data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Only a complete prepared manifest can be trained')
    records = manifest['frames']
    check_split(records, manifest.get('split_policy'))
    if sha(data_dir / 'frames.npz') != manifest['array']['sha256']:
        raise ValueError('Frame archive hash changed after preparation')
    arrays = np.load(data_dir / 'frames.npz')
    if arrays['x'].dtype != np.uint8:
        raise ValueError('Prepared archive must use uint8 BGR pixels')
    x = arrays['x'].astype(np.float32) / 255.0
    if x.shape != (len(records), 32, 32, 3):
        raise ValueError('Expected aligned Nx32x32x3 BGR frames and records')
    if any(r['array_index'] != i for i, r in enumerate(records)):
        raise ValueError('Array row order does not match manifest')
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    started = time.time()
    config = {'seed': 42, 'max_epochs': args.epochs, 'batch_size': args.batch_size,
              'cpu_threads': args.cpu_threads, 'tensorflow_version': tf.__version__,
              'architecture': 'LivenessNet CNN 32x32 BGR', 'input': 'uint8 BGR / 255',
              'optimizer': 'Adam', 'initial_learning_rate': 1e-4,
              'early_stopping': {'monitor': 'val_loss', 'patience': 8},
              'training_weighting': 'Each class equal weight; each video equal weight within class',
              'dataset_manifest_sha256': sha(manifest_path),
              'baseline_model_path': args.baseline_model,
              'baseline_model_sha256': sha(ROOT / args.baseline_model),
              'baseline_labels_sha256': sha(ROOT / args.baseline_labels),
              'test_policy': 'Frozen held-out videos from each data source; not used to choose weights or thresholds',
              'dataset_split_policy': manifest.get('split_policy', {}),
              'baseline_role': 'before: selected baseline model before this merged-data training run',
              'candidate_role': 'after: new CNN trained on the supplied combined dataset',
              'evaluation_loss': 'unweighted original frames in inference mode, no augmentation',
              'confidence_threshold': .8}
    config['training_script_sha256'] = sha(__file__)
    config['architecture_source_sha256'] = sha(ROOT / 'model/livenessnet.py')
    config['deterministic_ops_enabled'] = True
    config['baseline_training'] = baseline_training_metadata(ROOT / args.baseline_model)
    save_json(out / 'configuration.json', config)
    baseline = tf.keras.models.load_model(ROOT / args.baseline_model, compile=False)
    baseline_labels = tuple(pickle.loads((ROOT / args.baseline_labels).read_bytes()).classes_)
    if (baseline.input_shape[1:] != (32, 32, 3) or baseline.output_shape[-1] != 2
            or len(baseline_labels) != 2 or set(baseline_labels) != {'fake', 'real'}):
        raise ValueError('Baseline does not match expected API input/labels')
    before = evaluate_splits(baseline, baseline_labels, x, records, out, 'before')
    save_json(out / 'before.json', before)
    del baseline
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    le = LabelEncoder().fit(['fake', 'real'])
    y = tf.keras.utils.to_categorical(le.transform([r['label'] for r in records]), 2)
    train = np.asarray([i for i,r in enumerate(records) if r['split']=='train'])
    val = np.asarray([i for i,r in enumerate(records) if r['split']=='validation'])
    weights = video_balanced_weights([records[i] for i in train])
    model = LivenessNet.build(32, 32, 3, 2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    aug = ImageDataGenerator(rotation_range=12, zoom_range=.10, width_shift_range=.10,
                            height_shift_range=.10, horizontal_flip=True, fill_mode='nearest')

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            record = {'phase':'training', 'epoch':epoch+1, 'seconds_elapsed':time.time()-started,
                      **{k:float(v) for k,v in (logs or {}).items()}}
            save_json(out / 'progress.json', record)
            print(json.dumps(record), flush=True)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(out/'best.keras'), monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-6),
                 tf.keras.callbacks.CSVLogger(str(out/'training.csv')), Progress()]
    history = model.fit(aug.flow(x[train], y[train], sample_weight=weights,
                                batch_size=args.batch_size, seed=42),
                        validation_data=(x[val], y[val]), epochs=args.epochs,
                        callbacks=callbacks, verbose=2).history
    save_json(out / 'history.json', {k:[float(v) for v in values] for k,values in history.items()})
    best_epoch = int(np.argmin(history['val_loss']))+1
    best = tf.keras.models.load_model(out/'best.keras', compile=False)
    best.save(out/'liveness.h5')
    (out/'le.pickle').write_bytes(pickle.dumps(le))
    # Verify the exported legacy H5 rather than only the in-memory Keras model.
    exported = tf.keras.models.load_model(out/'liveness.h5', compile=False)
    compatibility_ids = np.concatenate((train[:4], val[:4]))
    expected = predictions(best, x[compatibility_ids])
    actual = predictions(exported, x[compatibility_ids])
    if (exported.input_shape[1:] != (32, 32, 3) or exported.output_shape[-1] != 2
            or not np.allclose(expected, actual, rtol=1e-5, atol=1e-6)):
        raise ValueError('Exported H5 does not reproduce the candidate API-compatible predictions')
    config['export_compatibility'] = {'format': 'legacy H5', 'input_shape': [None, 32, 32, 3],
                                     'labels': list(le.classes_), 'checked_frames': len(compatibility_ids),
                                     'max_probability_difference': float(np.max(np.abs(expected-actual))),
                                     'passed': True,
                                     'scope': 'Model serialization/input/output compatibility; not an HTTP or live-camera test'}
    save_json(out / 'configuration.json', config)
    del exported
    after = evaluate_splits(best, tuple(le.classes_), x, records, out, 'after')
    save_json(out/'after.json', after)
    report = {'configuration': config, 'before': before, 'after': after,
              'epochs_run':len(history['loss']), 'best_epoch':best_epoch,
              'history_at_best_epoch':{k:float(v[best_epoch-1]) for k,v in history.items()},
              'history_at_last_epoch':{k:float(v[-1]) for k,v in history.items()},
              'total_seconds':time.time()-started, 'new_model_sha256':sha(out/'liveness.h5'),
              'limitations':['Person groups inferred from supplied folder/name identifiers, not independently verified identities.',
                             'New data has label-dependent recording/codec differences; uniform crops cannot remove acquisition bias.',
                             'Adjacent frames and multiple videos of the same person are correlated.',
                             'Cropped-frame metrics exclude detector/quality/temporal runtime coverage.',
                             'Source-specific subject overlap is reported below; video-held-out old data may not be identity-independent.']}
    report['test_video_paired_bootstrap'] = paired_video_bootstrap(before['test'], after['test'])
    report['test_video_paired_bootstrap_by_source'] = {}
    for source, source_before in before['test']['per_source'].items():
        if {v['true_label'] for v in source_before['per_video']} == {'fake', 'real'}:
            report['test_video_paired_bootstrap_by_source'][source] = paired_video_bootstrap(
                source_before, after['test']['per_source'][source])
    report['subject_overlap_by_source'] = {}
    for source in sorted({source_of(r) for r in records}):
        groups = {split: {r['subject_id'] for r in records if source_of(r) == source and r['split'] == split}
                  for split in ('train', 'validation', 'test')}
        report['subject_overlap_by_source'][source] = {
            f'{a}/{b}': sorted(groups[a] & groups[b])
            for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))}
    report['dataset_summary'] = manifest['summary']
    report['source_split_summary'] = source_summary(records)
    report['dataset_exclusions'] = manifest.get('exclusions', [])
    save_json(out/'evaluation.json', report)
    plot_history(history, out/'learning_curves.png')
    save_json(out/'progress.json', {'phase':'complete','epochs_run':len(history['loss']),
                                   'best_epoch':best_epoch,'seconds_elapsed':time.time()-started})
    print(json.dumps({'phase':'complete','epochs_run':len(history['loss']), 'best_epoch':best_epoch,
                      'before_test_macro_f1':before['test']['classification_report']['macro avg']['f1-score'],
                      'after_test_macro_f1':after['test']['classification_report']['macro avg']['f1-score']}), flush=True)


if __name__ == '__main__':
    main()
