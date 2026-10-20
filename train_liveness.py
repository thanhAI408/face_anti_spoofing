"""Train with source-video-disjoint train/validation/test sets; preserve deployed weights."""
import argparse
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.livenessnet import LivenessNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/retraining-video-split')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    paths = sorted(p for p in Path('dataset').rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    groups = np.array([p.name.split('.mp4')[0].split('__frame_')[0] for p in paths])
    raw_labels = np.array([p.parent.name for p in paths])
    unique = np.unique(groups)
    group_labels = []
    for group in unique:
        labels = np.unique(raw_labels[groups == group])
        if len(labels) != 1:
            raise ValueError(f'Ambiguous source video: {group}')
        group_labels.append(labels[0])
    train_groups, held_groups = train_test_split(unique, test_size=0.3, stratify=group_labels, random_state=42)
    held_labels = [raw_labels[np.flatnonzero(groups == g)[0]] for g in held_groups]
    val_groups, test_groups = train_test_split(held_groups, test_size=0.5, stratify=held_labels, random_state=42)
    split_groups = dict(train=train_groups, validation=val_groups, test=test_groups)
    assert not (set(train_groups) & set(val_groups) or set(train_groups) & set(test_groups) or set(val_groups) & set(test_groups))
    le = LabelEncoder().fit(raw_labels)
    y = le.transform(raw_labels)
    x = np.empty((len(paths), 32, 32, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(f'Unreadable image: {path}')
        x[i] = cv2.resize(im, (32, 32)) / 255.0
    indices = {name: np.flatnonzero(np.isin(groups, gs)) for name, gs in split_groups.items()}
    manifest = {name: {'videos': gs.tolist(), 'frames': len(indices[name]), 'classes': {label: int(np.sum(raw_labels[indices[name]] == label)) for label in le.classes_}} for name, gs in split_groups.items()}
    (out / 'split.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'frame_manifest.json').write_text(json.dumps({name: [str(paths[i]) for i in ids] for name, ids in indices.items()}, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)
    train, val, test = (indices[n] for n in ['train', 'validation', 'test'])
    model = LivenessNet.build(32, 32, 3, 2)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    aug = ImageDataGenerator(rotation_range=20, zoom_range=.15, width_shift_range=.2, height_shift_range=.2, shear_range=.15, horizontal_flip=True, fill_mode='nearest')
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-6), tf.keras.callbacks.CSVLogger(str(out / 'training.csv'))]
    hist = model.fit(aug.flow(x[train], tf.keras.utils.to_categorical(y[train], 2), batch_size=32, seed=42), validation_data=(x[val], tf.keras.utils.to_categorical(y[val], 2)), epochs=args.epochs, callbacks=callbacks, verbose=2)
    model = tf.keras.models.load_model(out / 'best.keras', compile=False)
    model.save(out / 'liveness.h5')
    (out / 'le.pickle').write_bytes(pickle.dumps(le))
    (out / 'history.json').write_text(json.dumps(hist.history), encoding='utf-8')
    def evaluate(m, classes):
        probabilities = m.predict(x[test], batch_size=128, verbose=0)
        prediction = np.array([str(classes[i]) for i in probabilities.argmax(axis=1)])
        truth = raw_labels[test]
        video_results = []
        for group in test_groups:
            mask = groups[test] == group
            label = str(classes[probabilities[mask].mean(axis=0).argmax()])
            video_results.append({'video': group, 'truth': str(truth[mask][0]), 'prediction': label, 'frames': int(mask.sum())})
        return {'classification_report': classification_report(truth, prediction, labels=le.classes_, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(truth, prediction, labels=le.classes_).tolist(), 'label_order': le.classes_.tolist(), 'video_mean_probability_results': video_results}
    report = {'new_model': evaluate(model, le.classes_), 'epochs_run': len(hist.history['loss']), 'best_epoch': int(np.argmin(hist.history['val_loss'])) + 1, 'limitations': 'Video-disjoint, not person-disjoint. Old model may have trained on these test frames; baseline is not independent. Frame classification excludes detector/quality/temporal pipeline.'}
    if Path('liveness.h5').exists():
        old = tf.keras.models.load_model('liveness.h5', compile=False)
        old_labels = pickle.loads(Path('le.pickle').read_bytes()).classes_
        report['old_model_non_independent'] = evaluate(old, old_labels)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
