"""Training provenance and grouped split invariants, without fitting a CNN."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.train_retraining_comparison import (
    baseline_training_metadata, check_split, evaluate_groups, source_summary, video_balanced_weights,
)


def record(label, video, split, subject, source='legacy-cache', **extra):
    return {'label': label, 'video_id': video, 'split': split,
            'subject_id': subject, 'source': source, **extra}


def complete_records():
    return [record(label, f'{split}-{label}', split, f'{split}-person')
            for split in ('train', 'validation', 'test') for label in ('fake', 'real')]


class TrainingProvenanceTests(unittest.TestCase):
    def test_selected_baseline_directory_supplies_epochs_and_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'retrain-20260930'
            run.mkdir()
            model = run / 'liveness.h5'
            model.write_bytes(b'selected-September-30-model')
            history = {'loss': [.8, .6, .5], 'val_loss': [.4, .2, .3]}
            evaluation = {'epochs_run': 3, 'best_epoch': 2,
                          'new_model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
            (run / 'evaluation.json').write_text(json.dumps(evaluation), encoding='utf-8')
            (run / 'history.json').write_text(json.dumps(history), encoding='utf-8')
            metadata = baseline_training_metadata(model)
            self.assertEqual(metadata['epochs_run'], 3)
            self.assertEqual(metadata['best_epoch'], 2)
            self.assertEqual(metadata['historical_augmented_train_loss_at_best'], .6)
            self.assertEqual(metadata['historical_validation_loss_at_best'], .2)
            self.assertEqual(Path(metadata['run_directory']), run)
            self.assertTrue(metadata['model_hash_verified_by_metadata'])
            evaluation['new_model_sha256'] = 'different-model'
            (run / 'evaluation.json').write_text(json.dumps(evaluation), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'model hash'):
                baseline_training_metadata(model)
            evaluation['new_model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
            evaluation['best_epoch'] = 3
            (run / 'evaluation.json').write_text(json.dumps(evaluation), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'validation-loss history'):
                baseline_training_metadata(model)

    def test_missing_baseline_metadata_is_explicit_without_inventing_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'liveness.h5'
            model.write_bytes(b'model')
            metadata = baseline_training_metadata(model)
            self.assertEqual(metadata['status'], 'unavailable')
            self.assertIsNone(metadata['epochs_run'])
            self.assertIsNone(metadata['best_epoch'])

    def test_new_identity_disjoint_policy_preserves_video_held_out_old_data(self):
        records = complete_records()
        records[-1]['subject_id'] = records[0]['subject_id']
        with self.assertRaisesRegex(ValueError, 'subject_id leakage'):
            check_split(records)
        policy = {'subject_disjoint_sources': ['drive-20260930']}
        records.append(record('fake', 'new-validation', 'validation', 'new-person', 'drive-20260930'))
        check_split(records, policy)
        records[0]['source'] = records[5]['source'] = 'drive-20260930'
        with self.assertRaisesRegex(ValueError, 'subject_id leakage'):
            check_split(records, policy)

    def test_video_or_exact_crop_cross_split_is_always_rejected(self):
        policy = {'subject_disjoint_sources': ['drive-20260930']}
        records = complete_records()
        records.append(record('fake', 'new-validation', 'validation', 'new-person', 'drive-20260930'))
        records[-1]['video_id'] = records[0]['video_id']
        with self.assertRaisesRegex(ValueError, 'video_id leakage'):
            check_split(records, policy)
        records = complete_records()
        records.append(record('fake', 'new-validation', 'validation', 'new-person', 'drive-20260930'))
        records[0]['crop_pixel_sha256'] = records[-1]['crop_pixel_sha256'] = 'same-pixels'
        with self.assertRaisesRegex(ValueError, 'crop pixels'):
            check_split(records, policy)

    def test_source_policy_must_name_an_existing_source(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            check_split(complete_records(), {'subject_disjoint_sources': ['mistyped-source']})

    def test_source_summary_counts_held_out_old_and_new_videos(self):
        records = [record('real', 'old-r', 'test', 'unknown:old-r', identity_unknown=True),
                   record('real', 'old-r', 'test', 'unknown:old-r', identity_unknown=True),
                   record('fake', 'new-f', 'test', 'new', 'drive-20260930')]
        summary = source_summary(records)
        self.assertEqual(summary['legacy-cache']['test']['frames'], 2)
        self.assertEqual(summary['legacy-cache']['test']['videos'], 1)
        self.assertEqual(summary['legacy-cache']['test']['identity_unknown_frames'], 2)
        self.assertEqual(summary['drive-20260930']['test']['frames_by_label'], {'fake': 1})

    def test_per_source_metrics_preserve_combined_denominators(self):
        records = [record('fake', 'old-f', 'test', 'old'),
                   record('real', 'old-r', 'test', 'old'),
                   record('fake', 'new-f', 'test', 'new', 'drive-20260930'),
                   record('real', 'new-r', 'test', 'new', 'drive-20260930')]
        probabilities = np.asarray([[.9, .1], [.1, .9], [.1, .9], [.1, .9]])
        metrics = evaluate_groups(probabilities, records, ('fake', 'real'))
        self.assertEqual(metrics['frame_count'], 4)
        self.assertEqual(metrics['accuracy'], .75)
        self.assertEqual(metrics['per_source']['legacy-cache']['accuracy'], 1)
        self.assertEqual(metrics['per_source']['drive-20260930']['accuracy'], .5)
        self.assertEqual(sum(group['frame_count'] for group in metrics['per_source'].values()), 4)

    def test_long_videos_and_class_imbalance_do_not_dominate_training_weight(self):
        records = ([record('fake', 'long-f', 'train', 'a')] * 8
                   + [record('fake', 'short-f', 'train', 'b')]
                   + [record('real', 'r', 'train', 'c')] * 3)
        weights = video_balanced_weights(records)
        self.assertAlmostEqual(float(weights[:8].sum()), float(weights[8]), places=6)
        self.assertAlmostEqual(float(weights[:9].sum()), float(weights[9:].sum()), places=6)
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)


if __name__ == '__main__':
    unittest.main()
