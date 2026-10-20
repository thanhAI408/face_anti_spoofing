"""Small deterministic protocol checks; no TensorFlow or model fitting."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.train_balanced_merged import (
    BalancedVideoScheduler, canonical_sha, exposure_summary, groups_and_videos,
    load_train_validation_pixels, source_class_video_loss, source_class_video_weights,
    warmup_schedule,
)


def records(split='train'):
    result = []
    # Deliberately uneven lengths and video counts, resembling old/new mixtures.
    for source, lengths in [('old', {'fake': [2, 9], 'real': [7]}),
                            ('new', {'fake': [3], 'real': [1, 4, 6]})]:
        for label, counts in lengths.items():
            for video, count in enumerate(counts):
                result.extend({'split': split, 'source': source, 'label': label,
                               'video_id': f'{source}-{label}-{video}', 'subject_id': f'{source}-{video}'}
                              for _ in range(count))
    return result


class BalancedMergedProtocolTests(unittest.TestCase):
    def test_every_batch_contains_equal_source_class_counts(self):
        rows = records()
        schedule = BalancedVideoScheduler(rows, 32, steps=11).next_epoch()
        for batch in schedule:
            counts = Counter((rows[int(i)]['source'], rows[int(i)]['label']) for i in batch)
            self.assertEqual(sorted(counts.values()), [8, 8, 8, 8])

    def test_draws_are_valid_and_epoch_has_fixed_number_of_steps(self):
        rows = records()
        sampler = BalancedVideoScheduler(rows, 8, steps=7)
        schedule = sampler.next_epoch()
        self.assertEqual(schedule.shape, (7, 8))
        self.assertEqual(schedule.dtype, np.int64)
        self.assertTrue((schedule >= 0).all())
        self.assertTrue((schedule < len(rows)).all())

    def test_video_counts_are_uniform_within_each_group(self):
        rows = records()
        sampler = BalancedVideoScheduler(rows, 8, steps=17)
        draws = np.concatenate([sampler.next_epoch().ravel() for _ in range(3)])
        frequencies = Counter(rows[int(i)]['video_id'] for i in draws)
        for videos in sampler.groups.values():
            counts = [frequencies[video] for video in videos]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_video_frame_queue_is_exhausted_before_reuse(self):
        rows = records()
        sampler = BalancedVideoScheduler(rows, 4, steps=35)
        draws = sampler.next_epoch().ravel()
        for video, ids in sampler.video_indices.items():
            selected = [int(i) for i in draws if rows[int(i)]['video_id'] == video]
            for offset in range(0, len(selected) - len(ids) + 1, len(ids)):
                self.assertEqual(set(selected[offset:offset + len(ids)]), set(ids.tolist()))

    def test_frame_cycles_persist_across_epoch_boundary(self):
        rows = records()
        sampler = BalancedVideoScheduler(rows, 4, steps=1)
        first = sampler.next_epoch().ravel()
        second = sampler.next_epoch().ravel()
        # Single-video new/fake has three frames; consecutive epochs must use
        # distinct frames, rather than restarting the same shuffle each epoch.
        video = 'new-fake-0'
        ids = [int(i) for i in np.concatenate((first, second)) if rows[int(i)]['video_id'] == video]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(sampler.epochs_generated, 2)

    def test_identical_seed_reproduces_multiple_epochs_exactly(self):
        a, b = [BalancedVideoScheduler(records(), 8, steps=10, seed=42) for _ in range(2)]
        for _ in range(4):
            np.testing.assert_array_equal(a.next_epoch(), b.next_epoch())

    def test_different_seed_changes_draw_order(self):
        a = BalancedVideoScheduler(records(), 8, steps=10, seed=42)
        b = BalancedVideoScheduler(records(), 8, steps=10, seed=43)
        self.assertFalse(np.array_equal(a.next_epoch(), b.next_epoch()))

    def test_warmup_visits_every_frame_once_and_is_reproducible(self):
        rows = records()
        first = warmup_schedule(len(rows), 42)
        self.assertEqual(sorted(first.tolist()), list(range(len(rows))))
        np.testing.assert_array_equal(first, warmup_schedule(len(rows), 42))
        with self.assertRaisesRegex(ValueError, 'positive'):
            warmup_schedule(0, 42)

    def test_exposure_distinguishes_warmup_and_balanced_coverage(self):
        rows = records()
        draws = BalancedVideoScheduler(rows, 4, steps=1).next_epoch().ravel()
        counts = np.bincount(draws, minlength=len(rows))
        balanced = exposure_summary(rows, counts)
        all_used = exposure_summary(rows, counts + 1)
        self.assertLess(balanced['fraction_exposed'], 1)
        self.assertEqual(all_used['fraction_exposed'], 1)
        self.assertEqual(all_used['draws'], len(rows) + len(draws))
        self.assertEqual(sum(g['draws'] for g in balanced['groups'].values()), len(draws))

    def test_missing_source_class_or_inconsistent_video_is_rejected(self):
        rows = [r for r in records() if not (r['source'] == 'new' and r['label'] == 'fake')]
        with self.assertRaisesRegex(ValueError, 'Each source'):
            BalancedVideoScheduler(rows)
        rows = records()
        rows[-1]['video_id'] = rows[0]['video_id']
        with self.assertRaisesRegex(ValueError, 'different source/class'):
            BalancedVideoScheduler(rows)

    def test_invalid_batch_and_steps_are_rejected(self):
        for size in (0, 3, 6, 31):
            with self.assertRaisesRegex(ValueError, 'multiple of four'):
                BalancedVideoScheduler(records(), size)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            BalancedVideoScheduler(records(), steps=0)

    def test_nontrain_sampling_and_wrong_validation_split_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'train'):
            BalancedVideoScheduler(records('test'))
        with self.assertRaisesRegex(ValueError, 'validation'):
            source_class_video_weights(records('train'))

    def test_validation_weights_equalize_groups_and_videos(self):
        rows = records('validation')
        weights = source_class_video_weights(rows)
        groups, videos = groups_and_videos(rows, 'validation')
        self.assertAlmostEqual(weights.sum(), len(rows))
        for video_ids in groups.values():
            masses = [weights[videos[video]].sum() for video in video_ids]
            for mass in masses:
                self.assertAlmostEqual(mass, len(rows) / (4 * len(video_ids)))

    def test_selection_loss_matches_explicit_group_video_average(self):
        rows = records('validation')
        groups, videos = groups_and_videos(rows, 'validation')
        probabilities = np.empty((len(rows), 2))
        for video_number, (video, ids) in enumerate(sorted(videos.items())):
            correct = .20 + .10 * video_number
            for index in ids:
                target = 0 if rows[int(index)]['label'] == 'fake' else 1
                probabilities[index, target] = correct
                probabilities[index, 1 - target] = 1 - correct
        expected = np.mean([np.mean([-np.log(probabilities[videos[v], 0 if rows[int(videos[v][0])]['label'] == 'fake' else 1]).mean()
                                    for v in video_ids]) for video_ids in groups.values()])
        self.assertAlmostEqual(source_class_video_loss(probabilities, rows), expected)
        with self.assertRaisesRegex(ValueError, 'normalized'):
            source_class_video_loss(np.ones((len(rows), 2)), rows)

    def test_archive_loader_constructs_only_train_validation_pixels(self):
        rows = [{'split': 'train'}, {'split': 'test'}, {'split': 'validation'}]
        pixels = np.stack([np.full((32, 32, 3), value, np.uint8) for value in (17, 239, 48)])
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'frames.npz'
            np.savez_compressed(archive, x=pixels)
            arrays, ids = load_train_validation_pixels(archive, rows)
            self.assertEqual(set(arrays), {'train', 'validation'})
            self.assertEqual(ids, {'train': [0], 'validation': [2]})
            np.testing.assert_array_equal(arrays['train'][0], pixels[0])
            np.testing.assert_array_equal(arrays['validation'][0], pixels[2])
            np.savez_compressed(archive, x=pixels.astype(np.float32))
            with self.assertRaisesRegex(ValueError, 'uint8'):
                load_train_validation_pixels(archive, rows)

    def test_configuration_hash_is_order_independent_and_detects_changes(self):
        self.assertEqual(canonical_sha({'seed': 42, 'epochs': 50}), canonical_sha({'epochs': 50, 'seed': 42}))
        self.assertNotEqual(canonical_sha({'seed': 42}), canonical_sha({'seed': 43}))
        with self.assertRaises(ValueError):
            canonical_sha({'invalid': float('nan')})


if __name__ == '__main__':
    unittest.main()
