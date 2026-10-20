"""Protect frozen holdouts, label-independent gates, and duplicate quarantine."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import prepare_merged_data as prepare


class MergedPreparationTests(unittest.TestCase):
    def test_uncapped_sampling_keeps_more_than100_frames(self):
        indices = prepare.uncapped_indices(900, 30, .2)
        self.assertEqual(len(indices), 150)
        self.assertEqual(indices[-1], 894)
        self.assertEqual(indices[1] - indices[0], 6)
        for count, fps in ((0, 30), (900, 0), (900, float("nan"))):
            with self.assertRaises(ValueError):
                prepare.uncapped_indices(count, fps, .2)

    def test_quality_rejects_constant_small_dark_glare_and_blurred_crops(self):
        self.assertEqual(prepare.cached_crop_quality(np.zeros((100, 100, 3), np.uint8))[0], "constant_pixels")
        self.assertEqual(prepare.cached_crop_quality(np.zeros((20, 20, 3), np.uint8))[0], "Move closer")
        random = np.random.default_rng(42).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertIsNone(prepare.cached_crop_quality(random)[0])
        dark = random // 10
        self.assertEqual(prepare.cached_crop_quality(dark)[0], "Need more light")
        glare = random.copy()
        glare[:60] = 255
        self.assertEqual(prepare.cached_crop_quality(glare)[0], "Reduce glare")
        blurry = np.full((100, 100, 3), 128, np.uint8)
        blurry[:, :50] = 129
        self.assertEqual(prepare.cached_crop_quality(blurry)[0], "Hold still / improve focus")

    def test_conflicts_quarantine_all_copies_and_remap_rows(self):
        specs = [("shared-split", "train", "fake"), ("shared-split", "test", "fake"),
                 ("shared-label", "train", "fake"), ("shared-label", "train", "real"),
                 ("same", "train", "fake"), ("same", "train", "fake"),
                 ("unique", "validation", "real")]
        records = [{"array_index": i, "crop_pixel_sha256": digest, "split": split, "label": label}
                   for i, (digest, split, label) in enumerate(specs)]
        arrays = [np.full((32, 32, 3), i, np.uint8) for i in range(len(records))]
        accepted, pixels, excluded, remap = prepare.deduplicate_records(records, arrays)
        self.assertEqual(remap, {4: 0, 6: 1})
        self.assertEqual([r["array_index"] for r in accepted], [0, 1])
        self.assertEqual(pixels[0][0, 0, 0], 4)
        self.assertEqual(len(excluded), 5)
        self.assertEqual(sum(r["reason"] == "cross_split_identical_model_pixels" for r in excluded), 2)
        self.assertEqual(sum(r["reason"] == "conflicting_label_identical_model_pixels" for r in excluded), 2)

    def test_legacy_split_keeps_video_boundaries_and_unknown_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            files = {"train": "dataset/fake/fake.mp40.png",
                     "validation": "dataset/real/real-1.mp40.png",
                     "test": "dataset/real/thanh2_real.mp40.png"}
            for relative in files.values():
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"Decoding occurs only during prepare")
            frame_path = base / "frames.json"
            split_path = base / "split.json"
            frame_path.write_text(json.dumps({s: [p.replace("/", "\\")] for s, p in files.items()}))
            split_path.write_text(json.dumps({s: {"videos": [prepare.parse_cache_path(p)[1]]}
                                            for s, p in files.items()}))
            records, _, _ = prepare.collect_legacy(frame_path, split_path, base)
            self.assertEqual([r["split"] for r in records], ["train", "validation", "test"])
            self.assertEqual(records[0]["subject_id"], "unknown:fake")
            self.assertTrue(records[0]["identity_unknown"])
            self.assertEqual(records[2]["subject_id"], "thanh")
            self.assertFalse(records[2]["cached_crop_index_is_video_frame_index"])
            (base / "dataset/fake/untracked.mp40.png").write_bytes(b"untracked")
            with self.assertRaises(ValueError):
                prepare.collect_legacy(frame_path, split_path, base)


if __name__ == "__main__":
    unittest.main()
