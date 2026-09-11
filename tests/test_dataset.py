import tempfile
import unittest
from pathlib import Path

from face_antispoofing.dataset import (
    assert_no_subject_leakage,
    assign_subject_splits,
    parse_video_path,
    stable_sample_id,
)


class DatasetTests(unittest.TestCase):
    def test_parse_live_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = parse_video_path(root, root / "live" / "s01" / "day1" / "cam.mp4")
        self.assertEqual(item.label, 1)
        self.assertEqual(item.subject_id, "s01")
        self.assertEqual(item.session_id, "day1")

    def test_parse_spoof_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = parse_video_path(
                root, root / "spoof" / "replay" / "s02" / "day2" / "phone.mp4"
            )
        self.assertEqual(item.label, 0)
        self.assertEqual(item.attack_type, "replay")

    def test_invalid_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                parse_video_path(root, root / "misc" / "video.mp4")

    def test_subject_splits_are_disjoint_and_deterministic(self) -> None:
        subjects = [f"s{index:02d}" for index in range(20)]
        first = assign_subject_splits(subjects, seed=7)
        second = assign_subject_splits(subjects, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "val", "test"})

    def test_too_few_subjects_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assign_subject_splits(["one", "two"])

    def test_subject_leakage_is_rejected(self) -> None:
        rows = [
            {"subject_id": "s01", "split": "train"},
            {"subject_id": "s01", "split": "test"},
        ]
        with self.assertRaises(ValueError):
            assert_no_subject_leakage(rows)

    def test_sample_id_is_stable_and_frame_specific(self) -> None:
        self.assertEqual(stable_sample_id("a.mp4", 1), stable_sample_id("a.mp4", 1))
        self.assertNotEqual(stable_sample_id("a.mp4", 1), stable_sample_id("a.mp4", 2))


if __name__ == "__main__":
    unittest.main()
