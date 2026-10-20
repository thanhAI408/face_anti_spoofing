"""Boundary and extraction checks for the retraining manifest."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts import prepare_retraining_data as prepare


class SourceTests(unittest.TestCase):
    def test_sampling_spans_video_and_respects_minimum_spacing(self):
        result = prepare.sampling_indices(1000, 29.97, .2, 100)
        self.assertEqual(len(result), 100)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 996)
        self.assertGreaterEqual(min(np.diff(result)), 6)
        self.assertEqual(prepare.sampling_indices(9, 10, .2, 3), [0, 4, 8])
        for count, fps in ((0, 30), (10, 0), (10, float("nan"))):
            with self.assertRaises(ValueError):
                prepare.sampling_indices(count, fps, .2, 100)

    def test_old_identity_excludes_generic_names_and_unifies_thanh(self):
        for name in ("fake", "real", "gia", "that", "real-1", "fake-1", "someone_real"):
            self.assertIsNone(prepare.old_identity(name))
        self.assertEqual(prepare.old_identity("thanh2_real"), ("thanh", "real"))
        self.assertEqual(prepare.old_identity("fake_tuyen"), ("tuyen", "fake"))
        self.assertEqual(prepare.old_identity("bang_fake_962025"), ("bang", "fake"))

    def test_new_person_split_and_unsafe_source_path(self):
        self.assertEqual(prepare.canonical_subject(" Phúc "), "phuc")
        self.assertEqual(prepare.new_split("nga"), "test")
        self.assertEqual(prepare.new_split("thu"), "validation")
        self.assertEqual(prepare.new_split("a"), "train")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertEqual(prepare.safe_source_path("videos/A.mp4", base), base / "videos/A.mp4")
            with self.assertRaises(ValueError):
                prepare.safe_source_path("../elsewhere.mp4", base)
            with self.assertRaises(ValueError):
                prepare.safe_source_path(str(base / "absolute.mp4"), base)

    def test_duplicate_video_and_conflicting_label_or_split(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "first.mp4").write_bytes(b"same video bytes")
            (base / "second.mp4").write_bytes(b"same video bytes")
            first = {"video_id": "first", "relative_path": "first.mp4", "label": "fake", "split": "train"}
            second = {"video_id": "second", "relative_path": "second.mp4", "label": "fake", "split": "train"}
            unique, excluded = prepare.deduplicate_videos([first.copy(), second.copy()], base)
            self.assertEqual(len(unique), 1)
            self.assertEqual(excluded[0]["reason"], "duplicate_video_sha256")
            for field, value in (("label", "real"), ("split", "test")):
                wrong = {**second, field: value}
                with self.assertRaises(ValueError):
                    prepare.deduplicate_videos([first.copy(), wrong], base)
            with self.assertRaises(FileNotFoundError):
                prepare.deduplicate_videos([{**first, "relative_path": "missing.mp4"}], base)
            with self.assertRaises(ValueError):
                prepare.deduplicate_videos([{**first, "expected_bytes": 999}], base)

    def test_inventory_assigns_person_holdouts_and_old_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "videos").mkdir()
            for name in ("thanh2_real", "thanh_real_213", "fake", "bang_fake_962025"):
                (base / "videos" / (name + ".mp4")).write_bytes(b"not decoded here")
            inventory = base / "inventory.json"
            inventory.write_text(json.dumps({"files": [
                {"id": "test", "label": "real", "subject_id": "Phúc", "relative_path": "videos/new1.mp4"},
                {"id": "validation", "label": "fake", "subject_id": "B", "relative_path": "videos/new2.mp4"},
                {"id": "training", "label": "fake", "subject_id": "A", "relative_path": "videos/new3.mp4"},
            ]}), encoding="utf-8")
            split = base / "split.json"
            split.write_text(json.dumps({"test": {"videos": ["thanh2_real"]}}), encoding="utf-8")
            _, sources, exclusions = prepare.collect_sources(inventory, split, base)
            by_id = {source["video_id"]: source for source in sources}
            self.assertEqual(by_id["drive:test"]["split"], "test")
            self.assertEqual(by_id["drive:validation"]["split"], "validation")
            self.assertEqual(by_id["drive:training"]["split"], "train")
            self.assertEqual(by_id["old:thanh2_real"]["split"], "regression_old")
            self.assertEqual(by_id["old:thanh2_real"]["subject_id"], "thanh")
            self.assertEqual(by_id["old:thanh_real_213"]["split"], "train")
            self.assertEqual(exclusions[0]["relative_path"], "videos/fake.mp4")


class FakeCapture:
    def __init__(self, value):
        self.value = value
        self.index = -1

    def isOpened(self):
        return True

    def set(self, key, value):
        return key == cv2.CAP_PROP_ORIENTATION_AUTO and value == 1

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 9,
                cv2.CAP_PROP_FRAME_WIDTH: 120, cv2.CAP_PROP_FRAME_HEIGHT: 100,
                cv2.CAP_PROP_ORIENTATION_META: 90, cv2.CAP_PROP_ORIENTATION_AUTO: 1}.get(key, 0)

    def getBackendName(self):
        return "FAKE_TEST_BACKEND"

    def grab(self):
        self.index += 1
        return self.index < 9

    def read(self):
        self.index += 1
        if self.index >= 9:
            return False, None
        frame = np.full((100, 120, 3), (self.value + self.index, 20, 30), dtype=np.uint8)
        return True, frame

    def release(self):
        pass


class FakeDetector:
    def detect(self, frame):
        return [{"box": (10, 10, 90, 90), "detector_score": .99}]


class ExtractionTests(unittest.TestCase):
    def test_manifest_pixels_events_and_sequential_sampling_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "face_detector").mkdir()
            (base / "face_detector/face_detection_yunet_2023mar.onnx").write_bytes(b"model metadata only")
            (base / "pipeline.py").write_text("test metadata", encoding="utf-8")
            inventory = base / "inventory.json"
            inventory.write_text("{}", encoding="utf-8")
            sources = []
            for number, (split, label) in enumerate((s, l) for s in prepare.SPLITS for l in ("fake", "real")):
                name = f"clip{number}.mp4"
                (base / name).write_bytes(f"unique file {number}".encode())
                sources.append({"video_id": str(number), "relative_path": name,
                                "label": label, "subject_id": "person" + str(number),
                                "source": "test_fixture", "split": split})
            args = Namespace(inventory=inventory, old_split=base / "not_read.json",
                             output=base / "prepared", confidence=.85, min_interval=.2, max_per_video=3)
            def capture_factory(path):
                number = int(Path(path).stem[4:])
                return FakeCapture(40 + number * 20)
            with patch.object(prepare, "BASE", base), \
                    patch.object(prepare, "collect_sources", return_value=({}, sources, [])), \
                    patch.object(prepare, "FaceDetector", return_value=FakeDetector()), \
                    patch.object(prepare, "quality", return_value=None), \
                    patch.object(prepare.cv2, "VideoCapture", side_effect=capture_factory), \
                    contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare(args)
            manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            array = np.load(args.output / "frames.npz")["x"]
            self.assertEqual(array.shape, (24, 32, 32, 3))
            self.assertEqual(array.dtype, np.uint8)
            self.assertEqual(array[0, 0, 0].tolist(), [40, 20, 30])
            self.assertEqual(array[1, 0, 0].tolist(), [44, 20, 30])
            self.assertEqual(manifest["status"], "complete")
            for video in manifest["videos"]:
                events = video["sample_events"]
                self.assertEqual([event["frame_index"] for event in events], [0, 4, 8])
                self.assertEqual([event["time_seconds"] for event in events], [0, .4, .8])
                self.assertTrue(all(event["status"] == "valid" for event in events))
                self.assertEqual(video["accepted_frames"], 3)
                self.assertTrue(video["orientation_auto_active"])
                self.assertTrue((args.output / video["preview"]["frame"]).is_file())
                for event in events:
                    record = manifest["frames"][event["array_index"]]
                    self.assertEqual(record["video_id"], video["video_id"])
                    self.assertEqual(record["frame_index"], event["frame_index"])
            self.assertEqual(manifest["summary"]["test"]["classes"], {"fake": 3, "real": 3})


if __name__ == "__main__":
    unittest.main()
