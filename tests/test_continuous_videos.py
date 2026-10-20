"""Check the fixed continuous-video sampling plan and API-style status gate."""
import json
import unittest

import numpy as np

from pipeline import TemporalDecision
from scripts.evaluate_continuous_videos import classify_sample, prefix_indices, prepare_api_input
from scripts.evaluate_retraining_runtime import summarize


def face(status="checking", quality=None, identifier=1):
    return {"box": (10, 10, 90, 70), "detector_score": .98, "quality": quality,
            "track_id": identifier, "status": status, "scores": None}


class FakeEngine:
    def __init__(self, faces):
        self.faces = faces
        self.calls = []

    def process(self, frame, now, temporal):
        self.calls.append((frame.copy(), now, temporal))
        return self.faces


class ContinuousVideoTests(unittest.TestCase):
    def test_existing_sdk_transform_keeps_bgr_and_resizes_both_orientations(self):
        for height, width, expected in ((2000, 1000, [960, 480, 3]),
                                        (1000, 2000, [480, 960, 3])):
            with self.subTest(height=height, width=width):
                frame = np.full((height, width, 3), (25, 90, 180), dtype=np.uint8)
                decoded, metadata = prepare_api_input(frame)
                self.assertEqual(metadata["original_shape"], [height, width, 3])
                self.assertEqual(metadata["processed_shape"], expected)
                self.assertEqual(list(decoded.shape), expected)
                self.assertGreater(metadata["jpeg_bytes"], 0)
                self.assertEqual(len(metadata["jpeg_sha256"]), 64)
                self.assertTrue(np.allclose(decoded[0, 0], [25, 90, 180], atol=3))
                self.assertEqual(frame[0, 0].tolist(), [25, 90, 180])

    def test_prefix_is_fixed_first_twelve_seconds_with_at_most_five_fps(self):
        indices = prefix_indices(1800, 29.97)
        self.assertEqual(len(indices), 60)
        self.assertEqual(indices[:3], [0, 6, 12])
        self.assertEqual(indices[-1], 354)
        self.assertTrue(all(index / 29.97 < 12 for index in indices))
        self.assertEqual(prefix_indices(720, 60), list(range(0, 720, 12)))
        self.assertEqual(prefix_indices(20, 30), [0, 6, 12, 18])
        self.assertEqual(prefix_indices(100, 3), list(range(36)))
        for count, fps in ((0, 30), (1, 0), (1, float("nan")), (1, float("inf"))):
            with self.assertRaises(ValueError):
                prefix_indices(count, fps)

    def test_one_face_rule_preserves_adjust_and_clears_multiple_tracks(self):
        frame = np.full((80, 100, 3), (10, 20, 30), dtype=np.uint8)
        video = {"video_id": "v", "subject_id": "person", "label": "fake"}
        for faces, expected, input_status in (([], "no_face", "no_face"),
                                              ([face("adjust", "Move closer")], "adjust", "adjust"),
                                              ([face("real")], "real", "valid"),
                                              ([face(), face(identifier=2)], "multiple_faces", "multiple_faces")):
            with self.subTest(expected=expected):
                temporal = TemporalDecision()
                temporal.tracks = [object()]
                engine = FakeEngine(faces)
                result = classify_sample(engine, temporal, frame, 12, 30, video)
                self.assertEqual(result["runtime_status"], expected)
                self.assertEqual(result["input_status"], input_status)
                self.assertEqual(result["time_seconds"], .4)
                self.assertEqual(result["shape"], [80, 100, 3])
                self.assertEqual(engine.calls[0][1], .4)
                self.assertIs(engine.calls[0][2], temporal)
                if expected == "multiple_faces":
                    self.assertEqual(temporal.tracks, [])
                json.dumps(result, allow_nan=False)

    def test_all_sample_aggregation_retains_no_decisions(self):
        rows = [
            {"true_label": "fake", "input_status": "valid", "runtime_status": "real"},
            {"true_label": "fake", "input_status": "adjust", "runtime_status": "adjust"},
            {"true_label": "fake", "input_status": "valid", "runtime_status": "checking"},
            {"true_label": "real", "input_status": "no_face", "runtime_status": "no_face"},
            {"true_label": "real", "input_status": "valid", "runtime_status": "real"},
        ]
        result = summarize(rows)
        self.assertEqual(result["all_sample_count"], 5)
        self.assertEqual(result["conclusive_count"], 2)
        self.assertAlmostEqual(result["conclusive_coverage"], .4)
        self.assertEqual(result["correct_conclusive_count"], 1)
        self.assertEqual(result["wrong_conclusive_count"], 1)
        self.assertAlmostEqual(result["fake_as_real_fraction_of_all_fake_samples"], 1 / 3)
        self.assertEqual(result["status_counts"]["no_face"], 1)


if __name__ == "__main__":
    unittest.main()
