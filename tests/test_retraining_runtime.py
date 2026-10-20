"""Offline runtime replay checks, independent of model training."""
import json
import unittest

import numpy as np

from scripts.evaluate_retraining_runtime import replay_test


def fixture(videos):
    # A preceding non-test row catches incorrect saved-array/global-index mapping.
    records = [{"array_index": 0, "split": "train", "label": "real", "video_id": "train", "subject_id": "trainer"}]
    clips, probabilities = [], []
    for number, specification in enumerate(videos):
        events, label = specification["events"], specification.get("label", "fake")
        video_id = "v" + str(number)
        stored = []
        for index, given in enumerate(events):
            event = {"frame_index": index * 2, "time_seconds": index * .2, **given}
            if event["status"] in ("valid", "adjust"):
                event.setdefault("box", [0, 0, 100, 100])
            if event["status"] == "adjust":
                event.setdefault("quality_reason", "Need more light")
            if event["status"] == "valid":
                array_index = len(records)
                event["array_index"] = array_index
                records.append({"array_index": array_index, "split": "test", "label": label,
                                "video_id": video_id, "subject_id": "person" + str(number)})
                probabilities.append(event.pop("probs", [.95, .05]))
            stored.append(event)
        clips.append({"video_id": video_id, "subject_id": "person" + str(number), "label": label,
                      "split": "test", "sample_events": stored})
    return {"frames": records, "videos": clips}, np.asarray(probabilities).reshape(-1, 2)


def valid(count, **fields):
    return [{"status": "valid", **fields} for _ in range(count)]


class RuntimeReplayTests(unittest.TestCase):
    def test_denominators_retain_checking_and_wrong_decisions(self):
        manifest, probabilities = fixture([
            {"label": "fake", "events": valid(7, probs=[.05, .95])},
            {"label": "real", "events": valid(7, probs=[.95, .05])},
        ])
        result = replay_test(manifest, probabilities)
        self.assertEqual(result["all_sample_count"], 14)
        self.assertEqual(result["status_counts"]["checking"], 10)
        self.assertEqual(result["conclusive_count"], 4)
        self.assertEqual(result["wrong_conclusive_count"], 4)
        self.assertEqual(result["fake_as_real_count"], 2)
        self.assertEqual(result["real_as_fake_count"], 2)
        self.assertAlmostEqual(result["fake_as_real_percent_of_all_fake_samples"], 200 / 7)
        self.assertAlmostEqual(result["conclusive_coverage"], 2 / 7)
        self.assertEqual(result["accuracy_among_conclusive"], 0)
        # Global array_index=1 must map to saved test probability row 0.
        self.assertEqual(result["per_video"][0]["decisions"][0]["current_scores"], {"fake": .05, "real": .95})
        json.dumps(result, allow_nan=False)

    def test_quality_rejection_clears_history_and_tracks_reason(self):
        manifest, probabilities = fixture([{"events": valid(6) + [{"status": "adjust"}] + valid(6)}])
        result = replay_test(manifest, probabilities)
        statuses = [d["runtime_status"] for d in result["per_video"][0]["decisions"]]
        self.assertEqual(statuses, ["checking"] * 5 + ["fake", "adjust"] + ["checking"] * 5 + ["fake"])
        self.assertEqual(result["input_rejection_counts"], {"adjust: Need more light": 1})
        self.assertAlmostEqual(result["quality_valid_coverage"], 12 / 13)
        self.assertAlmostEqual(result["conclusive_coverage"], 2 / 13)

    def test_geometry_and_missing_or_multiple_face_reset(self):
        for reset in ("geometry", "no_face", "multiple_faces"):
            with self.subTest(reset=reset):
                if reset == "geometry":
                    events = valid(6) + valid(6, box=[150, 0, 250, 100])
                    expected = ["checking"] * 5 + ["fake"] + ["checking"] * 5 + ["fake"]
                else:
                    events = valid(6) + [{"status": reset}] + valid(6)
                    expected = ["checking"] * 5 + ["fake", reset] + ["checking"] * 5 + ["fake"]
                manifest, probabilities = fixture([{"events": events}])
                result = replay_test(manifest, probabilities)
                decisions = result["per_video"][0]["decisions"]
                self.assertEqual([d["runtime_status"] for d in decisions], expected)
                self.assertNotEqual(decisions[0]["track_id"], decisions[-1]["track_id"])

    def test_long_gap_and_opposing_current_prediction(self):
        events = valid(6) + [{"status": "valid", "time_seconds": 1.6 + index * .2} for index in range(6)]
        manifest, probabilities = fixture([{"events": events}])
        result = replay_test(manifest, probabilities)
        self.assertEqual([d["runtime_status"] for d in result["per_video"][0]["decisions"]],
                         ["checking"] * 5 + ["fake"] + ["checking"] * 5 + ["fake"])
        manifest, probabilities = fixture([{"events": valid(6) + valid(1, probs=[.05, .95])}])
        result = replay_test(manifest, probabilities)
        self.assertEqual(result["per_video"][0]["decisions"][-1]["runtime_status"], "uncertain")

    def test_reversed_label_order_and_zero_accepted_clip(self):
        manifest, probabilities = fixture([
            {"events": valid(6, probs=[.05, .95])},
            {"label": "real", "events": [{"status": "no_face"}] * 4},
        ])
        result = replay_test(manifest, probabilities, labels=("real", "fake"))
        self.assertEqual(result["correct_conclusive_count"], 1)
        self.assertEqual(result["all_sample_count"], 10)
        self.assertEqual(result["per_true_class"]["real"]["all_sample_count"], 4)
        self.assertIsNone(result["per_true_class"]["real"]["accuracy_among_conclusive"])
        self.assertEqual(result["real_as_fake_fraction_of_all_real_samples"], 0)

    def test_manifest_prediction_and_chronology_errors_are_rejected(self):
        manifest, probabilities = fixture([{"events": valid(2)}])
        with self.assertRaises(ValueError):
            replay_test(manifest, probabilities[:1])
        wrong = probabilities.copy()
        wrong[0] = [np.nan, .5]
        with self.assertRaises(ValueError):
            replay_test(manifest, wrong)
        manifest["videos"][0]["sample_events"][1]["time_seconds"] = 0
        with self.assertRaises(ValueError):
            replay_test(manifest, probabilities)
        manifest, probabilities = fixture([{"events": valid(2)}])
        manifest["frames"][1]["video_id"] = "different"
        with self.assertRaises(ValueError):
            replay_test(manifest, probabilities)


if __name__ == "__main__":
    unittest.main()
