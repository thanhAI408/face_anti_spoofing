"""Synthetic checks of loss, grouped weighting, rates, and abstentions."""

import json
import math
import unittest

import numpy as np

from scripts.retraining_metrics import evaluate_probabilities, paired_video_bootstrap


def record(label, video, subject="person", split="test"):
    return {"label": label, "video_id": video, "subject_id": subject, "split": split}


class RetrainingMetricsTests(unittest.TestCase):
    def test_loss_class_metrics_and_attack_rate_denominators(self):
        probabilities = [[.9, .1], [.2, .8], [.1, .9], [.8, .2], [.7, .3]]
        records = [record("fake", "f1"), record("fake", "f2"),
                   record("real", "r1"), record("real", "r2"), record("real", "r3")]
        result = evaluate_probabilities(probabilities, records)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [2, 1]])
        self.assertAlmostEqual(result["accuracy"], 2 / 5)
        self.assertAlmostEqual(result["loss"], -sum(map(math.log, [.9, .2, .9, .2, .3])) / 5)
        report = result["classification_report"]
        self.assertAlmostEqual(report["fake"]["precision"], 1 / 3)
        self.assertAlmostEqual(report["fake"]["recall"], 1 / 2)
        self.assertAlmostEqual(report["fake"]["f1-score"], .4)
        self.assertAlmostEqual(report["real"]["precision"], 1 / 2)
        self.assertAlmostEqual(report["real"]["recall"], 1 / 3)
        self.assertAlmostEqual(report["macro avg"]["f1-score"], .4)
        self.assertAlmostEqual(report["weighted avg"]["f1-score"], .4)
        self.assertEqual(report["fake"]["support"], 2)
        rates = result["presentation_attack_rates"]
        self.assertAlmostEqual(rates["APCER"], 1 / 2)
        self.assertAlmostEqual(rates["BPCER"], 2 / 3)
        self.assertAlmostEqual(rates["ACER"], 7 / 12)
        json.dumps(result, allow_nan=False)

    def test_long_video_does_not_dominate_equal_video_weight_or_video_decision(self):
        probabilities = [[.9, .1]] * 9 + [[.4, .6], [.2, .8]]
        records = [record("fake", "long", "alice")] * 10 + [record("fake", "short", "bob")]
        result = evaluate_probabilities(probabilities, records)
        self.assertAlmostEqual(result["accuracy"], 9 / 11)
        self.assertAlmostEqual(result["equal_video_weight_accuracy"], .45)
        self.assertAlmostEqual(result["video_level"]["accuracy"], .5)
        self.assertEqual(result["video_level"]["confusion_matrix"], [[1, 1], [0, 0]])
        self.assertEqual(result["per_video"][0]["predicted_label"], "fake")
        self.assertAlmostEqual(result["per_video"][0]["mean_probabilities"]["fake"], .85)
        self.assertAlmostEqual(result["video_level"]["loss"], -(math.log(.85) + math.log(.2)) / 2)
        subjects = {item["subject_id"]: item for item in result["per_subject"]}
        self.assertAlmostEqual(subjects["alice"]["accuracy"], .9)
        self.assertEqual(subjects["bob"]["frame_count"], 1)
        self.assertIsNone(result["presentation_attack_rates"]["BPCER"])
        self.assertIsNone(result["presentation_attack_rates"]["ACER"])

    def test_confidence_retains_uncertain_and_boundary_cases(self):
        probabilities = [[.8, .2], [.2, .8], [.51, .49], [.49, .51], [.1, .9]]
        records = [record("fake", "f1"), record("fake", "f2"), record("fake", "f3"),
                   record("real", "r1"), record("real", "r2")]
        result = evaluate_probabilities(probabilities, records)["confident_acceptance"]
        self.assertEqual(result["confident_count"], 3)
        self.assertEqual(result["uncertain_count"], 2)
        self.assertAlmostEqual(result["coverage"], .6)
        self.assertAlmostEqual(result["confident_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["error_rate_among_confident"], 1 / 3)
        self.assertAlmostEqual(result["error_fraction_of_all"], .2)
        self.assertEqual(result["confusion_matrix_with_uncertain"], [[1, 1, 1], [0, 1, 1]])
        self.assertEqual(result["uncertain_by_true_label"], {"fake": 1, "real": 1})
        self.assertAlmostEqual(result["fake_as_real_fraction_of_all_fake"], 1 / 3)
        none_confident = evaluate_probabilities([[.5, .5]], [record("real", "r")])["confident_acceptance"]
        self.assertEqual(none_confident["uncertain_count"], 1)
        self.assertIsNone(none_confident["confident_accuracy"])

    def test_explicit_label_order_and_zero_one_clipping(self):
        result = evaluate_probabilities([[0, 1], [1, 0]],
                                        [record("fake", "f"), record("real", "r")],
                                        labels=("real", "fake"))
        self.assertEqual(result["label_order"], ["real", "fake"])
        self.assertEqual(result["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertAlmostEqual(result["loss"], -math.log(1 - 1e-7))
        wrong = evaluate_probabilities([[0, 1]], [record("fake", "f")])
        self.assertAlmostEqual(wrong["loss"], -math.log(1e-7))
        self.assertEqual(wrong["classification_report"]["real"]["support"], 0)
        json.dumps(wrong, allow_nan=False)

    def test_invalid_arrays_and_manifest_are_rejected(self):
        valid = [record("fake", "f")]
        arrays = [[], [[.2, .2]], [[np.nan, 1]], [[np.inf, 0]],
                  [[-.1, 1.1]], [[.1, .2, .7]], [.2, .8], [[1, 0], [1, 0]]]
        for probabilities in arrays:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                evaluate_probabilities(probabilities, valid)
        for labels in (("fake", "fake"), ("genuine", "fake"), ("fake",)):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_probabilities([[1, 0]], valid, labels=labels)
        for invalid in ([{}], [None], [record("other", "f")],
                        [record("fake", "")], [record("fake", "f", subject=" ")]):
            with self.subTest(records=invalid), self.assertRaises(ValueError):
                evaluate_probabilities([[1, 0]], invalid)
        for conflicting in (record("real", "f"), record("fake", "f", subject="other"),
                            record("fake", "f", split="train")):
            with self.subTest(record=conflicting), self.assertRaises(ValueError):
                evaluate_probabilities([[1, 0], [1, 0]], valid + [conflicting])

    def test_paired_video_bootstrap_identical_models_have_exact_zero_deltas(self):
        records = [record("fake", "f1", "alice"), record("fake", "f2", "alice"),
                   record("real", "r1", "bob"), record("real", "r2", "bob")]
        before = evaluate_probabilities([[.9, .1], [.1, .9], [.9, .1], [.1, .9]], records)
        result = paired_video_bootstrap(before, before, iterations=200)
        self.assertEqual(result["subject_count"], 2)
        self.assertEqual(result["videos_by_class"], {"fake": 2, "real": 2})
        self.assertEqual(result["unit"], "fraction")
        for metric in result["metrics"].values():
            self.assertEqual(metric["delta"], 0)
            self.assertEqual(metric["delta_95_percentile_interval"], [0, 0])
        json.dumps(result, allow_nan=False)

    def test_paired_video_bootstrap_perfect_improvement_is_known_and_seeded(self):
        records = [record("fake", "f", "alice"), record("real", "r", "bob")]
        before = evaluate_probabilities([[.1, .9], [.9, .1]], records)
        after = evaluate_probabilities([[.9, .1], [.1, .9]], records)
        result = paired_video_bootstrap(before, after, iterations=20)
        for name in ("macro_f1", "accuracy"):
            self.assertEqual(result["metrics"][name]["delta"], 1)
            self.assertEqual(result["metrics"][name]["delta_95_percentile_interval"], [1, 1])
        for name in ("APCER", "BPCER", "ACER"):
            self.assertEqual(result["metrics"][name]["delta"], -1)
            self.assertEqual(result["metrics"][name]["delta_95_percentile_interval"], [-1, -1])
        # A nonconstant result verifies matching draws and seed reproducibility.
        larger = records + [record("fake", "f2", "alice"), record("real", "r2", "bob")]
        partial = evaluate_probabilities([[.1, .9], [.1, .9], [.9, .1], [.1, .9]], larger)
        perfect = evaluate_probabilities([[.9, .1], [.1, .9], [.9, .1], [.1, .9]], larger)
        first = paired_video_bootstrap(partial, perfect, iterations=200)
        self.assertEqual(first, paired_video_bootstrap(partial, perfect, iterations=200))
        self.assertEqual(first["metrics"]["accuracy"]["delta"], .25)
        self.assertEqual(first["metrics"]["accuracy"]["delta_95_percentile_interval"], [0, .5])

    def test_paired_video_bootstrap_rejects_unpaired_or_invalid_metadata(self):
        import copy
        records = [record("fake", "f"), record("real", "r")]
        valid = evaluate_probabilities([[.9, .1], [.1, .9]], records)
        for field, replacement in (("video_id", "different"), ("true_label", "real"),
                                   ("subject_id", "different"), ("predicted_label", "uncertain")):
            mismatch = copy.deepcopy(valid)
            mismatch["per_video"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_video_bootstrap(valid, mismatch)
        duplicated = copy.deepcopy(valid)
        duplicated["per_video"].append(duplicated["per_video"][0])
        with self.assertRaises(ValueError):
            paired_video_bootstrap(valid, duplicated)
        one_class = evaluate_probabilities([[.9, .1]], [record("fake", "f")])
        with self.assertRaises(ValueError):
            paired_video_bootstrap(one_class, one_class)
        for iterations in (0, -1, 1.5, True):
            with self.subTest(iterations=iterations), self.assertRaises(ValueError):
                paired_video_bootstrap(valid, valid, iterations=iterations)


if __name__ == "__main__":
    unittest.main()
