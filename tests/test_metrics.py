import math
import unittest

from face_antispoofing.evaluation.metrics import (
    aggregate_video_scores,
    compute_binary_metrics,
    find_best_threshold,
    roc_auc,
)


class MetricTests(unittest.TestCase):
    def test_perfect_metrics(self) -> None:
        result = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["tn"], 2)
        self.assertEqual(result["acer"], 0.0)
        self.assertEqual(result["roc_auc"], 1.0)

    def test_iso_error_rates(self) -> None:
        result = compute_binary_metrics([0, 0, 1, 1], [0.9, 0.1, 0.2, 0.8])
        self.assertEqual(result["apcer"], 0.5)
        self.assertEqual(result["bpcer"], 0.5)
        self.assertEqual(result["acer"], 0.5)

    def test_auc_ties(self) -> None:
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)

    def test_auc_requires_both_classes(self) -> None:
        self.assertTrue(math.isnan(roc_auc([1, 1], [0.4, 0.6])))

    def test_precision_is_zero_when_nothing_is_accepted(self) -> None:
        result = compute_binary_metrics([0, 1], [0.1, 0.2], threshold=0.9)
        self.assertEqual(result["precision_live"], 0.0)

    def test_best_threshold_is_selected_without_test_data(self) -> None:
        threshold = find_best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8])
        result = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8], threshold)
        self.assertEqual(result["acer"], 0.0)

    def test_video_aggregation(self) -> None:
        rows = [
            {"source_video": "a.mp4", "label": 1, "score": 0.8, "attack_type": "none"},
            {"source_video": "a.mp4", "label": 1, "score": 1.0, "attack_type": "none"},
            {"source_video": "b.mp4", "label": 0, "score": 0.2, "attack_type": "print"},
        ]
        result = aggregate_video_scores(rows)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["score"], 0.9)

    def test_mixed_video_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_video_scores(
                [
                    {"source_video": "a", "label": 0, "score": 0.2},
                    {"source_video": "a", "label": 1, "score": 0.8},
                ]
            )


if __name__ == "__main__":
    unittest.main()
