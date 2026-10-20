"""Actual TensorFlow normalization/topology and serialization smoke checks.

Run after active training finishes; these checks create models but do not fit.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np


class GroupNormArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.keras.utils.set_random_seed(42)
        from scripts.train_groupnorm_merged import build_groupnorm_model
        cls.tf = tf
        cls.model = build_groupnorm_model(tf)

    def test_only_normalization_changes_and_api_shapes_remain_compatible(self):
        from model.livenessnet import LivenessNet
        reference = LivenessNet.build(32, 32, 3, 2)
        norms = (self.tf.keras.layers.BatchNormalization, self.tf.keras.layers.GroupNormalization,
                 self.tf.keras.layers.LayerNormalization)
        actual = [layer for layer in self.model.layers if not isinstance(layer, norms)]
        expected = [layer for layer in reference.layers if not isinstance(layer, norms)]
        self.assertEqual(len(actual), len(expected))
        for new, old in zip(actual, expected):
            self.assertIs(type(new), type(old))
            configs = []
            for layer in (new, old):
                config = layer.get_config().copy()
                config.pop('name')
                configs.append(config)
            self.assertEqual(configs[0], configs[1])
        self.assertEqual(self.model.input_shape, (None, 32, 32, 3))
        self.assertEqual(self.model.output_shape, (None, 2))
        self.assertEqual(sum(isinstance(layer, self.tf.keras.layers.GroupNormalization)
                             for layer in self.model.layers), 5)
        self.assertEqual(sum(isinstance(layer, self.tf.keras.layers.LayerNormalization)
                             for layer in self.model.layers), 1)
        self.assertFalse(any(isinstance(layer, self.tf.keras.layers.BatchNormalization)
                             for layer in self.model.layers))

    def test_normalization_has_no_batch_statistics_and_export_preserves_probabilities(self):
        first_norm = self.model.layers[1]
        rng = np.random.default_rng(42)
        samples = rng.uniform(0, 1, (4, 32, 32, 3)).astype(np.float32)
        # Cache varying, zero-mean pre-normalization features. This isolates
        # GroupNorm from upstream convolution kernels and ill-conditioned,
        # near-constant activations; the same first feature tensor is used in
        # the single and mixed calls. Serialization below still uses BGR-like
        # [0, 1] input images through the entire actual CNN.
        features = rng.normal(0, 1, (4, 32, 32, 32)).astype(np.float32)
        alone = first_norm(features[:1], training=True).numpy()
        mixed = first_norm(features, training=True).numpy()[:1]
        groups = features[:1].astype(np.float64).reshape(1, 32, 32, 8, 4)
        mean = groups.mean(axis=(1, 2, 4), keepdims=True)
        variance = ((groups - mean) ** 2).mean(axis=(1, 2, 4), keepdims=True)
        reference = ((groups - mean) / np.sqrt(variance + first_norm.epsilon)).reshape(1, 32, 32, 32)
        reference = reference * first_norm.gamma.numpy() + first_norm.beta.numpy()
        single_error = float(np.max(np.abs(alone - reference)))
        mixed_error = float(np.max(np.abs(mixed - reference)))
        # Two group reductions plus the pointwise normalization operations:
        # bound float32 rounding at the reduction tree depth, then require
        # batch-composition differences to fit the independently measured
        # float64 reference errors instead of relaxing a fixed tolerance.
        reduction_depth = int(np.ceil(np.log2(32 * 32 * 4)))
        scale = max(1., float(np.abs(reference).max()))
        roundoff_budget = np.finfo(np.float32).eps * (2 * reduction_depth + 8) * scale
        self.assertLessEqual(single_error, roundoff_budget)
        self.assertLessEqual(mixed_error, roundoff_budget)
        np.testing.assert_allclose(alone, mixed, rtol=0,
                                   atol=single_error + mixed_error + np.finfo(np.float32).eps * scale)
        self.assertEqual(first_norm.groups, 8)
        self.assertFalse(first_norm.non_trainable_weights)
        probabilities = self.model(samples, training=False).numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('keras', 'h5'):
                path = Path(directory) / f'candidate.{extension}'
                self.model.save(path)
                restored = self.tf.keras.models.load_model(path, compile=False)
                np.testing.assert_allclose(restored(samples, training=False).numpy(), probabilities,
                                           rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
