import unittest

import numpy as np

from face_antispoofing.motion import procrustes_residual


class MotionTests(unittest.TestCase):
    def test_translation_scale_rotation_are_removed(self) -> None:
        reference = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        current = (reference @ rotation) * 3.0 + np.array([10.0, -4.0])
        self.assertLess(procrustes_residual(reference, current), 1e-10)

    def test_non_rigid_change_has_residual(self) -> None:
        reference = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        current = reference.copy()
        current[0] += [0.4, -0.2]
        self.assertGreater(procrustes_residual(reference, current), 0.03)


if __name__ == "__main__":
    unittest.main()
