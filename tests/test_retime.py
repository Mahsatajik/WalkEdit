"""Check timing contracts and rotation continuity without model downloads."""

import unittest

import numpy as np

from retime import interpolate, source_times


class TimingTests(unittest.TestCase):
    def test_uniform_speed(self):
        times = source_times(60, 30, 30, 1.5)
        self.assertEqual(len(times), 40)
        np.testing.assert_allclose(times[::2], np.arange(0, 60, 3) / 30)

    def test_segment_boundaries(self):
        times = source_times(121, 30, 30, 2, 1, 3)
        np.testing.assert_allclose(times[[0, 30, 45, 60, 90]], [0, 1, 2, 3, 4])
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_invalid_inputs(self):
        for speed in (0, -1, float("nan"), float("inf")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                source_times(60, 30, 30, speed)
        with self.assertRaises(ValueError):
            source_times(60, 30, 30, 1.5, 1, 0.5)

    def test_linear_translation(self):
        values = np.array([[0, 0, 0], [2, 4, 6]])
        np.testing.assert_allclose(interpolate(values, [0, 1], [0.5]), [[1, 2, 3]])

    def test_smooth_transition(self):
        times = source_times(241, 60, 60, 2, 1, 3, transition=0.3)
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertLessEqual(times[-1], 4)
        speed = np.diff(times) * 60
        self.assertLess(np.max(np.abs(np.diff(speed))), 0.25)
        self.assertGreater(speed.max(), 1.99)
        self.assertLess(speed.max(), 2.01)
        np.testing.assert_allclose(times[:50], np.arange(50) / 60, atol=1e-7)

    def test_smooth_identity(self):
        times = source_times(121, 30, 30, 1, 1, 3, transition=0.5)
        np.testing.assert_allclose(times, np.arange(121) / 30, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
