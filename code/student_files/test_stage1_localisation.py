"""Stage 1 integration checks. Run from code/ with PYTHONPATH=. python -m
unittest discover -s student_files -p 'test_stage1_localisation.py' -v.
"""
import unittest

import numpy as np

from overlaylab.config import LocalisationConfig
from overlaylab.localisation import (localise_board, artwork_to_frame,
                                     overlay_frame, evaluate_audit_frames)
from overlaylab.supplied import rect_corners
from overlaylab.synthetic import textured_patch


class TestStage1Localisation(unittest.TestCase):
    def setUp(self):
        self.reference = np.repeat(textured_patch(128, seed=19)[..., None], 3, axis=2)
        self.frame = np.zeros((180, 200, 3), dtype=np.float32)
        self.frame[20:148, 30:158] = self.reference
        self.config = LocalisationConfig()
        self.config.warp.reference_shape = self.reference.shape[:2]
        self.expected = rect_corners(128, 128) + [30, 20]

    def test_recovers_board_translation_without_corner_initialisation(self):
        H, corners, pairs, inliers, diag = localise_board(self.reference, self.frame, self.config)
        self.assertTrue(diag["status"])
        self.assertIsNotNone(H)
        np.testing.assert_allclose(corners, self.expected, atol=0.5)
        self.assertEqual(len(pairs), len(inliers))
        self.assertEqual(diag["n_inliers"], int(inliers.sum()))
        self.assertEqual(diag["residuals"].shape, (len(pairs),))
        self.assertTrue(np.all(pairs[:, 0] < len(diag["keypoints_ref"])))
        self.assertTrue(np.all(pairs[:, 1] < len(diag["keypoints_frame"])))

    def test_blank_frame_reports_failure(self):
        H, corners, pairs, inliers, diag = localise_board(
            self.reference, np.zeros_like(self.frame), self.config)
        self.assertIsNone(H)
        self.assertFalse(diag["status"])
        self.assertTrue(np.isnan(corners).all())
        self.assertEqual(pairs.shape, (0, 2))
        self.assertEqual(inliers.shape, (0,))

    def test_artwork_size_composition_and_background_preservation(self):
        artwork = np.ones((16, 24, 3), dtype=np.float32)
        H = np.array([[1., 0., 30.], [0., 1., 20.], [0., 0., 1.]])
        composed = artwork_to_frame(artwork, H, self.config)
        last = composed @ np.array([23., 15., 1.])
        np.testing.assert_allclose(last[:2] / last[2], [157., 147.])
        output = overlay_frame(self.frame, artwork, H, self.config)
        np.testing.assert_allclose(output[20:148, 30:158], 1.)
        outside = np.ones(self.frame.shape[:2], dtype=bool)
        outside[20:148, 30:158] = False
        np.testing.assert_array_equal(output[outside], self.frame[outside])
        self.assertEqual(output.dtype, np.float32)

    def test_audit_summary_counts_failed_localisation(self):
        frames = {0: self.frame, 1: np.zeros_like(self.frame)}
        truth = {0: self.expected, 1: self.expected}
        result = evaluate_audit_frames(self.reference, frames, truth, self.config)
        self.assertEqual(result["summary"]["n"], 2)
        self.assertEqual(result["summary"]["failed"], 1)
        self.assertEqual(result["summary"]["success"], 0.5)
        self.assertTrue(np.isinf(result["per_frame"][1]["error"]))


if __name__ == "__main__":
    unittest.main()
