"""Check coordinate scaling and preservation of scarce-class annotations."""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_utils import display_instances


class VisualizationTests(unittest.TestCase):
    def test_non_square_frame_and_sparse_cyclist_id(self):
        boxes = np.array([[0.1, 0.2, 0.8, 0.9]])
        original = boxes.copy()
        ax = display_instances(
            {
                "image": np.zeros((100, 200, 3), dtype=np.uint8),
                "groundtruth_boxes": boxes,
                "groundtruth_classes": np.array([4]),
            }
        )
        patch = ax.patches[0]
        self.assertEqual(patch.get_xy(), (40, 10))
        self.assertAlmostEqual(patch.get_width(), 140)
        self.assertAlmostEqual(patch.get_height(), 70)
        self.assertEqual(ax.texts[0].get_text(), "Cyclist")
        np.testing.assert_array_equal(boxes, original)
        plt.close(ax.figure)

    def test_empty_annotations(self):
        ax = display_instances(
            {
                "image": np.zeros((100, 200, 3), dtype=np.uint8),
                "groundtruth_boxes": np.empty((0, 4)),
                "groundtruth_classes": np.array([]),
            }
        )
        self.assertEqual(len(ax.patches), 0)
        plt.close(ax.figure)
