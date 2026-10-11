"""Check semantic channel remapping independently for every anchor."""

import importlib.util
import unittest

import numpy as np

HAS_API = importlib.util.find_spec("object_detection") is not None
if HAS_API:
    from prepare_head_transfer import remap_classifier


@unittest.skipUnless(
    HAS_API, "TensorFlow Object Detection API is not installed"
)
class HeadTransferTests(unittest.TestCase):
    def test_kernel_preserves_anchor_and_class_order(self):
        source = np.arange(2 * 6 * 91).reshape(1, 1, 2, 546)
        mapped = remap_classifier(source, (1, 1, 2, 30))
        original = source.reshape(2, 6, 91)
        actual = mapped.reshape(2, 6, 5)
        for target, coco in [(0, 0), (1, 3), (2, 1), (4, 2)]:
            np.testing.assert_array_equal(
                actual[..., target], original[..., coco]
            )
        self.assertTrue(np.all(actual[..., 3] == 0))

    def test_unused_bias_is_suppressed(self):
        source = np.arange(546, dtype=np.float32)
        mapped = remap_classifier(source, (30,)).reshape(6, 5)
        self.assertTrue(np.all(mapped[:, 3] == -10))
        np.testing.assert_array_equal(
            mapped[:, 1], source.reshape(6, 91)[:, 3]
        )


if __name__ == "__main__":
    unittest.main()
