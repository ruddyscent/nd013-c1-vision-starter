"""Validate generated protobuf configurations in the training environment."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("object_detection"),
    "TensorFlow Object Detection API is not installed",
)
class ConfigTests(unittest.TestCase):
    def test_sparse_labels_and_improvement_parameters(self):
        from edit_config import edit

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for split in ("train", "val"):
                (folder / split).mkdir()
                (folder / split / (split + ".tfrecord")).touch()
            checkpoint = folder / "ckpt-0"
            Path(str(checkpoint) + ".index").touch()
            config = edit(
                folder / "train",
                folder / "val",
                2,
                checkpoint,
                root / "label_map.pbtxt",
                template=root / "pipeline.config",
                output=folder / "pipeline.config",
                steps=2500,
                improved=True,
            )
            self.assertEqual(config.model.ssd.num_classes, 4)
            self.assertTrue(config.model.ssd.freeze_batchnorm)
            anchors = config.model.ssd.anchor_generator
            self.assertEqual(
                anchors.multiscale_anchor_generator.anchor_scale, 2.0
            )
            self.assertEqual(config.eval_input_reader[0].num_epochs, 1)
            self.assertFalse(config.eval_input_reader[0].shuffle)
            self.assertEqual(
                len(config.train_config.data_augmentation_options), 4
            )

    def test_overlap_rejected_before_writing(self):
        from edit_config import edit

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "segment.tfrecord").touch()
            with self.assertRaises(ValueError):
                edit(
                    folder,
                    folder,
                    2,
                    "unused",
                    "unused",
                    output=folder / "must_not_exist.config",
                )
            self.assertFalse((folder / "must_not_exist.config").exists())
