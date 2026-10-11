"""Regression checks for leakage, repeatability and source preservation."""

import tempfile
import unittest
from pathlib import Path

from create_splits import split


class SplitTests(unittest.TestCase):
    def test_disjoint_reproducible_and_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            for index in range(20):
                (source / "{}.tfrecord".format(index)).write_bytes(b"record")
            destination = Path(root) / "splits"
            first = split(source, destination)
            self.assertEqual(first, split(source, destination))
            paths = sum(first["splits"].values(), [])
            self.assertEqual(len(paths), 20)
            self.assertEqual(len(set(paths)), 20)
            self.assertEqual(len(list(source.glob("*.tfrecord"))), 20)
            self.assertTrue(
                all(p.is_symlink() for p in destination.glob("*/*.tfrecord"))
            )

    def test_preserves_external_test_set(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            for index in range(5):
                (source / "{}.tfrecord".format(index)).touch()
            destination = Path(root) / "splits"
            (destination / "test").mkdir(parents=True)
            existing = destination / "test" / "heldout.tfrecord"
            existing.write_bytes(b"held out")
            split(source, destination, test_fraction=0)
            self.assertEqual(existing.read_bytes(), b"held out")

    def test_rejects_insufficient_segments(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "one.tfrecord").touch()
            with self.assertRaises(ValueError):
                split(root, Path(root) / "out")


if __name__ == "__main__":
    unittest.main()
