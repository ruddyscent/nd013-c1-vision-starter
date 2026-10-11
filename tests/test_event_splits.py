"""Verify record-boundary splitting and preservation on malformed input."""

import struct
import tempfile
import unittest
from pathlib import Path

from split_event_files import split_file


class EventSplitTests(unittest.TestCase):
    def test_reassembly_preserves_every_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, archive = root / "events", root / "archive" / "original"
            records = [
                struct.pack("<Q", n) + b"a" * (n + 8) for n in (10, 11, 12)
            ]
            original = b"".join(records)
            path.write_bytes(original)
            report = split_file(path, archive, 50)
            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(
                b"".join((root / n).read_bytes() for n in report["parts"]),
                original,
            )
            self.assertFalse(path.exists())
            self.assertTrue(all(n <= 50 for n in report["part_bytes"]))

    def test_truncated_input_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events"
            archive = Path(directory) / "archive"
            path.write_bytes(struct.pack("<Q", 100) + b"short")
            with self.assertRaisesRegex(ValueError, "Truncated"):
                split_file(path, archive, 200)
            self.assertTrue(path.exists())
            self.assertFalse(archive.exists())

    def test_oversized_record_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events"
            archive = Path(directory) / "archive"
            path.write_bytes(struct.pack("<Q", 100) + b"a" * 108)
            with self.assertRaisesRegex(ValueError, "single record"):
                split_file(path, archive, 50)
            self.assertTrue(path.exists())
            self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
