"""Split oversized, closed TFRecord event files without changing any bytes."""

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path


def digest(paths):
    value = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
    return value.hexdigest()


def split_file(path, archive, max_bytes):
    """Verify concatenated chunks equal the original before archiving it."""
    path, archive = Path(path), Path(archive)
    if archive.exists():
        raise FileExistsError(str(archive))
    original_stat = path.stat()
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as temporary:
        parts, sizes, stream = [], [], None
        try:
            with path.open("rb") as source:
                while True:
                    header = source.read(8)
                    if not header:
                        break
                    if len(header) != 8:
                        raise ValueError("Truncated TFRecord length header")
                    length = struct.unpack("<Q", header)[0]
                    if length + 16 > max_bytes:
                        raise ValueError("A single record exceeds chunk size")
                    record = header + source.read(length + 8)
                    if len(record) != length + 16:
                        raise ValueError("Truncated TFRecord payload")
                    if stream is None or sizes[-1] + len(record) > max_bytes:
                        if stream is not None:
                            stream.close()
                        name = path.name + ".part-{:03d}".format(len(parts))
                        if (path.parent / name).exists():
                            raise FileExistsError(name)
                        parts.append(Path(temporary) / name)
                        sizes.append(0)
                        stream = parts[-1].open("wb")
                    stream.write(record)
                    sizes[-1] += len(record)
        finally:
            if stream is not None:
                stream.close()
        checksum = digest([path])
        current_stat = path.stat()
        unchanged = (
            current_stat.st_size == original_stat.st_size
            and current_stat.st_mtime_ns == original_stat.st_mtime_ns
            and current_stat.st_ino == original_stat.st_ino
        )
        if digest(parts) != checksum or not unchanged:
            raise ValueError("Source changed or byte preservation failed")
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archive)
        for part in parts:
            part.rename(path.parent / part.name)
        return {
            "source": str(path),
            "archive": str(archive),
            "sha256": checksum,
            "original_bytes": original_stat.st_size,
            "parts": [p.name for p in parts],
            "part_bytes": sizes,
            "concatenated_bytes_match": True,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", default="experiments")
    parser.add_argument("--max-mib", type=int, default=90)
    args = parser.parse_args()
    limit = args.max_mib * 1024 * 1024
    if limit <= 16:
        raise ValueError("Chunk size must exceed a TFRecord header")
    report = []
    for path in sorted(Path(args.logdir).rglob("*tfevents*")):
        if path.is_file() and path.stat().st_size > limit:
            archive = Path("artifacts/raw_events") / path.relative_to(
                args.logdir
            )
            report.append(split_file(path, archive, limit))
    if report:
        manifest = Path("artifacts/event_file_splits.json")
        previous = (
            json.loads(manifest.read_text()) if manifest.exists() else []
        )
        manifest.write_text(json.dumps(previous + report, indent=2))
    print(json.dumps(report, indent=2))
