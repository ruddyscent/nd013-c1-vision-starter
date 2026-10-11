"""Create reproducible, segment-disjoint splits without moving source data."""

import argparse
import json
import random
from pathlib import Path


def split(source, destination, seed=42, val_fraction=0.2, test_fraction=0.1):
    """Link whole segments; set test_fraction=0 to keep a held-out test set."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not 0 < val_fraction < 1 or not 0 <= test_fraction < 1 - val_fraction:
        raise ValueError(
            "Fractions must leave nonempty training and validation sets"
        )
    files = sorted(source.glob("*.tfrecord"))
    if not files:
        raise ValueError("No TFRecord files found in {}".format(source))
    random.Random(seed).shuffle(files)
    n_val = max(1, round(len(files) * val_fraction))
    n_test = max(1, round(len(files) * test_fraction)) if test_fraction else 0
    if n_val + n_test >= len(files):
        raise ValueError("Too few segments for the requested splits")
    groups = {
        "val": files[:n_val],
        "test": files[n_val : n_val + n_test],
        "train": files[n_val + n_test :],
    }
    # Validate all destinations before creating any links.
    for name, paths in groups.items():
        folder = destination / name
        expected = {p.name for p in paths}
        if folder.exists() and (name != "test" or test_fraction):
            if any(p.name not in expected for p in folder.glob("*.tfrecord")):
                raise ValueError("Existing split differs: {}".format(folder))
        for path in paths:
            target = folder / path.name
            if (
                target.exists() or target.is_symlink()
            ) and target.resolve() != path:
                raise FileExistsError(str(target))
    for name, paths in groups.items():
        (destination / name).mkdir(parents=True, exist_ok=True)
        for path in paths:
            target = destination / name / path.name
            if not target.exists():
                target.symlink_to(path)
    manifest = {
        "seed": seed,
        "source": str(source),
        "splits": {k: [str(p) for p in v] for k, v in groups.items()},
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    split(
        args.source,
        args.destination,
        args.seed,
        args.val_fraction,
        args.test_fraction,
    )
