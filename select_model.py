"""Select a retained checkpoint using validation mAP, never test images."""

import argparse
import json
from pathlib import Path


def select(logdir, output=None, report=None):
    import tensorflow as tf

    from export_metrics import read_scalars

    root = Path(logdir)
    destination = Path(output) if output else root / "selected"
    if destination.exists():
        raise FileExistsError(str(destination))
    report_path = Path(report) if report else root / "model_selection.json"
    if report_path.exists():
        raise FileExistsError(str(report_path))
    candidates = []
    for name in ("reference", "experiment0", "experiment1"):
        for row in read_scalars(root / name / "eval"):
            if row["tag"] == "DetectionBoxes_Precision/mAP":
                candidates.append(
                    {
                        "experiment": name,
                        "step": row["step"],
                        "validation_map": row["value"],
                    }
                )
    if not candidates:
        raise ValueError("No validation mAP measurements found")
    best = max(candidates, key=lambda r: r["validation_map"])
    directory = root / best["experiment"]
    matching = []
    for index in directory.glob("ckpt-*.index"):
        prefix = str(index.with_suffix(""))
        step = int(
            tf.train.load_variable(prefix, "step/.ATTRIBUTES/VARIABLE_VALUE")
        )
        if step == best["step"]:
            matching.append(Path(prefix))
    if not matching:
        raise ValueError("The best evaluated checkpoint was not retained")
    checkpoint = sorted(matching)[-1].resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for source in checkpoint.parent.glob(checkpoint.name + ".*"):
        (destination / source.name).symlink_to(source)
    (destination / "checkpoint").write_text(
        'model_checkpoint_path: "{}"\n'.format(checkpoint.name)
    )
    (destination / "pipeline_new.config").write_text(
        (directory / "pipeline_new.config").read_text()
    )
    report = dict(
        best,
        checkpoint=str(checkpoint),
        candidates=candidates,
        criterion="Highest validation COCO mAP; test excluded",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--report")
    select(**vars(parser.parse_args()))
