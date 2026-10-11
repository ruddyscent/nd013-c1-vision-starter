"""Export TensorBoard scalars and plot train/validation comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from tensorboard.util import tensor_util


def read_scalars(folder):
    """Retain the last recorded value per tag/step across resumed runs."""
    accumulator = EventAccumulator(
        str(folder), size_guidance={"tensors": 0, "scalars": 0}
    ).Reload()
    values = {}
    for tag in accumulator.Tags()["tensors"]:
        for event in accumulator.Tensors(tag):
            array = tensor_util.make_ndarray(event.tensor_proto)
            if array.size == 1 and array.dtype.kind in "fiu":
                key = (tag, event.step)
                row = {
                    "tag": tag,
                    "step": event.step,
                    "value": float(array.item()),
                    "wall_time": event.wall_time,
                }
                if (
                    key not in values
                    or values[key]["wall_time"] < event.wall_time
                ):
                    values[key] = row
    for tag in accumulator.Tags()["scalars"]:
        for event in accumulator.Scalars(tag):
            values[tag, event.step] = {
                "tag": tag,
                "step": event.step,
                "value": event.value,
                "wall_time": event.wall_time,
            }
    return sorted(values.values(), key=lambda row: (row["tag"], row["step"]))


def export(logdir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for experiment in ("reference", "experiment0", "experiment1"):
        for phase in ("train", "eval"):
            folder = Path(logdir) / experiment / phase
            if folder.exists():
                results[experiment + "/" + phase] = read_scalars(folder)
                if phase == "eval":
                    export_eval_images(folder, output / experiment)
    if not results:
        raise ValueError("No TensorBoard event directories found")
    (output / "scalars.json").write_text(json.dumps(results, indent=2))
    with (output / "scalars.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["run", "tag", "step", "value", "wall_time"]
        )
        writer.writeheader()
        for run, rows in results.items():
            for row in rows:
                writer.writerow(dict(run=run, **row))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tags = [
        "Loss/total_loss",
        "Loss/classification_loss",
        "Loss/localization_loss",
        "Loss/regularization_loss",
    ]
    colors = {
        "reference": "tab:orange",
        "experiment0": "tab:blue",
        "experiment1": "tab:green",
    }
    for ax, tag in zip(axes.flat, tags):
        for run, rows in results.items():
            experiment, phase = run.split("/")
            points = [row for row in rows if row["tag"] == tag]
            if points:
                ax.plot(
                    [p["step"] for p in points],
                    [p["value"] for p in points],
                    color=colors[experiment],
                    linestyle="-" if phase == "train" else "--",
                    marker=None if phase == "train" else "o",
                    label=run,
                    alpha=0.85,
                )
        ax.set(title=tag, xlabel="Recorded training step", ylabel="Loss")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Recorded TensorBoard scalars: train and validation")
    fig.tight_layout()
    fig.savefig(output / "loss_comparison.png", dpi=160)
    plt.close(fig)
    return results


def export_eval_images(folder, output):
    """Extract unmodified API prediction/ground-truth image summaries."""
    accumulator = EventAccumulator(
        str(folder), size_guidance={"tensors": 0}
    ).Reload()
    for tag in accumulator.Tags()["tensors"]:
        if tag not in ("eval_side_by_side_0_0", "eval_side_by_side_1_0"):
            continue
        for event in accumulator.Tensors(tag):
            values = tensor_util.make_ndarray(event.tensor_proto)
            if values.size >= 3 and values.dtype.kind == "O":
                output.mkdir(parents=True, exist_ok=True)
                name = "{}-step-{}.png".format(tag, event.step)
                (output / name).write_bytes(values.flat[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", default="experiments")
    parser.add_argument("--output", default="artifacts/metrics")
    args = parser.parse_args()
    export(args.logdir, args.output)
