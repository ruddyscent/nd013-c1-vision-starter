"""Measure every supplied frame and save class-coded annotations."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from analysis_utils import display_instances as draw_example

LABELS = {
    1: ("Vehicle", "red"),
    2: ("Pedestrian", "blue"),
    4: ("Cyclist", "green"),
}


def display_instances(example, ax):
    image, boxes, labels = example
    return draw_example(
        {
            "image": image,
            "groundtruth_boxes": boxes,
            "groundtruth_classes": labels,
        },
        ax=ax,
    )


def analyze():
    output = Path("artifacts/eda")
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    all_names = []
    for split in ["train", "val", "test"]:
        paths = sorted(Path("data", split).glob("*.tfrecord"))
        all_names.extend(p.name for p in paths)
        counts, shapes = Counter(), Counter()
        brightness, objects, areas, samples = [], [], [], []
        rng = np.random.RandomState(42)
        for path in paths:
            for raw in tf.data.TFRecordDataset(str(path)):
                f = tf.train.Example.FromString(raw.numpy()).features.feature
                image = tf.io.decode_jpeg(
                    f["image/encoded"].bytes_list.value[0]
                ).numpy()
                labels = np.array(
                    f["image/object/class/label"].int64_list.value
                )
                boxes = np.array(
                    [
                        f["image/object/bbox/" + key].float_list.value
                        for key in ["ymin", "xmin", "ymax", "xmax"]
                    ]
                ).T.reshape(-1, 4)
                counts.update(labels.tolist())
                shapes[str(image.shape)] += 1
                brightness.append(float(image.mean()))
                objects.append(len(labels))
                areas.extend(
                    (
                        (boxes[:, 2] - boxes[:, 0])
                        * (boxes[:, 3] - boxes[:, 1])
                    ).tolist()
                )
                sample = (image, boxes, labels)
                if len(samples) < 10:
                    samples.append(sample)
                else:
                    index = rng.randint(len(brightness))
                    if index < 10:
                        samples[index] = sample
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ids = sorted(counts)
        axes[0, 0].bar(
            [LABELS.get(i, (str(i), "gray"))[0] for i in ids],
            [counts[i] for i in ids],
            color=[LABELS.get(i, ("", "gray"))[1] for i in ids],
        )
        axes[0, 0].set(
            title=split + ": class distribution", ylabel="Bounding boxes"
        )
        for ax, values, title, xlabel in [
            (axes[0, 1], brightness, "Brightness", "Mean RGB intensity"),
            (axes[1, 0], objects, "Scene density", "Objects per frame"),
            (axes[1, 1], areas, "Box size", "Normalized area"),
        ]:
            ax.hist(values, bins=30)
            ax.set(title=split + ": " + title, xlabel=xlabel, ylabel="Count")
        fig.tight_layout()
        fig.savefig(str(output / (split + "_distributions.png")), dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        for i, (ax, example) in enumerate(zip(axes.flat, samples)):
            display_instances(example, ax)
            ax.set_title(split + " sample " + str(i + 1))
        fig.tight_layout()
        fig.savefig(str(output / (split + "_samples.png")), dpi=120)
        plt.close(fig)
        report[split] = dict(
            segments=len(paths),
            frames=len(brightness),
            class_counts=dict(counts),
            image_shapes=dict(shapes),
            mean_objects=float(np.mean(objects)),
            brightness_mean=float(np.mean(brightness)),
            brightness_std=float(np.std(brightness)),
            box_area_median=float(np.median(areas)),
            files=[p.name for p in paths],
        )
        (output / "statistics.json").write_text(json.dumps(report, indent=2))
        print(
            split,
            json.dumps(
                {k: v for k, v in report[split].items() if k != "files"}
            ),
            flush=True,
        )
    assert len(all_names) == len(set(all_names)), "Overlapping segments"


if __name__ == "__main__":
    analyze()
