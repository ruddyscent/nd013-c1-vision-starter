"""Visualize the actual Object Detection API photometric transforms."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.core import preprocessor

from run_eda import display_instances


def main():
    tf.random.set_seed(42)
    path = sorted(Path("data/train").glob("*.tfrecord"))[0]
    raw = next(iter(tf.data.TFRecordDataset(str(path))))
    f = tf.train.Example.FromString(raw.numpy()).features.feature
    image = tf.cast(
        tf.io.decode_jpeg(f["image/encoded"].bytes_list.value[0]), tf.float32
    )
    boxes = np.array(
        [
            f["image/object/bbox/" + k].float_list.value
            for k in ["ymin", "xmin", "ymax", "xmax"]
        ]
    ).T.reshape(-1, 4)
    labels = np.array(f["image/object/class/label"].int64_list.value)
    bright = preprocessor.random_adjust_brightness(
        image, max_delta=0.15, seed=42
    )
    contrast = preprocessor.random_adjust_contrast(
        image, min_delta=0.8, max_delta=1.2, seed=42
    )
    both = preprocessor.random_adjust_contrast(
        bright, min_delta=0.8, max_delta=1.2, seed=43
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, transformed, title in zip(
        axes.flat,
        [image, bright, contrast, both],
        [
            "Original",
            "Brightness: max delta 0.15",
            "Contrast: factor 0.8-1.2",
            "Brightness + contrast",
        ],
    ):
        display_instances(
            (
                np.clip(transformed.numpy(), 0, 255).astype(np.uint8),
                boxes,
                labels,
            ),
            ax,
        )
        ax.set_title(title)
    fig.tight_layout()
    Path("artifacts/augmentations").mkdir(parents=True, exist_ok=True)
    fig.savefig("artifacts/augmentations/photometric.png", dpi=140)
    plt.close(fig)
    print("AUGMENTATIONS_READY")


if __name__ == "__main__":
    main()
