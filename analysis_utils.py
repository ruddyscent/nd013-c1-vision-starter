"""Plot decoded examples without modifying their arrays."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

CLASSES = {
    1: ("Vehicle", "red"),
    2: ("Pedestrian", "blue"),
    4: ("Cyclist", "green"),
}


def display_instances(batch, ax=None, title="Annotated frame"):
    """Draw normalized [ymin, xmin, ymax, xmax] boxes on an HWC image."""
    image = np.asarray(batch["image"])
    boxes = np.asarray(batch["groundtruth_boxes"])
    classes = np.asarray(batch["groundtruth_classes"])
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(np.clip(image, 0, 255).astype(np.uint8))
    height, width = image.shape[:2]
    for box, class_id in zip(boxes, classes):
        label, color = CLASSES.get(int(class_id), ("Unknown", "yellow"))
        ymin, xmin, ymax, xmax = box * [height, width, height, width]
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )
        ax.text(
            xmin,
            ymin,
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.7, "pad": 1},
        )
    ax.set_title(title)
    ax.axis("off")
    return ax
