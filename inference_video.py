"""Render ordered detections from a processed TFRecord to MP4 or GIF."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils.label_map_util import (
    create_category_index_from_labelmap,
)
from PIL import Image


def main(
    labelmap_path,
    model_path,
    tf_record_path,
    config_path,
    output_path,
    fps=10,
    max_frames=None,
    score_threshold=0.3,
):
    """Stream MP4 frames or save a bounded GIF preview."""
    del config_path  # Retained for compatibility with the starter CLI.
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(str(output))
    if fps <= 0 or not 0 <= score_threshold <= 1:
        raise ValueError("Invalid FPS or confidence threshold")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if output.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("Output must be .mp4 or .gif")
    if output.suffix.lower() == ".gif" and max_frames is None:
        raise ValueError("Use --max_frames for a memory-bounded GIF preview")
    category_index = create_category_index_from_labelmap(
        labelmap_path, use_display_name=True
    )
    model = tf.saved_model.load(model_path)
    detect = model.signatures["serving_default"]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer, frames, count = None, [], 0
    try:
        for raw in tf.data.TFRecordDataset(tf_record_path):
            if max_frames is not None and count >= max_frames:
                break
            example = tf.train.Example.FromString(raw.numpy())
            encoded = example.features.feature[
                "image/encoded"
            ].bytes_list.value[0]
            image = tf.io.decode_jpeg(encoded, channels=3)
            predictions = detect(image[tf.newaxis, ...])
            number = int(predictions["num_detections"][0])
            frame = image.numpy().copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                predictions["detection_boxes"][0, :number].numpy(),
                predictions["detection_classes"][0, :number]
                .numpy()
                .astype(int),
                predictions["detection_scores"][0, :number].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=score_threshold,
                agnostic_mode=False,
            )
            if output.suffix.lower() == ".gif":
                preview = Image.fromarray(frame)
                preview.thumbnail((480, 480))
                frames.append(preview)
            else:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-loglevel",
                            "error",
                            "-f",
                            "rawvideo",
                            "-vcodec",
                            "rawvideo",
                            "-pix_fmt",
                            "rgb24",
                            "-s",
                            "{}x{}".format(width, height),
                            "-r",
                            str(fps),
                            "-i",
                            "-",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(output),
                        ],
                        stdin=subprocess.PIPE,
                    )
                writer.stdin.write(np.ascontiguousarray(frame).tobytes())
            count += 1
        if count == 0:
            raise ValueError("TFRecord contains no images")
        if frames:
            frames[0].save(
                str(output),
                save_all=True,
                append_images=frames[1:],
                duration=1000 / fps,
                loop=0,
            )
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the video")
    metadata = {
        "frames": count,
        "fps": fps,
        "model": model_path,
        "source": tf_record_path,
        "score_threshold": score_threshold,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelmap_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tf_record_path", required=True)
    parser.add_argument("--config_path", default="pipeline.config")
    parser.add_argument("--output_path", default="animation.mp4")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    print(main(**vars(parser.parse_args())))
