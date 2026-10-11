"""Adapt COCO SSD predictor weights to the supplied Waymo label slots."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2


def remap_classifier(value, target_shape):
    """Map background/car/person/unused/bicycle into five target slots."""
    source_slots, target_slots = 91, 5
    anchors = target_shape[-1] // target_slots
    if value.shape[-1] != anchors * source_slots:
        raise ValueError("Unexpected COCO classifier shape")
    grouped = value.reshape(value.shape[:-1] + (anchors, source_slots))
    selected = grouped[..., [0, 3, 1, 0, 2]].copy()
    selected[..., 3] = -10.0 if value.ndim == 1 else 0.0
    return selected.reshape(target_shape)


def prepare(source, template, output, checkpoint_output=None, report=None):
    """Copy matching tensors, remap two classifier tensors, verify restore."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(str(destination))
    target_prefix = Path(
        checkpoint_output or destination.parent / "initialization" / "ckpt"
    )
    if list(target_prefix.parent.glob(target_prefix.name + ".*")):
        raise FileExistsError(str(target_prefix))
    report_path = Path(report or destination.parent / "head_transfer.json")
    if report_path.exists():
        raise FileExistsError(str(report_path))
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(Path(template).read_text(), config)
    model_config = config.model.ssd
    if model_config.num_classes != 4 or not model_config.add_background_class:
        raise ValueError("Expected four classes plus the background slot")
    model_config.anchor_generator.multiscale_anchor_generator.anchor_scale = 4
    model_config.freeze_batchnorm = True
    learning_rate = (
        config.train_config.optimizer.momentum_optimizer.learning_rate
    )
    schedule = learning_rate.cosine_decay_learning_rate
    schedule.learning_rate_base = 0.001
    schedule.warmup_learning_rate = 0.0001
    schedule.total_steps = 2500
    schedule.warmup_steps = 250
    config.train_config.num_steps = 2500
    model = model_builder.build(config.model, is_training=False)
    image, shapes = model.preprocess(tf.zeros([1, 640, 640, 3]))
    model.predict(image, shapes)
    checkpoint = tf.train.Checkpoint(model=model)
    _ = checkpoint.save_counter
    pretrained = tf.train.load_checkpoint(source)
    pretrained_shapes = pretrained.get_variable_to_shape_map()
    target_prefix.parent.mkdir(parents=True, exist_ok=True)
    copied, remapped = [], []
    with tempfile.TemporaryDirectory() as temporary:
        initial = checkpoint.write(str(Path(temporary) / "initial"))
        reader = tf.train.load_checkpoint(initial)
        names, tensors = [], []
        for name, shape in tf.train.list_variables(initial):
            value = reader.get_tensor(name)
            if name.startswith("model/"):
                if name not in pretrained_shapes:
                    raise ValueError("Missing pretrained tensor: " + name)
                original = pretrained.get_tensor(name)
                if tuple(original.shape) == tuple(shape):
                    value = original
                    copied.append(name)
                elif "/class_predictions_with_background/" in name:
                    value = remap_classifier(original, tuple(shape))
                    remapped.append(name)
                else:
                    raise ValueError("Unexpected shape mismatch: " + name)
            names.append(name)
            tensors.append(tf.convert_to_tensor(value))
        if len(remapped) != 2:
            raise ValueError("Expected exactly classifier kernel and bias")
        tf.raw_ops.SaveV2(
            prefix=str(target_prefix),
            tensor_names=names,
            shape_and_slices=[""] * len(names),
            tensors=tensors,
        )
    checkpoint.restore(str(target_prefix)).assert_existing_objects_matched()
    verified = tf.train.load_checkpoint(str(target_prefix))
    for name in copied:
        np.testing.assert_array_equal(
            verified.get_tensor(name), pretrained.get_tensor(name)
        )
    for name in remapped:
        restored = verified.get_tensor(name)
        np.testing.assert_array_equal(
            restored,
            remap_classifier(pretrained.get_tensor(name), restored.shape),
        )
    config.train_config.fine_tune_checkpoint = str(target_prefix.resolve())
    config.train_config.fine_tune_checkpoint_type = "full"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text_format.MessageToString(config))
    report = {
        "source": source,
        "copied_tensors": len(copied),
        "remapped_tensors": remapped,
        "source_channels": [0, 3, 1, None, 2],
        "target_channels": [
            "background",
            "vehicle",
            "pedestrian",
            "unused",
            "cyclist",
        ],
        "limitation": "Car and bicycle are initialization proxies, not exact "
        "Waymo vehicle and cyclist category definitions.",
        "exact_copy_verification": "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--template", required=True
    )
    parser.add_argument(
        "--output", required=True
    )
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--report")
    prepare(**vars(parser.parse_args()))
