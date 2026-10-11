"""Create a portable Object Detection API configuration for a real dataset."""

import argparse
from pathlib import Path

from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util


def edit(
    train_dir,
    eval_dir,
    batch_size,
    checkpoint,
    label_map,
    template="pipeline.config",
    output="pipeline_new.config",
    steps=None,
    improved=False,
):
    """Preserve record label IDs and reject missing or overlapping inputs."""
    training = sorted(
        str(p.resolve()) for p in Path(train_dir).glob("*.tfrecord")
    )
    evaluation = sorted(
        str(p.resolve()) for p in Path(eval_dir).glob("*.tfrecord")
    )
    if not training or not evaluation:
        raise ValueError("Training and validation must each contain TFRecords")
    if set(training) & set(evaluation):
        raise ValueError("Training and validation segments overlap")
    if batch_size <= 0 or (steps is not None and steps <= 0):
        raise ValueError("Batch size and steps must be positive")
    checkpoint = str(Path(checkpoint).resolve())
    if not Path(checkpoint + ".index").is_file():
        raise FileNotFoundError(checkpoint + ".index")
    labels = label_map_util.get_label_map_dict(str(label_map))
    if not labels:
        raise ValueError("Label map is empty")
    config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(Path(template).read_text(), config)
    # Waymo starter records use IDs 1, 2, 4. ID 4 needs an output slot;
    # setting three classes silently excludes cyclists from one-hot targets.
    config.model.ssd.num_classes = max(labels.values())
    config.train_config.batch_size = batch_size
    config.train_config.fine_tune_checkpoint = checkpoint
    config.train_config.replicas_to_aggregate = 1
    if steps is not None:
        config.train_config.num_steps = steps
        schedule = (
            config.train_config.optimizer.momentum_optimizer.learning_rate
        )
        schedule.cosine_decay_learning_rate.total_steps = steps
        schedule.cosine_decay_learning_rate.warmup_steps = min(
            2000, steps // 10
        )
    config.train_input_reader.label_map_path = str(Path(label_map).resolve())
    config.train_input_reader.tf_record_input_reader.input_path[:] = training
    reader = config.eval_input_reader[0]
    reader.label_map_path = str(Path(label_map).resolve())
    reader.tf_record_input_reader.input_path[:] = evaluation
    reader.shuffle = False
    reader.num_epochs = 1
    if improved:
        # Small batches produce noisy batch statistics during fine-tuning.
        config.model.ssd.freeze_batchnorm = True
        anchors = config.model.ssd.anchor_generator.multiscale_anchor_generator
        anchors.anchor_scale = 2.0
        schedule = (
            config.train_config.optimizer.momentum_optimizer.learning_rate
        )
        schedule.cosine_decay_learning_rate.learning_rate_base = 0.0025
        schedule.cosine_decay_learning_rate.warmup_learning_rate = 0.0005
        option = config.train_config.data_augmentation_options.add()
        option.random_adjust_brightness.max_delta = 0.15
        option = config.train_config.data_augmentation_options.add()
        option.random_adjust_contrast.min_delta = 0.8
        option.random_adjust_contrast.max_delta = 1.2
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text_format.MessageToString(config))
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_dir", "eval_dir", "checkpoint", "label_map"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--template", default="pipeline.config")
    parser.add_argument("--output", default="pipeline_new.config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--improved", action="store_true")
    edit(**vars(parser.parse_args()))
