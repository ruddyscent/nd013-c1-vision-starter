"""Evaluate a completed checkpoint without a long polling delay."""

import argparse
from pathlib import Path

import tensorflow as tf
from object_detection import model_lib_v2


def evaluate(model_dir, steps):
    """Require the expected training step before running validation."""
    checkpoint = tf.train.latest_checkpoint(model_dir)
    if checkpoint is None:
        raise FileNotFoundError("No checkpoint in " + model_dir)
    actual = int(
        tf.train.load_variable(checkpoint, "step/.ATTRIBUTES/VARIABLE_VALUE")
    )
    if actual != steps:
        raise ValueError("Expected step {}, found {}".format(steps, actual))
    tf.config.set_soft_device_placement(True)
    model_lib_v2.eval_continuously(
        pipeline_config_path=str(Path(model_dir) / "pipeline_new.config"),
        model_dir=model_dir,
        checkpoint_dir=model_dir,
        train_steps=steps,
        sample_1_of_n_eval_examples=1,
        sample_1_of_n_eval_on_train_examples=1,
        wait_interval=0,
        timeout=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--steps", required=True, type=int)
    args = parser.parse_args()
    evaluate(args.model_dir, args.steps)
