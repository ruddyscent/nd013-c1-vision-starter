"""Verify isolated run routing and selection without loading GPU weights."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from select_model import select


class ReproductionTests(unittest.TestCase):
    def test_selection_uses_new_run_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new run"
            for name in ("reference", "experiment0", "experiment1"):
                folder = root / name
                folder.mkdir(parents=True)
                (folder / "pipeline_new.config").write_text(name)
                (folder / "ckpt-2.index").write_text("index")
                (folder / "ckpt-2.data-00000-of-00001").write_text("weights")
            fake_tf = SimpleNamespace(
                train=SimpleNamespace(load_variable=lambda *args: 2500)
            )

            def scalars(folder):
                return [{
                    "tag": "DetectionBoxes_Precision/mAP",
                    "step": 2500,
                    "value": 0.2 if folder.parent.name == "experiment1"
                    else 0.1,
                }]

            with patch.dict("sys.modules", {"tensorflow": fake_tf}), \
                    patch("export_metrics.read_scalars", scalars):
                select(root)
                selected = root / "selected"
                self.assertEqual(
                    (selected / "pipeline_new.config").read_text(),
                    "experiment1",
                )
                self.assertEqual(
                    (selected / "ckpt-2.data-00000-of-00001").read_text(),
                    "weights",
                )
                report = json.loads(
                    (root / "model_selection.json").read_text()
                )
                self.assertEqual(report["experiment"], "experiment1")
                with self.assertRaises(FileExistsError):
                    select(root)
                self.assertEqual(
                    (selected / "pipeline_new.config").read_text(),
                    "experiment1",
                )

    def test_training_commands_use_run_root_with_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "new run"
            for name in ("reference", "experiment0", "experiment1"):
                (run / name).mkdir(parents=True)
            executable = root / "python"
            executable.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$@" >> "$CALLS"\n'
            )
            executable.chmod(0o755)
            calls = root / "calls"
            environment = dict(os.environ, PATH=str(root) + ":" + os.environ[
                "PATH"
            ], CALLS=str(calls))
            subprocess.run(
                ["bash", "run_experiments.sh", str(run)],
                env=environment, check=True,
            )
            arguments = calls.read_text().splitlines()
            self.assertEqual(arguments.count("evaluate_once.py"), 6)
            self.assertEqual(
                arguments.count("experiments/model_main_tf2.py"), 6
            )
            for name in ("reference", "experiment0", "experiment1"):
                self.assertEqual(
                    arguments.count("--model_dir=" + str(run / name)), 2
                )


if __name__ == "__main__":
    unittest.main()
