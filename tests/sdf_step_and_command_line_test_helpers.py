"""Small helpers shared by the SDF step tests and the tests that run a pipeline's ``main``."""

import sys

from sdf_pipeline.shared_step_machinery.update_steps import ModelReply


def reply(text, stop_reason="end_turn"):
    return ModelReply(text=text, stop_reason=stop_reason)


def run_main(monkeypatch, config_file, *extra_arguments, main, label=None):
    """Run a pipeline's ``main`` as ``run.py --config <config_file> [--label <label>] <extra_arguments>``."""
    label_arguments = ["--label", label] if label else []
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_file), *label_arguments, *extra_arguments])
    main()
