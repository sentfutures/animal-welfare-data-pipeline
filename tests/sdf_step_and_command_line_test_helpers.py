"""Small helpers shared by the SDF step tests and the tests that run a pipeline's ``main``."""

import sys

from sdf_pipeline.document_event_log import read_document_events
from sdf_pipeline.update_steps import ModelReply


def events_of(stage_dir, step_name=None):
    """The run's document events (those of one step when ``step_name`` is given), read from the
    event log beside the stage folder."""
    events = read_document_events(stage_dir.parent / "document_events.jsonl")
    return events if step_name is None else [event for event in events if event["step"] == step_name]


def reply(text, stop_reason="end_turn"):
    return ModelReply(text=text, stop_reason=stop_reason)


def run_main(monkeypatch, config_file, *extra_arguments, main, label=None):
    """Run a pipeline's ``main`` as ``run.py --config <config_file> [--label <label>] <extra_arguments>``."""
    label_arguments = ["--label", label] if label else []
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_file), *label_arguments, *extra_arguments])
    main()
