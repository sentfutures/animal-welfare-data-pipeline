"""Small helpers shared by the SDF step tests and the tests that run a pipeline's ``main``."""

import sys

from sdf_pipeline.shared_step_machinery.step_record import Run, document_id_of
from sdf_pipeline.shared_step_machinery.update_steps import ModelReply


def reply(text, stop_reason="end_turn"):
    return ModelReply(text=text, stop_reason=stop_reason)


def run_main(monkeypatch, config_file, *extra_arguments, main, label=None):
    """Run a pipeline's ``main`` as ``run.py --config <config_file> [--label <label>] <extra_arguments>``."""
    label_arguments = ["--label", label] if label else []
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_file), *label_arguments, *extra_arguments])
    main()


def run_of(config, prompts_directory, folder_key=None, folder=None, **more_folders):
    """A ``Run`` over a config, a prompts folder and the stage folders given by folder key."""
    return Run(config, prompts_directory, {**({folder_key: folder} if folder_key else {}), **more_folders})


def outcome_of(step, document, reply_or_text, run, stop_reason="end_turn"):
    """The one outcome ``step.read_reply`` gives for a one-document step."""
    reply = reply_or_text if isinstance(reply_or_text, ModelReply) else ModelReply(reply_or_text, stop_reason)
    (outcome,) = step.read_reply([document], reply, run).values()
    return outcome


def records_of(step, document, outcome, run):
    """The ``OutcomeRecords`` the runner would write for a one-document step's outcome."""
    return step.records([document], {document_id_of(document): outcome}, run)
