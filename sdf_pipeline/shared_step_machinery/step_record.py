"""The ``Step`` record each step module defines, the ``Run`` its functions receive, and the id accessor for stage records.
A model-calling step is plain functions and one ``STEP = Step(...)``; ``build_request`` and ``read_reply`` take lists of documents."""

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sdf_pipeline.shared_step_machinery.update_steps import Accepted, ModelRequest, OutcomeRecords, StepPolicy, systemic_stop_message_for
from shared import utils

LAYER1, LAYER2, LAYER3, LAYER4 = "layer1", "layer2", "layer3", "layer4"   # folder keys: a run's stage folders


def id_field_of(record: dict) -> str:
    """The field holding a stage record's document id: ``prompt_id`` before the draft, ``doc_id`` from the draft on."""
    return "doc_id" if "doc_id" in record else "prompt_id"


def document_id_of(record: dict) -> str:
    return record[id_field_of(record)]


@dataclass(frozen=True)
class StageFile:
    folder_key: str   # which folder it lives in
    file_name: str
    id_field: str     # the field of each record that holds the document id


@dataclass(frozen=True)
class Run:
    """A run's configuration, prompt snapshot and stage folders ({folder key: folder}), and the inputs its steps
    loaded (``loaded_once_per_run``)."""
    config: dict
    prompts_directory: Path
    stage_directories: dict
    loaded: dict = field(default_factory=dict, compare=False)

    @property
    def workers(self) -> int:
        return self.config.get("workers", 1)

    def records(self, stage_file: StageFile) -> list:
        return utils.load_jsonl(self.stage_directories[stage_file.folder_key] / stage_file.file_name)


def loaded_once_per_run(function: Callable) -> Callable:
    """Make ``function(run)`` compute its value on the first call for a run and return it on every later call."""
    @functools.wraps(function)
    def load(run: Run):
        if function not in run.loaded:
            run.loaded[function] = function(run)
        return run.loaded[function]
    return load


def one_record_per_accepted_document(stage_file: StageFile, record_of: Callable) -> Callable:
    """A step's ``records`` for the step that takes one document per call: an accepted document is the one record
    ``record_of(document, update, run)`` in ``stage_file``; any other outcome writes nothing."""
    def records(documents: list, outcomes: dict, run: Run) -> OutcomeRecords:
        (document,) = documents
        outcome = outcomes[document_id_of(document)]
        if not isinstance(outcome, Accepted):
            return OutcomeRecords()
        return OutcomeRecords(((stage_file.file_name, record_of(document, outcome.update, run)),))
    return records


def _always(document: dict, run: Run) -> bool:
    return True


def _never(document: dict, run: Run) -> bool:
    return False


@dataclass(frozen=True)
class Step:
    """Everything about one model-calling step."""
    name: str
    folder_key: str                               # the folder the step writes in
    stage_tag: str                                # the cost-log tag of the step's calls
    model_setting_keys: tuple = ()                # ``sdf.<key>``; the first one set wins
    applies: Callable = _always                   # (document, run) -> False when the step skips the document
    finished_elsewhere: Callable = _never         # (document, run) -> True when another step finished it
    group: Callable | None = None                 # (documents, run) -> {group key: list}; None: groups of one document
    build_request: Callable | None = None
    read_reply: Callable | None = None
    records: Callable | None = None               # (documents, outcomes, run) -> every record the outcomes write
    writes: tuple = ()                            # stage files
    checkpoint_file_name: str | None = None       # the checkpoint file in the folder: a group's key is marked after its records
    writes_as_replies_arrive: bool = False        # write each outcome the moment it arrives
    policy: StepPolicy = StepPolicy()
    call_noun: str = "API call"
    unit_noun: str = "document"
    name_in_messages: str | None = None           # stop messages start with this instead of the stage tag
    progress_line: Callable | None = None
    after_run: Callable | None = None
    stop_message: Callable | None = None          # (documents, result, run) -> the exit message when a failed group stops the run

    def model_for(self, config: dict):
        """The model the step is configured with (the first setting key that is set)."""
        return next((config["sdf"].get(key) for key in self.model_setting_keys if config["sdf"].get(key)), None)

    def model_request(self, document: dict, user_message: str, system_prompt: str | None, run: Run,
                      **settings) -> ModelRequest:
        """The request for ``document``: the step's tag and model, an empty system prompt when none, and per-step ``settings``."""
        return ModelRequest(
            user_message=user_message, system_prompt=system_prompt or "", model=self.model_for(run.config),
            stage=self.stage_tag, item_id=document_id_of(document), **settings)

    @property
    def systemic_stop_message(self) -> str:
        """The exit message when every pending call of the step failed."""
        return systemic_stop_message_for(self.name_in_messages or self.stage_tag, self.call_noun, self.unit_noun)
