"""The ``Step`` record each step file defines, the ``Run`` its functions receive, and the id accessor for stage records.
A model-calling step file holds plain functions and one ``STEP = Step(...)``; ``build_request`` and ``read_reply`` take lists of documents."""

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sdf_pipeline.compatibility_with_runs_made_by_earlier_code.renamed_model_settings import model_setting
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, ModelRequest, OutcomeRecords, StepPolicy, systemic_stop_message_for)
from shared import sdf_run_layouts, utils


def id_field_of(record: dict) -> str:
    """The field holding a stage record's document id: ``prompt_id`` before the draft, ``doc_id`` from the draft on."""
    return "doc_id" if "doc_id" in record else "prompt_id"


def document_id_of(record: dict) -> str:
    return record[id_field_of(record)]


@dataclass(frozen=True)
class StageFile:
    folder_key: str   # which folder it lives in (``sdf_run_layouts.FOLDER_KEYS``)
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


def description_write_prompt_files(config: dict, prompts_directory: Path) -> list:
    """The prompt files the planning prompts and everything built on the dealt variables need."""
    return [utils.sdf_template_path(prompts_directory, 1).name, "variables.txt", "preamble.txt"]


def _no_files(config: dict, prompts_directory: Path) -> list:
    return []


def _active(config: dict, prompts_directory: Path) -> None:
    return None


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
    """Everything about one step. A step with no model call sets only ``name``, ``folder_key`` and the seeding fields."""
    name: str
    folder_key: str                               # ``shared/sdf_run_layouts.py``: the layer, folder and cost-log tag come from the layout
    seed_from_instead: str | None = None          # the step to seed from when this one cannot be seeded
    model_setting_keys: tuple = ()                # ``sdf.<key>``; the first one set wins
    required_prompt_files: Callable = _no_files   # (config, prompts_directory) -> files
    reason_inactive: Callable = _active           # (config, prompts_directory) -> the printed reason, or None
    reads: Callable | None = None                 # run -> the step's documents
    seeds_from: Callable | None = None            # run -> the StageFile whose ids seeding selects from; None: not seedable
    applies: Callable = _always                   # (document, run) -> False when the step skips the document
    finished_elsewhere: Callable = _never         # (document, run) -> True when another step finished it
    group: Callable | None = None                 # (documents, run) -> {group key: list}; None: groups of one document
    build_request: Callable | None = None
    read_reply: Callable | None = None
    records: Callable | None = None               # (documents, outcomes, run) -> every record the outcomes write
    writes: tuple = ()                            # stage files (seeding copies them)
    checkpoint_file_name: str | Callable | None = None   # the checkpoint file in the folder (or a function of the active layout giving it): a group's key is marked after its records
    writes_as_replies_arrive: bool = False        # write each outcome the moment it arrives
    policy: StepPolicy = StepPolicy()
    call_noun: str = "API call"
    unit_noun: str = "document"
    name_in_messages: Callable | None = None      # () -> what stop messages start with instead of the stage tag
    progress_line: Callable | None = None
    after_run: Callable | None = None
    stop_message: Callable | None = None          # (documents, result, run) -> the exit message when a failed group stops the run

    @property
    def makes_model_calls(self) -> bool:
        return self.build_request is not None

    @property
    def stage_tag(self) -> str | None:
        """The cost-log tag of the step's calls in the layout of the run being written; None for a step with no call."""
        return self.stage_tag_in(sdf_run_layouts.active_layout())

    def stage_tag_in(self, layout) -> str | None:
        """The cost-log tag of the step's calls in a layout; None for a step with no call."""
        return layout.stage_tag(self.name) if self.makes_model_calls else None

    def layer_in(self, layout) -> int:
        """The ``--layer`` number of the step in a layout."""
        return layout.layer_number_of_key[self.folder_key]

    def model_for(self, config: dict):
        """The model the step is configured with (the first setting key that is set; a renamed key also under its old name)."""
        return next((model_setting(config["sdf"], key) for key in self.model_setting_keys
                     if model_setting(config["sdf"], key)), None)

    def model_request(self, document: dict, user_message: str, system_prompt: str | None, run: Run,
                      **settings) -> ModelRequest:
        """The request for ``document``: the step's tag and model, an empty system prompt when none, and per-step ``settings``."""
        return ModelRequest(
            user_message=user_message, system_prompt=system_prompt or "", model=self.model_for(run.config),
            stage=self.stage_tag, item_id=document_id_of(document), **settings)

    @property
    def systemic_stop_message(self) -> str:
        """The exit message when every pending call of the step failed."""
        label = self.name_in_messages() if self.name_in_messages else self.stage_tag
        return systemic_stop_message_for(label, self.call_noun, self.unit_noun)
