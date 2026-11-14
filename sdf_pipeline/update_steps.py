"""The vocabulary of SDF update steps: what a step sends, what it can answer, and the three kinds.

A step decides what to ask and what the answer means. It does no file or network input and
output: ``sdf_pipeline/step_runner.py`` does the calling and the stage files.

``ModelRequest`` carries everything a layer passes to ``shared.api.call_claude``. A step outcome
is one of ``Accepted``, ``Removed``, ``Failed`` or ``Created``. The three step kinds are
``DocumentStep`` (one document, one model call per attempt), ``BatchStep`` (a group of documents,
one call per group) and ``CorpusStep`` (every document, no call). ``StepRunner`` runs a ``DocumentStep``, ``BatchStepRunner``
runs a ``BatchStep`` (the screen) and ``run_corpus_step`` runs a ``CorpusStep`` (the deal, the redeal,
the considerations assignment, the score gate and the near-duplicate cull); all are in
``sdf_pipeline/step_runner.py``.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum, Flag, auto


@dataclass(frozen=True)
class ModelRequest:
    """One request to a model, with every setting a layer passes today.

    ``max_tokens`` is None when the layer sets none and takes the config default (the plan and
    score calls). ``stage`` is the stage tag: it selects the reasoning effort from the config's
    ``effort`` map. ``item_id`` is the document id (the batch id for the screen).
    ``openai_reasoning_effort`` and ``openai_schema_name`` are for the description rewrite's
    OpenAI route and stay None for every Claude request.
    """
    user_message: str
    system_prompt: str
    model: str | None
    stage: str
    item_id: str
    max_tokens: int | None = None
    cache_system: bool = False
    temperature: float | None = None
    json_schema: dict | None = None
    openai_reasoning_effort: str | None = None
    openai_schema_name: str | None = None

    def request_hash(self) -> str:
        """SHA-256 of the request as a canonical JSON string, to tell requests apart (used by the
        request print command)."""
        canonical = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PROGRESS_INTERVAL_SECONDS = 180  # throttled progress lines are printed no more often than this


class ThrottledProgressClock:
    """The clock behind a throttled progress line: ``is_due`` once ``PROGRESS_INTERVAL_SECONDS`` have
    passed since the last line, ``mark_printed`` (returns the seconds since the start) when one is
    printed, ``restart`` at the start of each run."""

    def __init__(self):
        self.restart()

    def restart(self) -> None:
        self.started_at = self.last_printed_at = time.monotonic()

    def is_due(self) -> bool:
        return time.monotonic() - self.last_printed_at >= PROGRESS_INTERVAL_SECONDS

    def mark_printed(self) -> float:
        self.last_printed_at = time.monotonic()
        return self.last_printed_at - self.started_at

    def seconds_since_start(self) -> float:
        return time.monotonic() - self.started_at


def reason_without_error_prefix(reason: str) -> str:
    """A thrown call's failure reason without its leading ``error: ``."""
    return reason.removeprefix("error: ")


def line_for_document_left_to_retry(document_id: str, problem: str, ending: str = "will retry on resume") -> str:
    """The progress line for a document (or batch) whose call failed and which a resume runs again."""
    return f"  {document_id}: {problem} — {ending}"


@dataclass(frozen=True)
class ModelReply:
    """What came back: the text and the stop reason (None when the backend gives none)."""
    text: str
    stop_reason: str | None


class FailureKind(Enum):
    THROWN_CALL = "thrown_call"          # network, authentication or backend error
    UNUSABLE_ANSWER = "unusable_answer"  # truncated, missing tags, unparseable


@dataclass(frozen=True)
class Accepted:
    """The answer is good; ``update`` is the typed change to apply (a new text version, scores, ...)."""
    update: object


@dataclass(frozen=True)
class Removed:
    """The document leaves the pipeline. ``detail`` is a short string, or a JSON-safe dict when the
    step has structured facts to keep (the near-duplicate cull's kept document and similarity)."""
    reason_code: str
    detail: object


@dataclass(frozen=True)
class Failed:
    """The call threw or its answer was unusable. The document is not
    marked done, and a resume retries it."""
    kind: FailureKind
    reason: str


@dataclass(frozen=True)
class Created:
    """A new document enters the pipeline (corpus steps only: the deal and the redeal)."""
    new_state: object


StepOutcome = Accepted | Removed | Failed | Created


class WhatAFailedDocumentDoes(Enum):
    LEFT_BEHIND = "left_behind"                                    # the run carries on without it
    STOPS_THE_RUN_AFTER_THE_STEP = "stops_the_run_after_the_step"  # records are written, then the run stops


class WhereAnErrorIsAThrownCall(Flag):
    """The places in a step where an exception raised by the step's code is recorded as a ``Failed``
    ``THROWN_CALL`` for that document (or batch) instead of ending the run. An exception anywhere else
    propagates. The default is ``BUILD | SEND``; see ``StepPolicy``."""
    BUILD = auto()      # ``build_request`` (a prompt template that cannot be filled in for this document)
    SEND = auto()       # the model call (auth, backend, network)
    INTERPRET = auto()  # ``interpret`` (a reply the step's parser did not expect to fail on)


@dataclass(frozen=True)
class StepPolicy:
    """What a step declares about its attempts and failures (see the plan's step table).

    The exception policy, in one place (``errors_counted_as_thrown_calls``; both runners read it):

    * an exception in ``build_request`` is ``Failed(THROWN_CALL, "error: <Type>: <message>")`` for that
      document, with no call made; it is printed, logged and retried on resume;
    * an exception in the model call is the same ``Failed``;
    * an exception in ``interpret`` is the same ``Failed`` when the step declares ``INTERPRET``,
      otherwise it ends the run.

    ``failure_kinds_counting_towards_the_systemic_stop``: the run stops when every pending call
    failed with a kind in this set. ``maximum_attempts_in_one_run`` is 1 except consideration
    scoring: the runner sends the same request again after an ``UNUSABLE_ANSWER`` failure, up to
    that many attempts in one run. A thrown call is never retried within the run.
    """
    maximum_attempts_in_one_run: int = 1
    failure_kinds_counting_towards_the_systemic_stop: frozenset = frozenset({FailureKind.THROWN_CALL})
    failed_document: WhatAFailedDocumentDoes = WhatAFailedDocumentDoes.LEFT_BEHIND
    errors_counted_as_thrown_calls: WhereAnErrorIsAThrownCall = (
        WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND)


class DocumentStep(ABC):
    """A step that sees one document and makes one model call per attempt.

    The "document" is whatever object the step's layer works with (a plan record for the draft
    step, a draft record for the critique step); the step
    names its id with ``document_id_of``.
    """

    step_name: str
    stage_file_name: str
    # The field of a stage file record that holds the document's id (what ``document_id_of`` returns),
    # so the runner can find the documents already in the stage file.
    stage_file_id_field: str = "doc_id"
    policy: StepPolicy = StepPolicy()
    systemic_stop_message: str
    # When True, the runner writes each outcome's stage file record, checkpoint mark, failure record
    # and progress line from the worker, under a lock, the moment its answer arrives (completion
    # order), instead of on the calling thread in input order. For a step whose stage file is its
    # only checkpoint (consideration scoring): a crash must not lose finished paid answers.
    writes_stage_file_as_replies_arrive: bool = False
    # A second stage file for failed documents: the runner appends ``failure_record`` to it.
    failure_file_name: str | None = None

    @abstractmethod
    def document_id_of(self, document) -> str: ...

    @abstractmethod
    def applies_to(self, document) -> bool: ...

    @abstractmethod
    def build_request(self, document) -> ModelRequest: ...

    @abstractmethod
    def interpret(self, document, reply: ModelReply) -> StepOutcome: ...

    @abstractmethod
    def stage_file_record(self, document, update) -> dict:
        """The record the step's stage file holds for an accepted update."""

    def progress_line(self, document, outcome) -> str | None:
        """The line printed once the outcome is written (None prints nothing)."""
        return None

    def stage_file_record_for_removal(self, document, removed: "Removed") -> dict | None:
        """The record to write to the stage file, and mark done in the checkpoint, when the step
        removes a document (None writes nothing and marks nothing). The plan step uses it: an
        INCOHERENT plan is a finished plan whose record says so."""
        return None

    def failure_record(self, document, failed: "Failed") -> dict | None:
        """The record appended to ``failure_file_name`` (in the stage folder) when a document's
        final outcome is a failure; None writes nothing."""
        return None

    def failure_when_no_attempt_is_allowed(self, document) -> "Failed":
        """The failure for a document when the policy allows no attempt at all (zero calls)."""
        return Failed(FailureKind.UNUSABLE_ANSWER, "no attempt allowed")

    def after_documents_are_processed(self, run_result) -> None:
        """Called once after every pending document is processed and the systemic stop check has
        passed, before a failed document stops the run. The default does nothing."""

    def message_when_failed_documents_stop_the_run(self, run_result) -> str:
        """The exit message when a failed document stops the run after the step."""
        return f"{self.step_name}: {len(run_result.failed_document_ids)} document(s) failed; fix and --resume."


class BatchStep(ABC):
    """A step that sees a group of documents and makes one call per group (the screen)."""

    step_name: str
    stage: str  # the stage tag of its calls
    policy: StepPolicy = StepPolicy()
    systemic_stop_message: str

    @abstractmethod
    def group(self, states: list, seed: int, round_number: int) -> list: ...

    @abstractmethod
    def build_request(self, batch) -> ModelRequest: ...

    @abstractmethod
    def interpret(self, batch, reply: ModelReply) -> dict:
        """{document id: StepOutcome} for the members of the batch."""


class CorpusStep(ABC):
    """A step that sees every document and makes no model call (deal, redeal, assignment, gate, cull)."""

    step_name: str

    @abstractmethod
    def apply(self, states: list) -> dict:
        """{document id: StepOutcome}."""
