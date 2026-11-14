"""The vocabulary of SDF steps: what a step sends (``ModelRequest``), what it can answer (``Accepted``, ``Removed``,
``Failed``), what the runner records (``OutcomeRecords``) and its stop rules and counts (``StepPolicy``, ``StepProgress``).
The step record is in ``step_record.py``; the runner is in ``step_runner.py``."""

import time
from dataclasses import dataclass, field
from enum import Enum, Flag, auto


@dataclass(frozen=True)
class ModelRequest:
    """One request to a model, with every setting a step can pass.
    ``max_tokens`` None takes the config default; ``stage`` is the stage tag (it selects the effort); ``item_id`` is the document (or batch) id."""
    user_message: str
    system_prompt: str
    model: str | None
    stage: str
    item_id: str
    max_tokens: int | None = None
    cache_system: bool = False
    temperature: float | None = None
    json_schema: dict | None = None
    openai_reasoning_effort: str | None = None   # the description rewrite's OpenAI route only; None for Claude
    openai_schema_name: str | None = None        # likewise


PROGRESS_INTERVAL_SECONDS = 180  # throttled progress lines are printed no more often than this


class ThrottledProgressClock:
    """Decides when a throttled progress line is due (``PROGRESS_INTERVAL_SECONDS`` since the last one)."""

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


@dataclass
class StepProgress:
    """The counts and clock a step's progress lines are built from. The runner updates it; a step only reads it."""
    pending_count: int = 0
    groups_seen: int = 0
    answered: int = 0   # every outcome that is not a failure
    failed: int = 0
    removed: int = 0
    accepted_updates: list = field(default_factory=list)
    seconds_of_last_group: float = 0.0
    clock: ThrottledProgressClock = field(default_factory=ThrottledProgressClock)

    def record(self, outcomes, seconds: float = 0.0) -> None:
        self.groups_seen += 1
        self.seconds_of_last_group = seconds
        for outcome in outcomes:
            if isinstance(outcome, Failed):
                self.failed += 1
                continue
            self.answered += 1
            if isinstance(outcome, Removed):
                self.removed += 1
            elif isinstance(outcome, Accepted):
                self.accepted_updates.append(outcome.update)


def reason_without_error_prefix(reason: str) -> str:
    """A thrown call's failure reason without its leading ``error: ``."""
    return reason.removeprefix("error: ")


def line_for_document_left_to_retry(document_id: str, problem: str, ending: str = "will retry on resume") -> str:
    """The progress line for a document (or batch) whose call failed and which a resume runs again."""
    return f"  {document_id}: {problem} — {ending}"


@dataclass(frozen=True)
class ModelReply:
    """The text and stop reason of a reply; ``attempt`` is which send of the same request it answers (1 for the first)."""
    text: str
    stop_reason: str | None
    attempt: int = 1


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
    step has structured facts to keep (an INCOHERENT plan's text and description)."""
    reason_code: str
    detail: object


@dataclass(frozen=True)
class Failed:
    """The call threw or its answer was unusable. The document is not
    marked done, and a resume retries it."""
    kind: FailureKind
    reason: str


StepOutcome = Accepted | Removed | Failed


@dataclass(frozen=True)
class OutcomeRecords:
    """What the runner writes for a group's outcomes: ``records`` as (file name, record) pairs in order, and
    ``replacing`` as (file name, field, value) triples whose earlier records it drops first."""
    records: tuple = ()
    replacing: tuple = ()


class WhatAFailedDocumentDoes(Enum):
    LEFT_BEHIND = "left_behind"                                    # the run carries on without it
    STOPS_THE_RUN_AFTER_THE_STEP = "stops_the_run_after_the_step"  # records are written, then the run stops


class WhereAnErrorIsAThrownCall(Flag):
    """Where an exception from a step's code is recorded as a ``Failed`` ``THROWN_CALL`` rather than ending the run."""
    BUILD = auto()      # ``build_request`` (a prompt template that cannot be filled in for this document)
    SEND = auto()       # the model call (auth, backend, network)
    INTERPRET = auto()  # ``read_reply`` (a reply the step's parser did not expect to fail on)


@dataclass(frozen=True)
class StepPolicy:
    """A step's attempts and failure rules (the exception policy is in ``sdf_pipeline/CLAUDE.md``).
    A thrown call is never retried within the run."""
    maximum_attempts_in_one_run: int = 1
    maximum_attempts_setting: str | None = None
    failure_kinds_counting_towards_the_systemic_stop: frozenset = frozenset({FailureKind.THROWN_CALL})
    failed_document: WhatAFailedDocumentDoes = WhatAFailedDocumentDoes.LEFT_BEHIND
    errors_counted_as_thrown_calls: WhereAnErrorIsAThrownCall = WhereAnErrorIsAThrownCall.SEND

    def __post_init__(self):
        if self.maximum_attempts_in_one_run < 1:
            raise ValueError(f"maximum_attempts_in_one_run is {self.maximum_attempts_in_one_run}; it must be at least 1")


def systemic_stop_message_for(stage_tag: str, call_noun: str, unit_noun: str) -> str:
    """The exit message when every pending call of a step failed, for the step's tag, the word for its
    call ("API call", "scoring call") and the unit it works on ("document", "plan", "batch")."""
    return (
        f"{stage_tag}: every pending {call_noun} failed — this is systemic "
        f"(auth, backend, or network), not per-{unit_noun}; fix and --resume.")
