"""The append-only ``document_events.jsonl`` of a run: one event per step outcome per document.

Each event from a model call records the hash of the request that was sent. Events are appended as
replies arrive, under a lock, so a slow call never holds back the record of finished calls behind
it. Order comes from the deal, never from the log. Today's stage files stay the source of truth and
the checkpoint; the log is written alongside them.

``read_document_events`` ignores an unparseable last line (what an interrupted write leaves) and
raises for an unparseable line anywhere else. ``fold_document_events`` folds the events into the
latest accepted update, the latest failure and the removal per document and step. A later accepted
event lifts an earlier removal of the same step (a corpus step run again with a new threshold).
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DOCUMENT_EVENT_LOG_FILE_NAME = "document_events.jsonl"


class DocumentEventLogCorruptError(ValueError):
    """A line other than the last is not valid JSON."""


class DocumentEventLog:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append_outcome_event(self, *, document_id: str, step_name: str, outcome: str,
                             request_hash: str | None, stage: str | None, detail: dict) -> dict:
        event = {
            "event": "document_step_outcome",
            "recorded_at": datetime.now(UTC).isoformat(),
            "document_id": document_id,
            "step": step_name,
            "outcome": outcome,
            "request_hash": request_hash,
            "stage": stage,
            **detail,
        }
        self._append(event)
        return event

    def _append(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)

    # The three events that are not about one document.

    def append_round_dealt_event(self, *, round_number: int, prompt_ids: list, note: str) -> dict:
        """A screen round was dealt: its hand ids (empty when the removed cards could not be redealt)."""
        event = {"event": "round_dealt", "recorded_at": datetime.now(UTC).isoformat(),
                 "round": round_number, "prompt_ids": list(prompt_ids), "note": note}
        self._append(event)
        return event

    def append_screen_batches_planned_event(self, *, round_number: int, batches: list) -> dict:
        """A round's screen batches were planned, with their members, before the round's first call."""
        event = {"event": "screen_batches_planned", "recorded_at": datetime.now(UTC).isoformat(),
                 "round": round_number, "batches": [dict(batch) for batch in batches]}
        self._append(event)
        return event

    def append_corpus_step_finished_event(self, *, step_name: str, summary: dict) -> dict:
        """A corpus step (the deal, a redeal) finished; ``summary`` holds its JSON-safe facts."""
        event = {"event": "corpus_step_finished", "recorded_at": datetime.now(UTC).isoformat(),
                 "step": step_name, **summary}
        self._append(event)
        return event


def read_document_events(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                break  # an interrupted write leaves a cut-off last line
            raise DocumentEventLogCorruptError(f"{path} line {index + 1} is not valid JSON") from error
    return events


@dataclass
class FoldedDocumentSteps:
    """What the events say about one document: per step name, the latest accepted update, the
    latest failure that no later accepted event followed, and the removal if any."""
    accepted_update_by_step: dict = field(default_factory=dict)
    unresolved_failure_by_step: dict = field(default_factory=dict)
    removal_by_step: dict = field(default_factory=dict)
    request_hash_by_step: dict = field(default_factory=dict)
    latest_outcome_by_step: dict = field(default_factory=dict)  # step -> (outcome, accepted update or removal)


def fold_document_events(events: list[dict]) -> dict:
    """{document id: FoldedDocumentSteps}, in order of first appearance."""
    folded: dict = {}
    for event in events:
        if event.get("event") != "document_step_outcome":
            continue
        document = folded.setdefault(event["document_id"], FoldedDocumentSteps())
        step = event["step"]
        outcome = event["outcome"]
        if outcome == "accepted":
            document.accepted_update_by_step[step] = event["update"]
            document.unresolved_failure_by_step.pop(step, None)
            document.removal_by_step.pop(step, None)  # a later acceptance (a new threshold) lifts an earlier removal
            document.latest_outcome_by_step[step] = ("accepted", event["update"])
            document.request_hash_by_step[step] = event.get("request_hash")
        elif outcome == "failed":
            document.unresolved_failure_by_step[step] = {"kind": event["failure_kind"], "reason": event["reason"]}
        elif outcome == "removed":
            document.removal_by_step[step] = {"reason_code": event["reason_code"], "detail": event["detail"]}
            document.latest_outcome_by_step[step] = ("removed", document.removal_by_step[step])
    return folded


def fold_round_and_batch_plan_events(events: list[dict]) -> dict:
    """What the non-per-document events say about the screen: ``rounds`` ({round: {prompt_ids, note}}),
    ``batches_by_round`` ({round: batches}, the latest plan of each round winning, as a re-planned
    round replaces its earlier batches) and ``corpus_steps_finished`` (each step's summary, in order)."""
    rounds: dict = {}
    batches_by_round: dict = {}
    corpus_steps_finished: list = []
    for event in events:
        kind = event.get("event")
        if kind == "round_dealt":
            rounds[event["round"]] = {"prompt_ids": event["prompt_ids"], "note": event["note"]}
        elif kind == "screen_batches_planned":
            batches_by_round[event["round"]] = event["batches"]
        elif kind == "corpus_step_finished":
            corpus_steps_finished.append({key: value for key, value in event.items() if key not in ("event", "recorded_at")})
    return {"rounds": rounds, "batches_by_round": batches_by_round, "corpus_steps_finished": corpus_steps_finished}
