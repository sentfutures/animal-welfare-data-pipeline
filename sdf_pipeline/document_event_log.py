"""The append-only ``document_events.jsonl`` of a run: one event per step outcome per document.

Each event from a model call records the hash of the request that was sent. Events are appended as
replies arrive, under a lock, so a slow call never holds back the record of finished calls behind
it. Order comes from the deal, never from the log. Today's stage files stay the source of truth and
the checkpoint; the log is written alongside them.

``read_document_events`` ignores an unparseable last line (what an interrupted write leaves) and
raises for an unparseable line anywhere else. ``fold_document_events`` folds the events into the
latest accepted update, the latest failure and the removal per document and step.
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
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
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
            document.request_hash_by_step[step] = event.get("request_hash")
        elif outcome == "failed":
            document.unresolved_failure_by_step[step] = {"kind": event["failure_kind"], "reason": event["reason"]}
        elif outcome == "removed":
            document.removal_by_step[step] = {"reason_code": event["reason_code"], "detail": event["detail"]}
    return folded
