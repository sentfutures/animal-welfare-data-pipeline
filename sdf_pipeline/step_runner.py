"""``StepRunner``: the work every SDF layer repeats, for a ``DocumentStep``.

Pick the pending documents, fan out over a thread pool, call the model, append an event as each
reply arrives (under the event log's lock), then on the calling thread, in input order, write the
step's stage file, mark the checkpoint, print progress and apply the step's stop rules. Workers
only build the request, call the model and interpret the reply; every stage file write and
checkpoint mark stays on the main thread, in input order, exactly as the layers did before.

A document is pending when it applies to the step, has no record in the stage file and is not
marked done in the checkpoint. ``Failed`` marks nothing done, so a resume retries it. ``Removed``
marks nothing done unless the step supplies a record through ``stage_file_record_for_removal``.
"""

from dataclasses import dataclass, field

from shared import api, openai_api, utils
from sdf_pipeline.document_event_log import DocumentEventLog, fold_document_events, read_document_events
from sdf_pipeline.update_steps import (
    Accepted, CorpusStep, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, Removed, WhatAFailedDocumentDoes)


@dataclass
class StepRunResult:
    """``records_in_input_order``: the stage-file records of documents finished before this run
    (in input order) followed by the ones written in this run (in input order)."""
    records_in_input_order: list = field(default_factory=list)
    accepted_document_ids: list = field(default_factory=list)
    removed_document_ids: list = field(default_factory=list)
    failed_document_ids: list = field(default_factory=list)
    pending_document_ids: list = field(default_factory=list)


def send_model_request(request: ModelRequest) -> ModelReply:
    """Send the request through ``shared.api.call_claude`` with exactly the keyword arguments a
    layer passes today: settings left at their default are not passed."""
    if request.openai_schema_name is not None:
        # The description rewrite's OpenAI route (a ``gpt-`` model), with the arguments it always passed.
        text, stop_reason = openai_api.call_openai(
            request.user_message,
            system_prompt=request.system_prompt,
            model=request.model,
            max_output_tokens=request.max_tokens,
            reasoning_effort=request.openai_reasoning_effort,
            stage=request.stage,
            item_id=request.item_id,
            json_schema=request.json_schema,
            schema_name=request.openai_schema_name,
        )
        return ModelReply(text=text, stop_reason=stop_reason)
    keyword_arguments = dict(
        system_prompt=request.system_prompt,
        model=request.model,
        stage=request.stage,
        item_id=request.item_id,
        return_stop_reason=True,
    )
    if request.max_tokens is not None:
        keyword_arguments["max_tokens"] = request.max_tokens
    if request.cache_system:
        keyword_arguments["cache_system"] = True
    if request.temperature is not None:
        keyword_arguments["temperature"] = request.temperature
    if request.json_schema is not None:
        keyword_arguments["json_schema"] = request.json_schema
    text, stop_reason = api.call_claude(request.user_message, **keyword_arguments)
    return ModelReply(text=text, stop_reason=stop_reason)


class StepRunner:
    def __init__(self, step: DocumentStep, stage_directory, event_log: DocumentEventLog,
                 workers: int = 1, send=send_model_request, existing_records_by_document_id: dict | None = None,
                 checkpoint: utils.Checkpoint | None = None):
        self.step = step
        self.stage_directory = stage_directory
        self.event_log = event_log
        self.workers = workers
        self.send = send
        self.stage_file_path = stage_directory / step.stage_file_name
        self.checkpoint = checkpoint if checkpoint is not None else utils.Checkpoint(stage_directory / "_checkpoint.json")
        self.existing_records_by_document_id = (
            existing_records_by_document_id if existing_records_by_document_id is not None
            else {record["doc_id"]: record for record in utils.load_jsonl(self.stage_file_path)})

    def select_pending(self, documents: list) -> list:
        step = self.step
        return [
            document for document in documents
            if step.applies_to(document)
            and step.document_id_of(document) not in self.existing_records_by_document_id
            and not self.checkpoint.is_done(step.document_id_of(document))
        ]

    def _answer_one(self, document):
        """Worker: build, send, interpret, record the event. Returns (request, outcome)."""
        step = self.step
        try:
            request = step.build_request(document)
        except Exception as error:
            if not step.build_request_errors_count_as_thrown_calls:
                raise
            outcome = Failed(FailureKind.THROWN_CALL, f"error: {type(error).__name__}: {error}")
            self._append_event(document, None, None, outcome)
            return outcome
        request_hash = request.request_hash()
        try:
            reply = self.send(request)
        except Exception as error:
            # One poison document must not kill the step and discard its siblings' in-flight work.
            outcome = Failed(FailureKind.THROWN_CALL, f"error: {type(error).__name__}: {error}")
        else:
            outcome = step.interpret(document, reply)
        self._append_event(document, request, request_hash, outcome)
        return outcome

    def _append_event(self, document, request, request_hash, outcome):
        step = self.step
        if isinstance(outcome, Accepted):
            outcome_name, detail = "accepted", {"update": step.stage_file_record(document, outcome.update)}
        elif isinstance(outcome, Removed):
            outcome_name, detail = "removed", {"reason_code": outcome.reason_code, "detail": outcome.detail}
        else:
            outcome_name, detail = "failed", {"failure_kind": outcome.kind.value, "reason": outcome.reason}
        self.event_log.append_outcome_event(
            document_id=step.document_id_of(document), step_name=step.step_name, outcome=outcome_name,
            request_hash=request_hash, stage=request.stage if request is not None else None, detail=detail)

    def run(self, documents: list) -> StepRunResult:
        step = self.step
        policy = step.policy
        considered = [d for d in documents if step.applies_to(d)]
        pending = self.select_pending(documents)
        result = StepRunResult(
            records_in_input_order=[
                self.existing_records_by_document_id[step.document_id_of(d)] for d in considered
                if step.document_id_of(d) in self.existing_records_by_document_id],
            pending_document_ids=[step.document_id_of(d) for d in pending])
        counted_failures = 0
        for document, outcome in zip(pending, utils.parallel_map(self._answer_one, pending, self.workers)):
            document_id = step.document_id_of(document)
            if isinstance(outcome, Failed):
                result.failed_document_ids.append(document_id)
                if outcome.kind in policy.failure_kinds_counting_towards_the_systemic_stop:
                    counted_failures += 1
            elif isinstance(outcome, Removed):
                result.removed_document_ids.append(document_id)
                record = step.stage_file_record_for_removal(document, outcome)
                if record is not None:
                    result.records_in_input_order.append(record)
                    utils.append_jsonl(record, self.stage_file_path)
                    self.checkpoint.mark_done(document_id)
            elif isinstance(outcome, Accepted):
                record = step.stage_file_record(document, outcome.update)
                result.records_in_input_order.append(record)
                utils.append_jsonl(record, self.stage_file_path)
                self.checkpoint.mark_done(document_id)
                result.accepted_document_ids.append(document_id)
            line = step.progress_line(document, outcome)
            if line is not None:
                print(line)
        if pending and counted_failures == len(pending):
            raise SystemExit(step.systemic_stop_message)
        step.after_documents_are_processed(result)
        if result.failed_document_ids and policy.failed_document is WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP:
            raise SystemExit(step.message_when_failed_documents_stop_the_run(result))
        return result


def run_corpus_step(step: CorpusStep, documents: list, event_log: DocumentEventLog) -> dict:
    """Apply a corpus step (no model call) and record its outcomes: {document id: StepOutcome}.

    The step runs every time (it is deterministic and cheap). An event is appended for a document
    only when its outcome differs from the latest recorded one for this step, so a resume that
    changes nothing appends nothing and a new threshold appends exactly the changed outcomes.
    """
    outcomes = step.apply(documents)
    folded = fold_document_events(read_document_events(event_log.path))
    for document_id, outcome in outcomes.items():
        if isinstance(outcome, Accepted):
            outcome_name, detail, latest = "accepted", {"update": outcome.update}, ("accepted", outcome.update)
        elif isinstance(outcome, Removed):
            outcome_name = "removed"
            detail = {"reason_code": outcome.reason_code, "detail": outcome.detail}
            latest = ("removed", {"reason_code": outcome.reason_code, "detail": outcome.detail})
        else:
            raise TypeError(f"{step.step_name}: a corpus step returned {type(outcome).__name__} for {document_id}")
        recorded = folded.get(document_id)
        if recorded is not None and recorded.latest_outcome_by_step.get(step.step_name) == latest:
            continue
        event_log.append_outcome_event(
            document_id=document_id, step_name=step.step_name, outcome=outcome_name,
            request_hash=None, stage=None, detail=detail)
    return outcomes
