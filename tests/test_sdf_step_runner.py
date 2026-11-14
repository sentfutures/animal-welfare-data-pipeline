"""StepRunner on a toy DocumentStep with an injected send function: no model, no API stub.

Covers pending selection, stage files in input order, progress
lines, each stop rule, and that Failed marks nothing as done.
"""

import json
import threading

import pytest

from sdf_pipeline.shared_step_machinery.step_runner import StepRunner, send_model_request
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, DocumentStep, Failed, FailureKind, ModelReply, ModelRequest, Removed, StepPolicy,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall)
from shared import api, utils


class ToyStep(DocumentStep):
    """Document is a dict with id, applies, and optional text. Reply "FAIL" is unusable, "REMOVE" removes."""
    step_name = "toy"
    stage_file_name = "toy_records.jsonl"
    systemic_stop_message = "toy: every pending call failed (systemic)"

    def __init__(self, policy=StepPolicy()):
        self.policy = policy
        self.outcomes_by_document_id = {}   # what each processed document's outcome was, for the tests

    def document_id_of(self, document):
        return document["id"]

    def applies_to(self, document):
        return document.get("applies", True)

    def build_request(self, document):
        return ModelRequest(user_message=f"say {document['id']}", system_prompt="sys", model="m",
                            stage="toy_stage", item_id=document["id"], max_tokens=10)

    def interpret(self, document, reply):
        if reply.text == "FAIL":
            return Failed(FailureKind.UNUSABLE_ANSWER, "unusable")
        if reply.text == "REMOVE":
            return Removed("toy_removal", "removed for the test")
        return Accepted(reply.text)

    def stage_file_record(self, document, update):
        return {"doc_id": document["id"], "content": update}

    def progress_line(self, document, outcome, progress):
        self.outcomes_by_document_id[document["id"]] = outcome
        return f"  {document['id']}: {type(outcome).__name__}"


def make_runner(tmp_path, step=None, send=None, workers=1):
    stage_directory = tmp_path / "layer_x"
    stage_directory.mkdir(parents=True, exist_ok=True)
    send = send or (lambda request: ModelReply(f"reply to {request.item_id}", "end_turn"))
    return StepRunner(step or ToyStep(), stage_directory, workers=workers, send=send), stage_directory


def docs(*ids, **extra):
    return [{"id": i, **extra} for i in ids]


def test_pending_selection_skips_inapplicable_finished_and_checkpointed(tmp_path):
    stage_directory = tmp_path / "layer_x"
    utils.append_jsonl({"doc_id": "b", "content": "old"}, stage_directory / "toy_records.jsonl")
    utils.Checkpoint(stage_directory / "_checkpoint.json").mark_done("c")
    runner, _ = make_runner(tmp_path)
    documents = docs("a", "b", "c", "e") + [{"id": "d", "applies": False}]
    assert [d["id"] for d in runner.select_pending(documents)] == ["a", "e"]


def test_stage_file_in_input_order_with_random_arrival(tmp_path):
    import random
    delays = random.Random(3)

    def send(request):
        threading.Event().wait(delays.random() * 0.02)
        return ModelReply("ok", "end_turn")

    runner, stage_directory = make_runner(tmp_path, send=send, workers=4)
    ids = [f"d{i}" for i in range(12)]
    result = runner.run(docs(*ids))
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ids
    assert [r["doc_id"] for r in result.records_in_input_order] == ids
    assert utils.Checkpoint(stage_directory / "_checkpoint.json").done_count == 12


def test_progress_lines_in_input_order(tmp_path, capsys):
    runner, _ = make_runner(tmp_path, workers=3)
    runner.run(docs("a", "b", "c"))
    assert capsys.readouterr().out.splitlines() == ["  a: Accepted", "  b: Accepted", "  c: Accepted"]


def test_failed_marks_nothing_done_and_is_not_written_and_siblings_are_kept(tmp_path):
    def send(request):
        if request.item_id == "b":
            raise RuntimeError("usage policy")
        return ModelReply("FAIL" if request.item_id == "c" else "fine", "end_turn")

    step = ToyStep()
    runner, stage_directory = make_runner(tmp_path, step=step, send=send)
    result = runner.run(docs("a", "b", "c"))
    assert result.failed_document_ids == ["b", "c"]
    checkpoint = utils.Checkpoint(stage_directory / "_checkpoint.json")
    assert checkpoint.is_done("a") and not checkpoint.is_done("b") and not checkpoint.is_done("c")
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a"]
    assert result.thrown_call_document_ids == ["b"]
    assert step.outcomes_by_document_id["b"] == Failed(FailureKind.THROWN_CALL, "error: RuntimeError: usage policy")
    assert step.outcomes_by_document_id["c"] == Failed(FailureKind.UNUSABLE_ANSWER, "unusable")
    # a second run retries exactly the failed documents
    second_runner, _ = make_runner(tmp_path, send=lambda request: ModelReply("again", "end_turn"))
    assert [d["id"] for d in second_runner.select_pending(docs("a", "b", "c"))] == ["b", "c"]


def test_systemic_stop_when_every_pending_call_threw(tmp_path):
    def send(request):
        raise RuntimeError("auth failure")

    runner, _ = make_runner(tmp_path, send=send)
    with pytest.raises(SystemExit, match="systemic"):
        runner.run(docs("a", "b"))


def test_unusable_answers_do_not_count_towards_the_systemic_stop_by_default(tmp_path):
    runner, stage_directory = make_runner(tmp_path, send=lambda request: ModelReply("FAIL", "end_turn"))
    result = runner.run(docs("a", "b"))  # no SystemExit
    assert result.failed_document_ids == ["a", "b"]


def test_unusable_answers_count_when_the_step_says_so(tmp_path):
    policy = StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset(FailureKind))
    runner, _ = make_runner(tmp_path, step=ToyStep(policy), send=lambda request: ModelReply("FAIL", "end_turn"))
    with pytest.raises(SystemExit, match="systemic"):
        runner.run(docs("a", "b"))


def test_no_systemic_stop_when_something_succeeded_or_nothing_was_pending(tmp_path):
    def send(request):
        if request.item_id == "a":
            raise RuntimeError("one bad document")
        return ModelReply("ok", "end_turn")

    runner, _ = make_runner(tmp_path, send=send)
    assert runner.run(docs("a", "b")).failed_document_ids == ["a"]
    empty_runner, _ = make_runner(tmp_path / "other")
    assert empty_runner.run([]).pending_document_ids == []


def test_a_step_whose_failures_stop_the_run_writes_records_first_then_stops(tmp_path):
    policy = StepPolicy(failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP)

    def send(request):
        return ModelReply("FAIL" if request.item_id == "b" else "ok", "end_turn")

    runner, stage_directory = make_runner(tmp_path, step=ToyStep(policy), send=send)
    with pytest.raises(SystemExit, match="1 document"):
        runner.run(docs("a", "b", "c"))
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a", "c"]


def test_removed_writes_no_stage_record(tmp_path):
    step = ToyStep()
    runner, stage_directory = make_runner(tmp_path, step=step, send=lambda request: ModelReply("REMOVE", "end_turn"))
    runner.run(docs("a"))
    assert utils.load_jsonl(stage_directory / "toy_records.jsonl") == []
    assert step.outcomes_by_document_id["a"] == Removed("toy_removal", "removed for the test")


def test_a_slow_reply_does_not_change_the_input_order_of_the_stage_file(tmp_path):
    fast_reply_sent = threading.Event()

    def send(request):
        if request.item_id == "slow":
            fast_reply_sent.wait(timeout=5)   # the slow reply arrives after the fast one
        else:
            fast_reply_sent.set()
        return ModelReply(f"reply {request.item_id}", "end_turn")

    runner, stage_directory = make_runner(tmp_path, send=send, workers=2)
    runner.run(docs("slow", "fast"))
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["slow", "fast"]


def test_every_request_carries_the_steps_stage_tag_and_requests_differ_by_document(tmp_path):
    seen = []
    runner, _ = make_runner(tmp_path, send=lambda request: (seen.append(request), ModelReply("ok", "end_turn"))[1])
    runner.run(docs("a", "b"))
    assert [request.stage for request in seen] == ["toy_stage", "toy_stage"]
    assert len({request.request_hash() for request in seen}) == 2


def test_request_hash_changes_with_any_setting():
    base = ModelRequest(user_message="u", system_prompt="s", model="m", stage="t", item_id="i")
    variants = [
        ModelRequest(user_message="u2", system_prompt="s", model="m", stage="t", item_id="i"),
        ModelRequest(user_message="u", system_prompt="s2", model="m", stage="t", item_id="i"),
        ModelRequest(user_message="u", system_prompt="s", model="m2", stage="t", item_id="i"),
        ModelRequest(user_message="u", system_prompt="s", model="m", stage="t2", item_id="i"),
        ModelRequest(user_message="u", system_prompt="s", model="m", stage="t", item_id="i", max_tokens=5),
        ModelRequest(user_message="u", system_prompt="s", model="m", stage="t", item_id="i", cache_system=True),
        ModelRequest(user_message="u", system_prompt="s", model="m", stage="t", item_id="i", temperature=0.0),
    ]
    assert len({base.request_hash(), *(v.request_hash() for v in variants)}) == 1 + len(variants)
    assert base.request_hash() == ModelRequest(user_message="u", system_prompt="s", model="m",
                                               stage="t", item_id="i").request_hash()


def test_send_model_request_passes_only_the_settings_that_are_set(monkeypatch):
    calls = []

    def fake_call_claude(user_message, **keyword_arguments):
        calls.append((user_message, keyword_arguments))
        return "text", "end_turn"

    monkeypatch.setattr(api, "call_claude", fake_call_claude)
    reply = send_model_request(ModelRequest(user_message="u", system_prompt="s", model=None, stage="t", item_id="i"))
    assert reply == ModelReply("text", "end_turn")
    assert calls[-1] == ("u", {"system_prompt": "s", "model": None, "stage": "t", "item_id": "i",
                               "return_stop_reason": True})
    send_model_request(ModelRequest(user_message="u", system_prompt="s", model="m", stage="t", item_id="i",
                                    max_tokens=6000, cache_system=True, temperature=0.5, json_schema={"a": 1}))
    assert calls[-1][1] == {"system_prompt": "s", "model": "m", "stage": "t", "item_id": "i",
                            "return_stop_reason": True, "max_tokens": 6000, "cache_system": True,
                            "temperature": 0.5, "json_schema": {"a": 1}}


def test_run_returns_existing_records_for_documents_the_step_no_longer_applies_to(tmp_path):
    stage_directory = tmp_path / "layer_x"
    utils.append_jsonl({"doc_id": "b", "content": "old b"}, stage_directory / "toy_records.jsonl")
    utils.append_jsonl({"doc_id": "d", "content": "old d"}, stage_directory / "toy_records.jsonl")
    requested = []

    def send(request):
        requested.append(request.item_id)
        return ModelReply(f"new {request.item_id}", "end_turn")

    runner, _ = make_runner(tmp_path, send=send)
    documents = [{"id": "a"}, {"id": "b", "applies": False}, {"id": "c"}, {"id": "d"}]
    result = runner.run(documents)
    assert requested == ["a", "c"]
    # earlier records in input order (b is inapplicable but kept), then this run's, in input order
    assert [(r["doc_id"], r["content"]) for r in result.records_in_input_order] == [
        ("b", "old b"), ("d", "old d"), ("a", "new a"), ("c", "new c")]


# ---------------------------------------------------------------- the declared exception policy

class ToyStepWithBrokenAndFragileDocuments(ToyStep):
    """A document with ``"broken_template": True`` raises in ``build_request`` (a template that cannot be
    filled in); a reply "EXPLODE" raises in ``interpret``."""

    def build_request(self, document):
        if document.get("broken_template"):
            raise KeyError("missing_placeholder")
        return super().build_request(document)

    def interpret(self, document, reply):
        if reply.text == "EXPLODE":
            raise ValueError("parser bug")
        return super().interpret(document, reply)


def test_a_prompt_building_error_is_a_recorded_failure_for_that_document_only(tmp_path, capsys):
    sent = []

    def send(request):
        sent.append(request.item_id)
        return ModelReply("fine", "end_turn")

    step = ToyStepWithBrokenAndFragileDocuments()
    runner, stage_directory = make_runner(tmp_path, step=step, send=send)
    result = runner.run([{"id": "a"}, {"id": "b", "broken_template": True}, {"id": "c"}])
    assert sent == ["a", "c"]
    assert result.failed_document_ids == ["b"] and result.thrown_call_document_ids == ["b"]
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a", "c"]
    assert not utils.Checkpoint(stage_directory / "_checkpoint.json").is_done("b")
    assert step.outcomes_by_document_id["b"] == Failed(FailureKind.THROWN_CALL, "error: KeyError: 'missing_placeholder'")
    assert "  b: Failed" in capsys.readouterr().out


def test_a_resume_retries_a_document_whose_prompt_could_not_be_built(tmp_path):
    runner, stage_directory = make_runner(tmp_path, step=ToyStepWithBrokenAndFragileDocuments())
    runner.run([{"id": "a"}, {"id": "b", "broken_template": True}])
    sent = []

    def send(request):
        sent.append(request.item_id)
        return ModelReply("fixed", "end_turn")

    resumed, _ = make_runner(tmp_path, step=ToyStepWithBrokenAndFragileDocuments(), send=send)
    assert [d["id"] for d in resumed.select_pending([{"id": "a"}, {"id": "b"}])] == ["b"]
    resumed.run([{"id": "a"}, {"id": "b"}])
    assert sent == ["b"]
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a", "b"]


def test_a_template_broken_for_every_document_stops_the_run_with_no_model_call(tmp_path):
    sent = []
    runner, _ = make_runner(
        tmp_path, step=ToyStepWithBrokenAndFragileDocuments(), send=lambda request: sent.append(request) or None)
    with pytest.raises(SystemExit, match="systemic"):
        runner.run([{"id": "a", "broken_template": True}, {"id": "b", "broken_template": True}])
    assert sent == []


def test_an_error_in_interpret_ends_the_run_unless_the_step_declares_it_a_thrown_call(tmp_path):
    explode = lambda request: ModelReply("EXPLODE", "end_turn")  # noqa: E731
    runner, _ = make_runner(tmp_path, step=ToyStepWithBrokenAndFragileDocuments(), send=explode)
    with pytest.raises(ValueError, match="parser bug"):
        runner.run([{"id": "a"}])
    declared = StepPolicy(errors_counted_as_thrown_calls=(
        WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND | WhereAnErrorIsAThrownCall.INTERPRET))
    step = ToyStepWithBrokenAndFragileDocuments(declared)
    runner, _ = make_runner(tmp_path / "declared", step=step, send=explode)
    with pytest.raises(SystemExit, match="systemic"):
        runner.run([{"id": "a"}, {"id": "b"}])
    assert set(step.outcomes_by_document_id.values()) == {Failed(FailureKind.THROWN_CALL, "error: ValueError: parser bug")}
    assert set(step.outcomes_by_document_id) == {"a", "b"}


def test_a_place_the_step_does_not_declare_lets_the_exception_through(tmp_path):
    only_send = StepPolicy(errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.SEND)
    runner, _ = make_runner(tmp_path, step=ToyStepWithBrokenAndFragileDocuments(only_send))
    with pytest.raises(KeyError):
        runner.run([{"id": "a", "broken_template": True}])
    no_send = StepPolicy(errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD)

    def send(request):
        raise RuntimeError("down")

    runner, _ = make_runner(tmp_path / "other", step=ToyStepWithBrokenAndFragileDocuments(no_send), send=send)
    with pytest.raises(RuntimeError, match="down"):
        runner.run([{"id": "a"}])
