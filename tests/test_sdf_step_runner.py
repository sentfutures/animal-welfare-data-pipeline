"""The runner on a toy step with an injected send function: no model, no API stub.

Covers pending selection, stage files in input order, progress
lines, each stop rule, and that Failed marks nothing as done.
"""

import threading

import pytest

from sdf_pipeline.shared_step_machinery.step_record import Run, StageFile, Step, one_record_per_accepted_document
from sdf_pipeline.shared_step_machinery.step_runner import find_pending, run_step, send_model_request
from sdf_pipeline.shared_step_machinery.update_steps import (
    Accepted, Failed, FailureKind, ModelReply, ModelRequest, OutcomeRecords, Removed, StepPolicy,
    WhatAFailedDocumentDoes, WhereAnErrorIsAThrownCall)
from shared import api, utils

FOLDER_KEY = "toy_folder"
TOY_RECORDS = StageFile(FOLDER_KEY, "toy_records.jsonl", "doc_id")


def toy_step(policy=StepPolicy(), outcomes_by_document_id=None, **changes):
    """A step whose documents are dicts with ``doc_id``, optionally ``applies``: reply "FAIL" is unusable, "REMOVE"
    removes. ``outcomes_by_document_id`` collects every outcome the runner printed a line for."""
    outcomes_by_document_id = {} if outcomes_by_document_id is None else outcomes_by_document_id

    def build_request(documents, run):
        (document,) = documents
        return ModelRequest(user_message=f"say {document['doc_id']}", system_prompt="sys", model="m",
                            stage="toy_stage", item_id=document["doc_id"], max_tokens=10)

    def read_reply(documents, reply, run):
        (document,) = documents
        if reply.text == "FAIL":
            outcome = Failed(FailureKind.UNUSABLE_ANSWER, "unusable")
        elif reply.text == "REMOVE":
            outcome = Removed("toy_removal", "removed for the test")
        else:
            outcome = Accepted(reply.text)
        return {document["doc_id"]: outcome}

    def progress_line(documents, outcomes, progress, run):
        (document,), (outcome,) = documents, outcomes.values()
        outcomes_by_document_id[document["doc_id"]] = outcome
        return f"  {document['doc_id']}: {type(outcome).__name__}"

    fields = dict(
        name="toy", folder_key=FOLDER_KEY, stage_tag="toy_stage", build_request=build_request, read_reply=read_reply,
        records=one_record_per_accepted_document(
            TOY_RECORDS, lambda document, update, run: {"doc_id": document["doc_id"], "content": update}),
        writes=(TOY_RECORDS,), policy=policy,
        applies=lambda document, run: document.get("applies", True), name_in_messages="toy",
        progress_line=progress_line)
    return Step(**{**fields, **changes})


def toy_run(tmp_path, workers=1):
    stage_directory = tmp_path / "layer_x"
    stage_directory.mkdir(parents=True, exist_ok=True)
    return Run({"workers": workers}, tmp_path, {FOLDER_KEY: stage_directory}), stage_directory


def run_toy(tmp_path, documents, step=None, send=None, workers=1):
    run, stage_directory = toy_run(tmp_path, workers)
    send = send or (lambda request: ModelReply(f"reply to {request.item_id}", "end_turn"))
    return run_step(step or toy_step(), run, documents, send=send), stage_directory


def docs(*ids, **extra):
    return [{"doc_id": i, **extra} for i in ids]


def pending_ids(tmp_path, documents, step=None):
    run, _ = toy_run(tmp_path)
    return [group[0]["doc_id"] for group in find_pending(step or toy_step(), run, documents)[1].values()]


def test_pending_selection_skips_inapplicable_finished_and_marked_by_earlier_code(tmp_path):
    stage_directory = tmp_path / "layer_x"
    stage_directory.mkdir()
    utils.append_jsonl({"doc_id": "b", "content": "old"}, stage_directory / "toy_records.jsonl")
    utils.Checkpoint(stage_directory / "_checkpoint.json").mark_done("c")
    documents = docs("a", "b", "c", "e") + [{"doc_id": "d", "applies": False}]
    assert pending_ids(tmp_path, documents, toy_step(checkpoint_file_name="_checkpoint.json")) == ["a", "e"]
    assert pending_ids(tmp_path, documents) == ["a", "c", "e"]   # a step without a checkpoint file ignores marks


def test_a_step_with_a_checkpoint_file_marks_each_finished_document_after_its_record_and_one_without_writes_none(tmp_path):
    _, stage_directory = run_toy(tmp_path / "plain", docs("a"))
    assert sorted(path.name for path in stage_directory.iterdir()) == ["toy_records.jsonl"]
    step = toy_step(checkpoint_file_name="_checkpoint.json")
    send = lambda request: ModelReply("FAIL" if request.item_id == "b" else "REMOVE" if request.item_id == "c" else "ok", "end_turn")  # noqa: E731
    _, stage_directory = run_toy(tmp_path / "marked", docs("a", "b", "c"), step=step, send=send)
    assert sorted(path.name for path in stage_directory.iterdir()) == ["_checkpoint.json", "toy_records.jsonl"]
    assert sorted(utils.Checkpoint(stage_directory / "_checkpoint.json")._completed) == ["a"]   # only a written record finishes a document


def test_a_document_another_step_finished_is_not_pending(tmp_path):
    step = toy_step(finished_elsewhere=lambda document, run: document["doc_id"] == "b")
    assert pending_ids(tmp_path, docs("a", "b", "c"), step) == ["a", "c"]


def test_stage_file_in_input_order_with_random_arrival(tmp_path):
    import random
    delays = random.Random(3)

    def send(request):
        threading.Event().wait(delays.random() * 0.02)
        return ModelReply("ok", "end_turn")

    ids = [f"d{i}" for i in range(12)]
    result, stage_directory = run_toy(tmp_path, docs(*ids), send=send, workers=4)
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ids
    assert [r["doc_id"] for r in result.records_in_input_order] == ids


def test_progress_lines_in_input_order(tmp_path, capsys):
    run_toy(tmp_path, docs("a", "b", "c"), workers=3)
    assert capsys.readouterr().out.splitlines() == ["  a: Accepted", "  b: Accepted", "  c: Accepted"]


def test_failed_is_not_written_and_siblings_are_kept(tmp_path):
    def send(request):
        if request.item_id == "b":
            raise RuntimeError("usage policy")
        return ModelReply("FAIL" if request.item_id == "c" else "fine", "end_turn")

    outcomes = {}
    result, stage_directory = run_toy(tmp_path, docs("a", "b", "c"), step=toy_step(outcomes_by_document_id=outcomes), send=send)
    assert result.failed_document_ids == ["b", "c"]
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a"]
    assert result.thrown_call_document_ids == ["b"]
    assert outcomes["b"] == Failed(FailureKind.THROWN_CALL, "error: RuntimeError: usage policy")
    assert outcomes["c"] == Failed(FailureKind.UNUSABLE_ANSWER, "unusable")
    # a second run retries exactly the failed documents
    assert pending_ids(tmp_path, docs("a", "b", "c")) == ["b", "c"]


def test_systemic_stop_when_every_pending_call_threw(tmp_path):
    def send(request):
        raise RuntimeError("auth failure")

    with pytest.raises(SystemExit, match="systemic"):
        run_toy(tmp_path, docs("a", "b"), send=send)


def test_unusable_answers_do_not_count_towards_the_systemic_stop_by_default(tmp_path):
    result, _ = run_toy(tmp_path, docs("a", "b"), send=lambda request: ModelReply("FAIL", "end_turn"))  # no SystemExit
    assert result.failed_document_ids == ["a", "b"]


def test_unusable_answers_count_when_the_step_says_so(tmp_path):
    policy = StepPolicy(failure_kinds_counting_towards_the_systemic_stop=frozenset(FailureKind))
    with pytest.raises(SystemExit, match="systemic"):
        run_toy(tmp_path, docs("a", "b"), step=toy_step(policy), send=lambda request: ModelReply("FAIL", "end_turn"))


def test_no_systemic_stop_when_something_succeeded_or_nothing_was_pending(tmp_path):
    def send(request):
        if request.item_id == "a":
            raise RuntimeError("one bad document")
        return ModelReply("ok", "end_turn")

    assert run_toy(tmp_path, docs("a", "b"), send=send)[0].failed_document_ids == ["a"]
    assert run_toy(tmp_path / "other", [])[0].pending_document_ids == []


def test_a_step_whose_failures_stop_the_run_writes_records_first_then_stops(tmp_path):
    policy = StepPolicy(failed_document=WhatAFailedDocumentDoes.STOPS_THE_RUN_AFTER_THE_STEP)

    def send(request):
        return ModelReply("FAIL" if request.item_id == "b" else "ok", "end_turn")

    with pytest.raises(SystemExit, match="toy: 1 document"):
        run_toy(tmp_path, docs("a", "b", "c"), step=toy_step(policy), send=send)
    assert [r["doc_id"] for r in utils.load_jsonl(tmp_path / "layer_x" / "toy_records.jsonl")] == ["a", "c"]


def test_removed_writes_no_stage_record(tmp_path):
    outcomes = {}
    _, stage_directory = run_toy(tmp_path, docs("a"), step=toy_step(outcomes_by_document_id=outcomes),
                                 send=lambda request: ModelReply("REMOVE", "end_turn"))
    assert utils.load_jsonl(stage_directory / "toy_records.jsonl") == []
    assert outcomes["a"] == Removed("toy_removal", "removed for the test")


def test_a_slow_reply_does_not_change_the_input_order_of_the_stage_file(tmp_path):
    fast_reply_sent = threading.Event()

    def send(request):
        if request.item_id == "slow":
            fast_reply_sent.wait(timeout=5)   # the slow reply arrives after the fast one
        else:
            fast_reply_sent.set()
        return ModelReply(f"reply {request.item_id}", "end_turn")

    _, stage_directory = run_toy(tmp_path, docs("slow", "fast"), send=send, workers=2)
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["slow", "fast"]


def test_every_request_carries_the_steps_stage_tag_and_requests_differ_by_document(tmp_path):
    seen = []
    run_toy(tmp_path, docs("a", "b"), send=lambda request: (seen.append(request), ModelReply("ok", "end_turn"))[1])
    assert [request.stage for request in seen] == ["toy_stage", "toy_stage"]
    assert [request.item_id for request in seen] == ["a", "b"]


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
    stage_directory.mkdir()
    utils.append_jsonl({"doc_id": "b", "content": "old b"}, stage_directory / "toy_records.jsonl")
    utils.append_jsonl({"doc_id": "d", "content": "old d"}, stage_directory / "toy_records.jsonl")
    requested = []

    def send(request):
        requested.append(request.item_id)
        return ModelReply(f"new {request.item_id}", "end_turn")

    documents = [{"doc_id": "a"}, {"doc_id": "b", "applies": False}, {"doc_id": "c"}, {"doc_id": "d"}]
    result, _ = run_toy(tmp_path, documents, send=send)
    assert requested == ["a", "c"]
    # earlier records in input order (b is inapplicable but kept), then this run's, in input order
    assert [(r["doc_id"], r["content"]) for r in result.records_in_input_order] == [
        ("b", "old b"), ("d", "old d"), ("a", "new a"), ("c", "new c")]


def test_a_step_with_a_group_runs_each_group_in_one_call_and_skips_the_groups_the_checkpoint_marks(tmp_path):
    requests = []

    def group(documents, run):
        return {"group_of_a": documents[:2], "group_of_c": documents[2:]}

    def build_request(documents, run):
        return ModelRequest(user_message="+".join(d["doc_id"] for d in documents), system_prompt="", model=None,
                            stage="toy_stage", item_id=f"group_of_{documents[0]['doc_id']}")

    def read_reply(documents, reply, run):
        return {d["doc_id"]: Accepted(reply.text) for d in documents}

    def records(documents, outcomes, run):
        return OutcomeRecords((("grouped_records.jsonl", {"group": [d["doc_id"] for d in documents]}),))

    step = toy_step(group=group, build_request=build_request, read_reply=read_reply, records=records,
                    checkpoint_file_name="_checkpoint.json", progress_line=None)
    documents = docs("a", "b", "c")
    send = lambda request: (requests.append(request.user_message), ModelReply("ok", "end_turn"))[1]  # noqa: E731
    result, stage_directory = run_toy(tmp_path, documents, step=step, send=send)
    assert requests == ["a+b", "c"] and result.pending_document_ids == ["a", "b", "c"]
    assert [r["group"] for r in utils.load_jsonl(stage_directory / "grouped_records.jsonl")] == [["a", "b"], ["c"]]
    run_toy(tmp_path, documents, step=step, send=send)   # every group is marked under its key
    assert requests == ["a+b", "c"]


def test_a_group_that_fails_is_reported_whole_and_nothing_is_written_for_it(tmp_path):
    step = toy_step(group=lambda documents, run: {"all": documents},
                    build_request=lambda documents, run: ModelRequest("u", "", None, "toy_stage", "all"),
                    read_reply=lambda documents, reply, run: {d["doc_id"]: Failed(FailureKind.UNUSABLE_ANSWER, "no") for d in documents},
                    records=lambda documents, outcomes, run: OutcomeRecords(), progress_line=None)
    result, _ = run_toy(tmp_path, docs("a", "b"), step=step)
    assert result.failed_groups == [docs("a", "b")] and result.failed_document_ids == ["a", "b"]


# ---------------------------------------------------------------- the declared exception policy

BUILD_AND_SEND = StepPolicy(errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND)


def broken_and_fragile(policy=BUILD_AND_SEND, outcomes_by_document_id=None):
    """A toy step where a document with ``"broken_template": True`` raises in ``build_request`` and a reply
    "EXPLODE" raises in ``read_reply``."""
    step = toy_step(policy, outcomes_by_document_id)

    def build_request(documents, run):
        if documents[0].get("broken_template"):
            raise KeyError("missing_placeholder")
        return step.build_request(documents, run)

    def read_reply(documents, reply, run):
        if reply.text == "EXPLODE":
            raise ValueError("parser bug")
        return step.read_reply(documents, reply, run)

    return toy_step(policy, outcomes_by_document_id, build_request=build_request, read_reply=read_reply)


def test_a_prompt_building_error_is_a_recorded_failure_for_that_document_only(tmp_path, capsys):
    sent = []

    def send(request):
        sent.append(request.item_id)
        return ModelReply("fine", "end_turn")

    outcomes = {}
    result, stage_directory = run_toy(
        tmp_path, [{"doc_id": "a"}, {"doc_id": "b", "broken_template": True}, {"doc_id": "c"}],
        step=broken_and_fragile(outcomes_by_document_id=outcomes), send=send)
    assert sent == ["a", "c"]
    assert result.failed_document_ids == ["b"] and result.thrown_call_document_ids == ["b"]
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a", "c"]
    assert outcomes["b"] == Failed(FailureKind.THROWN_CALL, "error: KeyError: 'missing_placeholder'")
    assert "  b: Failed" in capsys.readouterr().out


def test_a_resume_retries_a_document_whose_prompt_could_not_be_built(tmp_path):
    run_toy(tmp_path, [{"doc_id": "a"}, {"doc_id": "b", "broken_template": True}], step=broken_and_fragile())
    sent = []

    def send(request):
        sent.append(request.item_id)
        return ModelReply("fixed", "end_turn")

    assert pending_ids(tmp_path, [{"doc_id": "a"}, {"doc_id": "b"}]) == ["b"]
    _, stage_directory = run_toy(tmp_path, [{"doc_id": "a"}, {"doc_id": "b"}], step=broken_and_fragile(), send=send)
    assert sent == ["b"]
    assert [r["doc_id"] for r in utils.load_jsonl(stage_directory / "toy_records.jsonl")] == ["a", "b"]


def test_a_template_broken_for_every_document_stops_the_run_with_no_model_call(tmp_path):
    sent = []
    with pytest.raises(SystemExit, match="systemic"):
        run_toy(tmp_path, [{"doc_id": "a", "broken_template": True}, {"doc_id": "b", "broken_template": True}],
                step=broken_and_fragile(), send=lambda request: sent.append(request) or None)
    assert sent == []


def test_by_default_an_error_building_a_request_ends_the_run(tmp_path):
    with pytest.raises(KeyError, match="missing_placeholder"):
        run_toy(tmp_path, [{"doc_id": "a", "broken_template": True}], step=broken_and_fragile(StepPolicy()))


def test_an_error_in_read_reply_ends_the_run_unless_the_step_declares_it_a_thrown_call(tmp_path):
    explode = lambda request: ModelReply("EXPLODE", "end_turn")  # noqa: E731
    with pytest.raises(ValueError, match="parser bug"):
        run_toy(tmp_path, [{"doc_id": "a"}], step=broken_and_fragile(), send=explode)
    declared = StepPolicy(errors_counted_as_thrown_calls=(
        WhereAnErrorIsAThrownCall.BUILD | WhereAnErrorIsAThrownCall.SEND | WhereAnErrorIsAThrownCall.INTERPRET))
    outcomes = {}
    with pytest.raises(SystemExit, match="systemic"):
        run_toy(tmp_path / "declared", [{"doc_id": "a"}, {"doc_id": "b"}],
                step=broken_and_fragile(declared, outcomes), send=explode)
    assert set(outcomes.values()) == {Failed(FailureKind.THROWN_CALL, "error: ValueError: parser bug")}
    assert set(outcomes) == {"a", "b"}


def test_a_place_the_step_does_not_declare_lets_the_exception_through(tmp_path):
    only_send = StepPolicy(errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.SEND)
    with pytest.raises(KeyError):
        run_toy(tmp_path, [{"doc_id": "a", "broken_template": True}], step=broken_and_fragile(only_send))
    no_send = StepPolicy(errors_counted_as_thrown_calls=WhereAnErrorIsAThrownCall.BUILD)

    def send(request):
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        run_toy(tmp_path / "other", [{"doc_id": "a"}], step=broken_and_fragile(no_send), send=send)
