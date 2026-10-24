"""Offline smoke run of the considerations selector, end to end on stubs.

One command runs screen, plan, description rewrite, select, draft, rewrite and score for a small run
with a stubbed shared.api.call_claude, then prints the coverage report from the
layer-1 stage and from the final corpus:

    pytest tests/test_considerations_selector_smoke.py -s

No model is called and nothing leaves tmp_path (PIPELINE_OUTPUT_ROOT redirects
the run). The assertions are the contracts the paid checks will rely on: the
picks reach each document's draft prompt and records, targets are never
exceeded, and a resume at layer 2 hands the drafter the same picks.
"""

import json
import re
import zlib

import pytest
import yaml

import sdf_pipeline.run as sdf_run
from sdf_pipeline import layer1_select_considerations as select
from sdf_pipeline.considerations_catalog import load_considerations
from shared import utils

N_DOCUMENTS = 40


@pytest.fixture
def smoke_config_file(tiny_config, tmp_path):
    tiny_config["sdf"]["n_prompts"] = N_DOCUMENTS
    tiny_config["workers"] = 4
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    return path


@pytest.fixture
def outputs_root(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(root))
    return root


def run_main(monkeypatch, config_file, *extra_args):
    monkeypatch.setattr("sys.argv", ["run.py", "--config", str(config_file), "--label", "smoke", *extra_args])
    sdf_run.main()


def offered_labels(user_message):
    shown = {}
    for name in ("moral", "epistemic"):
        body = re.search(rf"<{name}_list>\n(.*?)\n</{name}_list>", user_message, re.DOTALL).group(1)
        shown[name] = [line[2:].split(": ")[0] for line in body.splitlines() if line.startswith("- ")]
    return shown


def selector_answer(user_message, item_id):
    """A deterministic stand-in for the model: scores every shown entry 1-5 by hash."""
    shown = offered_labels(user_message)
    answer = {}
    for name, labels in shown.items():
        if not labels:
            continue
        items = []
        for label in labels:
            score = 1 + zlib.crc32(f"{item_id}:{label}".encode()) % 5
            items.append({"label": label, "score": score,
                          "where_it_arises": f"Where {label} arises in this scenario." if score >= 3 else ""})
        answer[name] = items
    return json.dumps(answer)


def smoke_dispatch(user_message, **kw):
    stage = kw["stage"]
    if stage == "layer1_screen":
        count = user_message.count("<hand number=")
        return json.dumps({"specifications": [
            {"number": n, "reasoning": "Considered it.", "remove": False, "reason": ""}
            for n in range(1, count + 1)]})
    if stage == "layer1_description_rewrite":
        return json.dumps({
            "reasoning": "Checked it.", "rewrite_needed": False, "rewritten_description": "", "changes": [],
            "dealt_value_conflict": False, "conflict_reason": ""})
    if stage == "layer1_select":
        return selector_answer(user_message, kw["item_id"])
    if stage == "layer2_draft":
        return f"<document>Drafted body for {kw['item_id']}.</document>"
    if stage == "layer3_rewrite":
        return f"Review.\n<improved_document>Rewritten document for {kw['item_id']}.</improved_document>"
    if stage == "layer4_score":
        return json.dumps({"alignment": 9, "realism": 9, "spec_conformance": 9, "notes": ""})
    if stage == "layer1_plan":
        return (f"<document_planning>notes</document_planning>\n"
                f"<document_description>Spec for {kw['item_id']}.</document_description>")
    raise AssertionError(f"Unrecognized stage {stage!r}")


def test_offline_smoke_screen_plan_description_rewrite_select_draft_rewrite(smoke_config_file, outputs_root, stub_claude, monkeypatch, capsys):
    calls = stub_claude(smoke_dispatch)
    run_main(monkeypatch, smoke_config_file)
    run_dir = outputs_root / "sdf" / "latest"
    catalog = load_considerations(run_dir / "inputs" / "prompts")

    corpus = utils.load_jsonl(run_dir / "final" / "sdf_corpus.jsonl")
    assert len(corpus) == N_DOCUMENTS
    picked_documents = 0
    for record in corpus:
        labels = record["variables"]["moral_considerations"] + record["variables"]["epistemic_considerations"]
        entries = [catalog.entry_by_label(label) for label in labels]
        # one spec line per pick, in the order moral then epistemic, and nothing when there are no picks
        assert (record["considerations_spec"].count("\n") + 1 == len(labels)) if labels \
            else (record["considerations_spec"] == "")
        for entry in entries:
            assert entry.display.split(": ", 1)[0] in record["considerations_spec"]
        picked_documents += bool(labels)
        draft_call = next(c for c in calls if c["stage"] == "layer2_draft" and c["item_id"] == record["doc_id"])
        assert (record["considerations_spec"] in draft_call["user_message"]) if labels else \
            ("considerations" not in draft_call["user_message"])
    assert 0 < picked_documents < N_DOCUMENTS

    report = json.loads((run_dir / "layer1" / select.COVERAGE_FILENAME).read_text())
    for entry in report["entries"]:
        assert entry["achieved"] <= entry["target_count"]
    selector_calls = [c for c in calls if c["stage"] == "layer1_select"]
    assert len(selector_calls) == N_DOCUMENTS                      # one scoring call per eligible document

    outputs = select.load_considerations_outputs(run_dir)
    corpus_report = select.corpus_coverage_report(outputs, corpus)
    with capsys.disabled():
        print("\n--- considerations coverage at the end of layer 1 ---")
        select.print_coverage_report(report)
        print("--- considerations coverage measured on the final corpus ---")
        select.print_coverage_report({**corpus_report, "entries": [
            {**e, "target_count": round(e["target_share"] * e["eligible_documents"], 1)}
            for e in corpus_report["entries"]]})
    assert corpus_report["eligible_documents"] == N_DOCUMENTS


def test_resuming_at_layer_two_hands_the_drafter_the_same_picks(smoke_config_file, outputs_root, stub_claude, monkeypatch):
    first_calls = stub_claude(smoke_dispatch)
    run_main(monkeypatch, smoke_config_file)
    run_dir = outputs_root / "sdf" / "latest"
    first_prompts = {c["item_id"]: c["user_message"] for c in first_calls if c["stage"] == "layer2_draft"}
    selector_call_count = sum(c["stage"] == "layer1_select" for c in first_calls)
    assert selector_call_count > 0

    # throw the drafts and everything after them away, then resume at layer 2
    for directory in ("layer2", "layer3", "layer4", "final"):
        for path in (run_dir / directory).glob("*"):
            path.unlink()
    resumed_calls = stub_claude(smoke_dispatch)
    run_main(monkeypatch, smoke_config_file, "--resume", "--layer", "2")
    resumed_prompts = {c["item_id"]: c["user_message"] for c in resumed_calls if c["stage"] == "layer2_draft"}
    assert resumed_prompts == first_prompts
    assert not any(c["stage"] in ("layer1_select", "layer1_plan", "layer1_screen", "layer1_description_rewrite") for c in resumed_calls)
