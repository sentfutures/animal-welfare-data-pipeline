"""Seeding a new SDF run from a chosen step of an earlier run (``run.py --seed-from ... --from-step ...``).

Offline, with canned replies. The core claims:

* seeding at the critique step with selected drafts sends exactly the requests that the layer 3 rerun
  setup script (retired; it set up a run folder from selected drafts) followed by ``--resume --layer 3``
  sent, reproduced here from what that script's code did;
* for every seedable step, the seeded run's requests for the selected documents equal the source run's
  own requests for those documents from that step on, so seeding changes no request;
* a request that cannot be met is refused with a message, before a run folder is created and before
  any call;
* ``--layer``, ``--resume`` and ``--run-id`` still behave as before.
"""

import functools
import json
import shutil
from pathlib import Path

import pytest
import yaml

import sdf_pipeline.run as sdf_run
from sdf_pipeline import seeding
from shared import utils
from sdf_step_and_command_line_test_helpers import run_main as run_pipeline_main

REQUEST_FIELDS = ("system_prompt", "user_message", "model", "max_tokens", "stage", "item_id", "cache_system",
                  "temperature", "json_schema", "injection")
DOCUMENT_COUNT = 3
STAGES_FROM_STEP = {
    "plan": ["layer1_plan", "layer1_description_rewrite", "layer1_select", "layer2_draft", "layer3_critique",
             "layer3_revise", "layer4_score"],
    "description_rewrite": ["layer1_description_rewrite", "layer1_select", "layer2_draft", "layer3_critique",
                            "layer3_revise", "layer4_score"],
    "consideration_scoring": ["layer1_select", "layer2_draft", "layer3_critique", "layer3_revise", "layer4_score"],
    "draft": ["layer2_draft", "layer3_critique", "layer3_revise", "layer4_score"],
    "critique": ["layer3_critique", "layer3_revise", "layer4_score"],
    "revise": ["layer3_revise", "layer4_score"],
    "score": ["layer4_score"],
}


# ------------------------------------------------------------------ canned replies

def _screen_answer(user_message):
    count = user_message.count("<hand number=")
    return json.dumps({"specifications": [
        {"number": n, "reasoning": "Considered it.", "remove": False, "reason": ""} for n in range(1, count + 1)]})


def _dispatch(user_message, **kw):
    stage = kw["stage"]
    if stage == "layer1_screen":
        return _screen_answer(user_message)
    if stage == "layer1_description_rewrite":
        return json.dumps({"reasoning": "Checked it.", "rewrite_needed": False, "rewritten_description": "",
                           "changes": [], "dealt_value_conflict": False, "conflict_reason": ""})
    if stage == "layer1_select":
        return json.dumps({
            name: [{"label": label, "score": 1, "where_it_arises": ""} for label in schema["items"]["properties"]["label"]["enum"]]
            for name, schema in kw["json_schema"]["properties"].items()})
    if stage == "layer1_plan":
        return (f"<document_planning>notes</document_planning>\n"
                f"<document_description>Spec for {kw['item_id']}.</document_description>")
    if stage == "layer2_draft":
        return f"<document>Drafted body for {kw['item_id']}.</document>"
    if stage == "layer3_critique":
        return f"<review>Problems with {kw['item_id']}.</review>"
    if stage == "layer3_revise":
        return f"Review notes.\n<improved_document>Rewritten document for {kw['item_id']}.</improved_document>"
    if stage == "layer4_score":
        return json.dumps({"alignment": 9, "realism": 9, "spec_conformance": 9, "notes": ""})
    raise AssertionError(f"Unrecognized stage {stage!r}")


# ------------------------------------------------------------------ helpers

@pytest.fixture
def outputs_root(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(root))
    return root


@pytest.fixture
def config_file(tiny_config, tmp_path):
    tiny_config["sdf"]["n_prompts"] = DOCUMENT_COUNT
    tiny_config["sdf"]["revision"] = {"split": True, "style_mode": "axes", "use_length": True}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    return path


run_main = functools.partial(run_pipeline_main, main=sdf_run.main)


def comparable(calls):
    """The requests as sent, ordered by stage and item so concurrent ordering cannot matter."""
    return sorted(({field: call.get(field) for field in REQUEST_FIELDS} for call in calls),
                  key=lambda request: (request["stage"], request["item_id"] or "", request["user_message"]))


def only_run_directory_not_in(outputs_root: Path, known: set) -> Path:
    new = [d for d in (outputs_root / "sdf" / "runs").iterdir() if d.is_dir() and d.name not in known]
    assert len(new) == 1, new
    return new[0]


def run_names(outputs_root: Path) -> set:
    runs = outputs_root / "sdf" / "runs"
    return {d.name for d in runs.iterdir() if d.is_dir()} if runs.is_dir() else set()


@pytest.fixture
def source_run(config_file, outputs_root, stub_claude, monkeypatch):
    """A finished run of DOCUMENT_COUNT documents, and the requests it sent."""
    calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--label", "source")
    run = only_run_directory_not_in(outputs_root, set())
    return run, list(calls)


def document_ids_of(run: Path) -> list:
    return [r["doc_id"] for r in utils.load_jsonl(run / "layer2" / "drafts.jsonl")]


def tree_fingerprint(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def write_run_the_way_the_layer_3_rerun_setup_script_does(
        config: dict, source_run_dir: Path, runs_root: Path, label: str, document_ids: list, repository_root: Path) -> Path:
    """What the retired layer 3 rerun setup script did, reproduced from its code: a run folder with the checkout's
    current prompts and constitution, the source run's layer 1 folder copied whole, and layer2/drafts.jsonl
    holding only the selected documents' draft records."""
    run_dir = utils.create_run_dir(runs_root, label=label, config=config,
                                   snapshot_dirs={"prompts": repository_root / "prompts" / "sdf",
                                                  "constitution": repository_root / "constitution"})
    shutil.copytree(utils.sdf_stage_dir(source_run_dir, 1), utils.sdf_stage_dir(run_dir, 1), dirs_exist_ok=True)
    drafts = [d for d in utils.load_jsonl(utils.sdf_stage_dir(source_run_dir, 2) / "drafts.jsonl")
              if d.get("doc_id") in set(document_ids)]
    layer2 = utils.sdf_stage_dir(run_dir, 2)
    utils.ensure_dir(layer2)
    with open(layer2 / "drafts.jsonl", "w", encoding="utf-8") as handle:
        for draft in drafts:
            handle.write(json.dumps(draft, ensure_ascii=False) + "\n")
    return run_dir


REPOSITORY_ROOT = Path(__file__).parent.parent


# ------------------------------------------------------------------ equivalence with the retired hand-written setup

def test_seeding_at_the_critique_step_sends_the_requests_of_the_layer_3_rerun_setup_script(
        source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    selected = document_ids_of(source_directory)[:2]
    config = utils.load_config(str(config_file))
    runs_root = outputs_root / "sdf" / "runs"

    hand_calls = stub_claude(_dispatch)
    hand_run = write_run_the_way_the_layer_3_rerun_setup_script_does(
        config, source_directory, runs_root, "hand-rerun", selected, REPOSITORY_ROOT)
    run_main(monkeypatch, config_file, "--resume", "--run-id", hand_run.name, "--layer", "3")

    known = run_names(outputs_root)
    seeded_calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "critique",
             "--document-ids", ",".join(selected), "--use-current-prompts", "--label", "seeded-rerun")
    seeded_run = only_run_directory_not_in(outputs_root, known)

    assert {c["stage"] for c in hand_calls} == {"layer3_critique", "layer3_revise", "layer4_score"}
    assert len(hand_calls) == 3 * len(selected)
    assert comparable(seeded_calls) == comparable(hand_calls)
    for name in ("layer3/critiques.jsonl", "layer3/rewrites.jsonl", "layer4/scores.jsonl", "final/sdf_corpus.jsonl"):
        assert utils.load_jsonl(seeded_run / name) == utils.load_jsonl(hand_run / name), name


def test_seeding_at_the_revise_step_sends_the_revise_requests_of_the_rerun_when_the_critiques_match(
        source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    selected = document_ids_of(source_directory)[:2]
    config = utils.load_config(str(config_file))

    hand_calls = stub_claude(_dispatch)
    hand_run = write_run_the_way_the_layer_3_rerun_setup_script_does(
        config, source_directory, outputs_root / "sdf" / "runs", "hand-rerun", selected, REPOSITORY_ROOT)
    run_main(monkeypatch, config_file, "--resume", "--run-id", hand_run.name, "--layer", "3")

    seeded_calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", str(source_directory), "--from-step", "revise",
             "--document-ids", ",".join(selected), "--use-current-prompts")
    # the source's critiques are the canned critique text, so each revise request is the rerun's
    assert {c["stage"] for c in seeded_calls} == {"layer3_revise", "layer4_score"}
    assert comparable(seeded_calls) == comparable([c for c in hand_calls if c["stage"] != "layer3_critique"])


# ------------------------------------------------------------------ every step sends the source run's requests

@pytest.mark.parametrize("from_step", list(STAGES_FROM_STEP))
def test_seeding_at_each_step_sends_the_requests_the_source_run_sent_for_those_documents(
        from_step, source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, source_calls = source_run
    selected = document_ids_of(source_directory)[:2]
    known = run_names(outputs_root)
    source_before = tree_fingerprint(source_directory)

    seeded_calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", from_step,
             "--document-ids", ",".join(selected), "--label", "seeded")
    seeded_run = only_run_directory_not_in(outputs_root, known)

    expected = [c for c in source_calls if c["stage"] in STAGES_FROM_STEP[from_step] and c["item_id"] in selected]
    assert expected
    assert comparable(seeded_calls) == comparable(expected)
    corpus = utils.load_jsonl(seeded_run / "final" / "sdf_corpus.jsonl")
    assert sorted(r["doc_id"] for r in corpus) == sorted(selected)
    source_corpus = {r["doc_id"]: r for r in utils.load_jsonl(source_directory / "final" / "sdf_corpus.jsonl")}
    assert {r["doc_id"]: r for r in corpus} == {d: source_corpus[d] for d in selected}
    assert tree_fingerprint(source_directory) == source_before  # the source run is never touched


def _dispatch_that_scores_the_first_consideration_of_each_list_highly(user_message, **kw):
    if kw["stage"] == "layer1_select":
        return json.dumps({
            name: [{"label": label, "score": 5 if index == 0 else 1, "where_it_arises": "it arises in the scenario" if index == 0 else ""}
                   for index, label in enumerate(schema["items"]["properties"]["label"]["enum"])]
            for name, schema in kw["json_schema"]["properties"].items()})
    return _dispatch(user_message, **kw)


@pytest.mark.parametrize("from_step", ["consideration_scoring", "draft", "critique"])
def test_seeding_keeps_the_considerations_a_document_was_assigned(
        from_step, config_file, outputs_root, stub_claude, monkeypatch):
    source_calls = stub_claude(_dispatch_that_scores_the_first_consideration_of_each_list_highly)
    run_main(monkeypatch, config_file, "--label", "source")
    source_directory = only_run_directory_not_in(outputs_root, set())
    assigned = [r for r in utils.load_jsonl(source_directory / "layer2" / "drafts.jsonl") if r["considerations_spec"]]
    assert assigned, "the fixture must assign at least one consideration for this test to mean anything"
    selected = [assigned[0]["doc_id"]]
    known = run_names(outputs_root)
    seeded_calls = stub_claude(_dispatch_that_scores_the_first_consideration_of_each_list_highly)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", from_step,
             "--document-ids", selected[0])
    seeded_run = only_run_directory_not_in(outputs_root, known)
    # the assignment of a single scored document is solved over that document alone; it is the same here
    # because each document's scores decide its own picks in this fixture
    stages = STAGES_FROM_STEP[from_step]
    expected = [c for c in source_calls if c["stage"] in stages and c["item_id"] == selected[0]]
    if from_step != "consideration_scoring":
        assert comparable(seeded_calls) == comparable(expected)
    assert utils.load_jsonl(seeded_run / "layer2" / "drafts.jsonl")[0]["considerations_spec"] == assigned[0]["considerations_spec"]


def test_a_seeded_run_is_a_fresh_run_that_records_where_it_came_from(
        source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    selected = document_ids_of(source_directory)[:1]
    known = run_names(outputs_root)
    stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "draft",
             "--document-ids", selected[0], "--label", "seeded")
    seeded_run = only_run_directory_not_in(outputs_root, known)

    manifest = json.loads((seeded_run / "run_manifest.json").read_text())
    assert manifest["label"] == "seeded"
    assert manifest["run_id"] == seeded_run.name != source_directory.name
    seeded_from = manifest["seeded_from"]
    assert seeded_from["source_run_id"] == source_directory.name
    assert seeded_from["from_step"] == "draft"
    assert seeded_from["document_ids"] == selected
    assert seeded_from["start_layer"] == 2
    assert seeded_from["copied_record_counts"]["layer1/plans.jsonl"] == 1
    # the frozen snapshot is the source run's own
    assert tree_fingerprint(seeded_run / "inputs") == tree_fingerprint(source_directory / "inputs")
    assert (outputs_root / "sdf" / "latest").resolve() == seeded_run.resolve()
    # only the selected document's records were copied from the steps before the draft
    assert [r["prompt_id"] for r in utils.load_jsonl(seeded_run / "layer1" / "plans.jsonl")] == selected
    assert [r["doc_id"] for r in utils.load_jsonl(seeded_run / "layer2" / "drafts.jsonl")] == selected


def test_seeding_without_document_ids_takes_every_document_the_step_can_read(
        source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    known = run_names(outputs_root)
    calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "score")
    seeded_run = only_run_directory_not_in(outputs_root, known)
    assert len(calls) == DOCUMENT_COUNT
    assert sorted(r["doc_id"] for r in utils.load_jsonl(seeded_run / "final" / "sdf_corpus.jsonl")) == sorted(document_ids_of(source_directory))


# ------------------------------------------------------------------ the revision override

def test_a_revision_override_is_written_into_the_seeded_drafts_and_reaches_the_revise_request(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    first, second = document_ids_of(source_directory)[:2]
    override_file = tmp_path / "revision_overrides.json"
    override_file.write_text(json.dumps({first: {"cards": {"voice": "hesitant and tentative"}}}))

    known = run_names(outputs_root)
    plain_calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "revise",
             "--document-ids", f"{first},{second}", "--label", "plain")
    overridden_calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "revise",
             "--document-ids", f"{first},{second}", "--label", "overridden", "--revision-override-file", str(override_file))
    overridden_run = max((d for d in (outputs_root / "sdf" / "runs").iterdir() if d.name not in known), key=lambda d: d.stat().st_mtime_ns)

    revise = lambda calls, document: next(c for c in calls if c["stage"] == "layer3_revise" and c["item_id"] == document)  # noqa: E731
    assert "more hesitant and tentative than a typical" in revise(overridden_calls, first)["user_message"]
    assert "hesitant and tentative" not in revise(plain_calls, first)["user_message"]
    assert revise(overridden_calls, second) == revise(plain_calls, second)  # a document without an entry is untouched
    drafts = {r["doc_id"]: r for r in utils.load_jsonl(overridden_run / "layer2" / "drafts.jsonl")}
    assert drafts[first]["revision"] == {"cards": {"voice": "hesitant and tentative"}}
    assert "revision" not in drafts[second]
    rewrites = {r["doc_id"]: r for r in utils.load_jsonl(overridden_run / "layer3" / "rewrites.jsonl")}
    assert rewrites[first]["revision"] == {"cards": {"voice": "hesitant and tentative"}}
    manifest = json.loads((overridden_run / "run_manifest.json").read_text())
    assert manifest["seeded_from"]["revision_override_document_ids"] == [first]


def test_a_revision_override_for_every_document_uses_the_star_entry(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    override_file = tmp_path / "revision_overrides.json"
    override_file.write_text(json.dumps({"*": {"cards": {"voice": "terse"}}}))
    calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "critique",
             "--revision-override-file", str(override_file))
    revises = [c for c in calls if c["stage"] == "layer3_revise"]
    assert len(revises) == DOCUMENT_COUNT
    assert all("more terse than a typical" in c["user_message"] for c in revises)


# ------------------------------------------------------------------ refusals

def refusal(monkeypatch, config_file, outputs_root, stub_claude, *arguments):
    """Run main with arguments, expecting a refusal; returns (message, calls). No run folder may appear."""
    calls = stub_claude(_dispatch)
    before = run_names(outputs_root)
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, config_file, *arguments)
    assert run_names(outputs_root) == before, "a refused seeding must not create a run folder"
    assert calls == [], "a refused seeding must make no calls"
    return str(caught.value), calls


def test_an_unknown_step_is_refused_with_the_seedable_steps(source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude,
                         "--seed-from", source_directory.name, "--from-step", "polish")
    assert "unknown step 'polish'" in message
    for step in STAGES_FROM_STEP:
        assert step in message


@pytest.mark.parametrize("step", ["screen", "deal", "redeal", "consideration_assignment", "gate", "near_duplicate_cull"])
def test_a_step_that_cannot_be_seeded_is_refused_with_its_reason(
        step, source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude,
                         "--seed-from", source_directory.name, "--from-step", step)
    assert f"step '{step}' cannot be seeded" in message


def test_an_unknown_document_id_is_refused_by_name(source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    real = document_ids_of(source_directory)[0]
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", source_directory.name,
                         "--from-step", "critique", "--document-ids", f"{real},matrix_999999,nonsense")
    assert "document ids not found" in message
    assert "matrix_999999" in message and "nonsense" in message and real not in message.split("not found")[1]


def test_an_unknown_source_run_is_refused(config_file, outputs_root, stub_claude, monkeypatch):
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude,
                         "--seed-from", "no-such-run", "--from-step", "critique")
    assert "no-such-run" in message


def damaged_copy_of(source_directory: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "damaged_source_runs" / source_directory.name
    shutil.copytree(source_directory, copy)
    return copy


def test_a_source_without_the_records_the_step_needs_is_refused_naming_each_document(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    damaged = damaged_copy_of(source_directory, tmp_path)
    ids = document_ids_of(source_directory)
    # a critique and a draft are missing for one document each
    critiques = [r for r in utils.load_jsonl(damaged / "layer3" / "critiques.jsonl") if r["doc_id"] != ids[0]]
    utils.save_jsonl(critiques, damaged / "layer3" / "critiques.jsonl")
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", str(damaged),
                         "--from-step", "revise", "--document-ids", ",".join(ids[:2]))
    assert source_directory.name in message and "revise" in message
    assert f"{ids[0]}: " in message and "critique" in message
    assert f"{ids[1]}: " not in message


def test_a_source_that_dropped_the_document_at_the_plan_is_refused(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    damaged = damaged_copy_of(source_directory, tmp_path)
    ids = document_ids_of(source_directory)
    plans = utils.load_jsonl(damaged / "layer1" / "plans.jsonl")
    for plan in plans:
        if plan["prompt_id"] == ids[0]:
            plan["description"] = ""
            plan["incoherent"] = True
    utils.save_jsonl(plans, damaged / "layer1" / "plans.jsonl")
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", str(damaged),
                         "--from-step", "draft", "--document-ids", ids[0])
    assert "declared incoherent" in message


@pytest.mark.parametrize("from_step, missing_template", [
    ("critique", "layer3a_critique.txt"), ("revise", "layer3b_revise.txt"), ("draft", "layer2.txt"),
    ("score", "layer4.txt"), ("plan", "layer1.txt"),
    ("description_rewrite", "layer1_description_rewrite.txt"), ("consideration_scoring", "layer1_select_considerations.txt")])
def test_a_source_snapshot_lacking_a_template_the_step_needs_is_refused_naming_run_and_template(
        from_step, missing_template, source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    damaged = damaged_copy_of(source_directory, tmp_path)
    (damaged / "inputs" / "prompts" / missing_template).unlink()
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", str(damaged),
                         "--from-step", from_step)
    assert source_directory.name in message and missing_template in message and from_step in message


def test_a_source_without_a_frozen_snapshot_is_refused_unless_current_prompts_are_asked_for(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    damaged = damaged_copy_of(source_directory, tmp_path)
    shutil.rmtree(damaged / "inputs")
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", str(damaged), "--from-step", "score")
    assert "no frozen inputs/prompts snapshot" in message and "--use-current-prompts" in message
    calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--seed-from", str(damaged), "--from-step", "score", "--use-current-prompts")
    assert len(calls) == DOCUMENT_COUNT


def test_a_revision_override_is_refused_for_a_step_that_does_not_read_it(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    override_file = tmp_path / "overrides.json"
    override_file.write_text(json.dumps({"*": {"cards": {}}}))
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", source_directory.name,
                         "--from-step", "draft", "--revision-override-file", str(override_file))
    assert "critique or revise" in message


@pytest.mark.parametrize("contents, expected", [
    ("not json", "could not be read as JSON"), ('["a"]', "must be a JSON object"),
    ('{"matrix_424242": {"cards": {}}}', "not selected"), ('{"*": {"cards": "loud"}}', "must be a record like")])
def test_a_malformed_revision_override_file_is_refused(
        contents, expected, source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    override_file = tmp_path / "overrides.json"
    override_file.write_text(contents)
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", source_directory.name,
                         "--from-step", "revise", "--revision-override-file", str(override_file))
    assert expected in message


@pytest.mark.parametrize("arguments", [
    ["--seed-from", "x"],                                                   # no step
    ["--from-step", "revise"],                                              # no source
    ["--seed-from", "x", "--from-step", "revise", "--resume"],
    ["--seed-from", "x", "--from-step", "revise", "--layer", "3"],
    ["--seed-from", "x", "--from-step", "revise", "--run-id", "y"],
    ["--document-ids", "a"], ["--use-current-prompts"]])
def test_seeding_flags_that_do_not_belong_together_are_refused(arguments, config_file, outputs_root, stub_claude, monkeypatch):
    calls = stub_claude(_dispatch)
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, config_file, *arguments)
    assert caught.value.code == 2
    assert calls == [] and run_names(outputs_root) == set()


def test_a_run_in_the_pre_renumber_layout_cannot_be_seeded_from(source_run, config_file, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    damaged = damaged_copy_of(source_directory, tmp_path)
    (damaged / "layer1").rename(damaged / "layer12")
    message, _ = refusal(monkeypatch, config_file, outputs_root, stub_claude, "--seed-from", str(damaged), "--from-step", "score")
    assert "current layout" in message


# ------------------------------------------------------------------ --layer, --resume and --run-id are unchanged

def test_resume_at_a_layer_still_reruns_only_that_layer_and_after_in_the_same_run(
        source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    (source_directory / "layer4" / "scores.jsonl").unlink()
    (source_directory / "layer4" / "_checkpoint.json").unlink(missing_ok=True)
    known = run_names(outputs_root)
    calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--resume", "--run-id", source_directory.name, "--layer", "4")
    assert run_names(outputs_root) == known  # the same run, no new one
    assert [c["stage"] for c in calls] == ["layer4_score"] * DOCUMENT_COUNT
    manifest = json.loads((source_directory / "run_manifest.json").read_text())
    assert "seeded_from" not in manifest


def test_a_resume_with_no_layer_starts_at_layer_one_and_makes_no_call_for_finished_work(
        source_run, config_file, outputs_root, stub_claude, monkeypatch):
    source_directory, _ = source_run
    calls = stub_claude(_dispatch)
    run_main(monkeypatch, config_file, "--resume", "--run-id", source_directory.name)
    assert calls == []


def test_a_layer_outside_one_to_four_is_still_rejected(config_file, outputs_root, stub_claude, monkeypatch):
    stub_claude(_dispatch)
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, config_file, "--layer", "5")
    assert caught.value.code == 2


def test_an_ordinary_run_has_no_seeded_from_entry(source_run):
    source_directory, _ = source_run
    assert "seeded_from" not in json.loads((source_directory / "run_manifest.json").read_text())


def test_the_seedable_steps_and_the_reasons_for_the_others_do_not_overlap():
    assert not set(seeding.SEEDABLE_STEPS) & set(seeding.UNSEEDABLE_STEPS)


# ------------------------------------------------------------------ run resolution, revise under one call, re-solved assignment


def test_a_bare_name_is_a_run_id_and_a_local_folder_is_used_only_when_no_run_has_that_name(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    (runs_root / "run_a").mkdir(parents=True)
    working_directory = tmp_path / "working"
    (working_directory / "run_b").mkdir(parents=True)
    monkeypatch.chdir(working_directory)
    assert seeding.resolve_source_run("run_a", runs_root) == runs_root / "run_a"
    assert seeding.resolve_source_run("run_b", runs_root) == (working_directory / "run_b").resolve()


def test_a_bare_name_that_is_both_a_run_and_a_local_folder_is_refused_naming_both(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    (runs_root / "shared_name").mkdir(parents=True)
    working_directory = tmp_path / "working"
    (working_directory / "shared_name").mkdir(parents=True)
    monkeypatch.chdir(working_directory)
    with pytest.raises(seeding.SeedingRefused) as caught:
        seeding.resolve_source_run("shared_name", runs_root)
    message = str(caught.value)
    assert "ambiguous" in message
    assert str(runs_root / "shared_name") in message and str((working_directory / "shared_name").resolve()) in message


def test_a_value_with_a_separator_is_a_path_not_a_run_id(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    (runs_root / "shared_name").mkdir(parents=True)
    (runs_root / "run_a").mkdir(parents=True)
    working_directory = tmp_path / "working"
    (working_directory / "shared_name").mkdir(parents=True)
    monkeypatch.chdir(working_directory)
    assert seeding.resolve_source_run("./shared_name", runs_root) == (working_directory / "shared_name").resolve()
    with pytest.raises(seeding.SeedingRefused) as caught:
        seeding.resolve_source_run("missing/run_a", runs_root)  # not read as the run id "run_a"
    assert "no run folder at missing/run_a" in str(caught.value)


def test_seeding_at_revise_under_the_single_call_configuration_is_refused_saying_to_seed_at_critique(
        source_run, tiny_config, outputs_root, stub_claude, monkeypatch, tmp_path):
    source_directory, _ = source_run
    tiny_config["sdf"]["revision"] = {"split": False}
    one_call_config = tmp_path / "one_call_config.yaml"
    one_call_config.write_text(yaml.safe_dump(tiny_config))
    calls = stub_claude(_dispatch)
    before = run_names(outputs_root)
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, one_call_config, "--seed-from", source_directory.name, "--from-step", "revise")
    message = str(caught.value)
    assert "revises in one call together with the critique, so there is no separate revise step; seed from 'critique' instead" in message
    assert "sdf.revision.split: false" in message
    assert calls == [] and run_names(outputs_root) == before


@pytest.mark.parametrize("document_selection", ["one", "all"])
def test_seeding_at_consideration_scoring_says_and_records_that_the_assignment_is_solved_again(
        document_selection, config_file, outputs_root, stub_claude, monkeypatch, capsys):
    stub_claude(_dispatch_that_scores_the_first_consideration_of_each_list_highly)
    run_main(monkeypatch, config_file, "--label", "source")
    source_directory = only_run_directory_not_in(outputs_root, set())
    known = run_names(outputs_root)
    stub_claude(_dispatch_that_scores_the_first_consideration_of_each_list_highly)
    extra = ["--document-ids", document_ids_of(source_directory)[0]] if document_selection == "one" else []
    expected_count = 1 if document_selection == "one" else DOCUMENT_COUNT
    capsys.readouterr()
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "consideration_scoring", *extra)
    output = capsys.readouterr().out
    assert f"the consideration assignment will be solved again over the {expected_count} selected document(s)" in output
    assert "may differ from the source run's" in output
    assert "seed at consideration_scoring for a fresh assignment, at draft to keep the source's" in output
    manifest = json.loads((only_run_directory_not_in(outputs_root, known) / "run_manifest.json").read_text())
    seeded_from = manifest["seeded_from"]
    assert seeded_from["consideration_assignment_solved_again_over_document_count"] == expected_count
    assert seeded_from["consideration_assignment"] == (
        f"solved again over {expected_count} document(s); may differ from the source run's")


def test_seeding_at_draft_prints_no_assignment_notice_and_records_no_assignment_field(
        source_run, config_file, outputs_root, stub_claude, monkeypatch, capsys):
    source_directory, _ = source_run
    known = run_names(outputs_root)
    stub_claude(_dispatch)
    capsys.readouterr()
    run_main(monkeypatch, config_file, "--seed-from", source_directory.name, "--from-step", "draft",
             "--document-ids", document_ids_of(source_directory)[0])
    assert "solved again" not in capsys.readouterr().out
    manifest = json.loads((only_run_directory_not_in(outputs_root, known) / "run_manifest.json").read_text())
    assert "consideration_assignment" not in manifest["seeded_from"]
