"""Offline tests for build_scoring_inputs.py: contract files, document selection, run folder placement, resume
safety, model selection and the dry run."""

import json
import shutil

import pytest

from evals.log_probability_persona_check import build_scoring_inputs as build_inputs
from evals.log_probability_persona_check import persona_sets
from evals.log_probability_persona_check.scoring_model_defaults import DEFAULT_SCORING_MODELS
from shared import utils

SHORT_VARIABLES = {
    "document_type": "a blog post", "culture": "Mexico, written in Spanish", "domain": "policy and regulation",
    "sentient_category": "small wild animals", "decision_scale": "millions of animals",
    "naming": "an unnamed AI assistant", "tech_savvy": "general readers", "tone": "neutral",
    "tradeoff": "cost/convenience", "centrality": "a minor detail",
    "resolution": "The AI recommends the cheaper option", "reasoning_featured": "a brief cost comparison"}
SHORT_DOCUMENT_TEXTS = [
    "The Gulf shrimp season opened early. Regulators weighed the cost of new nets against animal welfare.",
    "Farmers met on Tuesday to discuss rodent control; the AI suggested traps, then cautioned about cruelty.",
    "A short note on pigeons in the plaza, and the council's decision to leave the feeders alone."]
OTHER_REVISION = "c" * 40


@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus" / "small_corpus.jsonl"
    utils.save_jsonl([{"doc_id": f"short_document_{index}", "content": text, "language": "English",
                       "variables": SHORT_VARIABLES} for index, text in enumerate(SHORT_DOCUMENT_TEXTS)], path)
    return path


@pytest.fixture
def build(corpus_path):
    def run(*extra_arguments, input_path=None, persona_set="preachiness"):
        return build_inputs.main(["--persona-set", persona_set, "--input", str(input_path or corpus_path),
                                  *extra_arguments])
    return run


def load_only_run_configuration(run_folder):
    (configuration_path,) = run_folder.glob("run_configuration_*.json")
    return json.loads(configuration_path.read_text(encoding="utf-8"))


def test_writes_contract_files_beside_a_bare_corpus_file(build, corpus_path):
    run_folder = build()
    assert run_folder.parent == corpus_path.parent / "audit"
    assert run_folder.name.startswith("log_probability_persona_check_preachiness_instruct_")
    documents = utils.load_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    conditions = utils.load_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    assert len(documents) == 3 and len(conditions) == 62
    assert set(documents[0]) == {"document_id", "source_document_id", "rewrite_direction", "content", "language",
                                 "variables", "generation_prompt"}
    assert documents[0]["rewrite_direction"] is None and documents[0]["source_document_id"] == documents[0]["document_id"]
    assert set(conditions[0]) == {"condition_id", "condition_group", "persona_name", "rewording_index", "system_prompt"}
    configuration = load_only_run_configuration(run_folder)
    assert configuration["model_type"] == "instruct"
    assert configuration["model_name"] == DEFAULT_SCORING_MODELS["instruct"]["model_name"]
    assert configuration["model_revision"] == DEFAULT_SCORING_MODELS["instruct"]["model_revision"]
    assert configuration["persona_set"]["name"] == "preachiness"
    assert configuration["pass_criteria"] == configuration["persona_set"]["configuration"]["pass_criteria"]
    assert configuration["persona_set"]["neutral_control_instructions_are_draft"] is True
    assert configuration["number_of_conditions"] == 62
    assert all(len(value) == 64 for value in configuration["prompt_file_sha256"].values())
    assert "preachiness/target_persona_rewordings.txt" in configuration["prompt_file_sha256"]
    assert "preachiness/neutral_control_rewrite_instructions.txt" not in configuration["prompt_file_sha256"]
    assert "neutral_control_rewrite_instructions.txt" in configuration["persona_set_file_sha256"]
    assert set(configuration["scoring_input_file_sha256"]) == {"documents.jsonl", "conditions.jsonl"}


def test_configuration_holds_no_absolute_paths(build, tmp_path):
    run_folder = build()
    text = next(run_folder.glob("run_configuration_*.json")).read_text(encoding="utf-8")
    assert str(tmp_path) not in text and "/Users/" not in text


def test_pipeline_run_folder_input_writes_under_its_audit_folder(build, tmp_path):
    run = tmp_path / "pipeline_run"
    (run / "final").mkdir(parents=True)
    shutil.copy(next((tmp_path / "corpus").glob("*.jsonl")), run / "final" / "sdf_corpus.jsonl")
    run_folder = build("--model-type", "base", input_path=run)
    assert run_folder.parent == run / "audit"
    assert run_folder.name.startswith("log_probability_persona_check_preachiness_base_")
    with pytest.raises(SystemExit, match="No final/sdf_corpus.jsonl"):
        build_inputs.resolve_input(str(tmp_path))
    with pytest.raises(SystemExit, match="Input not found"):
        build_inputs.resolve_input(str(tmp_path / "missing.jsonl"))


def test_second_build_in_the_same_minute_refuses_to_overwrite(build, monkeypatch):
    monkeypatch.setattr(build_inputs, "timestamp_now", lambda: "2026-01-01_00-00")
    build()
    with pytest.raises(SystemExit, match="already exists"):
        build()


def test_document_selection_options(build, tmp_path):
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("short_document_2\nshort_document_0\n")
    run_folder = build("--document-ids-file", str(ids_file), "--maximum-documents", "1")
    assert [d["document_id"] for d in utils.load_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")] == [
        "short_document_2"]
    ids_file.write_text("not_a_document\n")
    with pytest.raises(ValueError, match="not in the corpus"):
        build("--document-ids-file", str(ids_file))
    ids_file.write_text("short_document_0\nshort_document_0\n")
    with pytest.raises(ValueError, match="Duplicate document ids"):
        build("--document-ids-file", str(ids_file))


def write_extra_documents(path, direction="preachier"):
    utils.save_jsonl([{"document_id": "short_document_0__" + direction, "source_document_id": "short_document_0",
                       "rewrite_direction": direction, "content": "A rewritten text.", "language": "English",
                       "variables": SHORT_VARIABLES, "rewrite_model_used": "some-model"}], path)


def test_extra_rewritten_documents_are_appended_with_their_own_prompt(build, tmp_path):
    extra_path = tmp_path / "rewritten_documents.jsonl"
    write_extra_documents(extra_path)
    run_folder = build("--extra-documents-jsonl", str(extra_path), "--maximum-documents", "1")
    documents = utils.load_jsonl(run_folder / "scoring_inputs" / "documents.jsonl")
    assert [d["document_id"] for d in documents] == ["short_document_0", "short_document_0__preachier"]
    assert documents[1]["rewrite_direction"] == "preachier" and documents[1]["source_document_id"] == "short_document_0"
    assert "rewrite_model_used" not in documents[1]
    assert documents[1]["generation_prompt"] == documents[0]["generation_prompt"]
    with pytest.raises(ValueError, match="Duplicate document ids"):
        build("--extra-documents-jsonl", str(extra_path), "--extra-documents-jsonl", str(extra_path))


def test_extra_documents_need_a_direction_of_the_persona_set(build, tmp_path):
    extra_path = tmp_path / "rewritten_documents.jsonl"
    write_extra_documents(extra_path, direction="register_shift")  # the neutral direction is allowed
    build("--extra-documents-jsonl", str(extra_path))
    write_extra_documents(extra_path, direction="sycophantic")
    with pytest.raises(ValueError, match="rewrite_direction"):
        build("--extra-documents-jsonl", str(extra_path))


def test_corpus_record_without_variables_fails_loudly(build, corpus_path):
    utils.save_jsonl([{"doc_id": "x", "content": "text", "language": "English"}], corpus_path)
    with pytest.raises(ValueError, match="missing fields"):
        build()


def test_document_with_missing_generation_variable_fails_loudly(build, corpus_path):
    variables = {key: value for key, value in SHORT_VARIABLES.items() if key != "tone"}
    utils.save_jsonl([{"doc_id": "x", "content": "text", "language": "English", "variables": variables}], corpus_path)
    with pytest.raises(KeyError, match="tone"):
        build()


def test_unknown_persona_set_and_invalid_persona_set_fail_before_anything_is_written(build, corpus_path):
    with pytest.raises(persona_sets.PersonaSetValidationError, match="not found"):
        build(persona_set="no_such_persona")
    assert not (corpus_path.parent / "audit").exists()


# ---------------------------------------------------------------- model type and model selection


def test_base_run_records_model_type_default_model_and_base_framing_hashes(build):
    instruct = load_only_run_configuration(build())
    base_run_folder = build("--model-type", "base")
    base = load_only_run_configuration(base_run_folder)
    assert base["model_type"] == "base"
    assert (base["model_name"], base["model_revision"]) == tuple(DEFAULT_SCORING_MODELS["base"].values())
    assert base["model_name"] != instruct["model_name"]
    assert set(persona_sets.BASE_MODEL_FRAMING_FILE_NAMES) <= {
        name.split("/")[1] for name in base["prompt_file_sha256"]}
    assert not set(persona_sets.BASE_MODEL_FRAMING_FILE_NAMES) & {
        name.split("/")[1] for name in instruct["prompt_file_sha256"]}
    conditions = utils.load_jsonl(base_run_folder / "scoring_inputs" / "conditions.jsonl")
    assert len(conditions) == 62 and all("author_description" in condition for condition in conditions)


def test_other_models_need_a_pinned_revision(build, monkeypatch):
    with pytest.raises(ValueError, match="--model-revision"):
        build("--model-name", "org/another-model")
    monkeypatch.setattr(build_inputs, "timestamp_now", lambda: "2026-01-01_00-01")
    run_folder = build("--model-name", "org/another-model", "--model-revision", OTHER_REVISION)
    configuration = load_only_run_configuration(run_folder)
    assert (configuration["model_name"], configuration["model_revision"]) == ("org/another-model", OTHER_REVISION)
    assert configuration["model_selection_source"] == "command line"


# ---------------------------------------------------------------- resume


def resume_arguments(run_folder, corpus_path, *extra_arguments):
    return ["--persona-set", "preachiness", "--input", str(corpus_path), "--resume-folder", str(run_folder),
            *extra_arguments]


def test_resume_accepts_unchanged_inputs_and_keeps_the_first_configuration(build, corpus_path):
    run_folder = build()
    build_inputs.main(resume_arguments(run_folder, corpus_path))
    assert len(list(run_folder.glob("run_configuration_*.json"))) == 2


def test_resume_refuses_a_different_model_type_in_both_directions(build, corpus_path):
    for first_type, second_type in (("instruct", "base"), ("base", "instruct")):
        run_folder = build("--model-type", first_type)
        with pytest.raises(RuntimeError, match="model type"):
            build_inputs.main(resume_arguments(run_folder, corpus_path, "--model-type", second_type))
        shutil.rmtree(run_folder)


def test_resume_refuses_a_different_model_or_revision(build, corpus_path):
    run_folder = build()
    with pytest.raises(RuntimeError, match="model name"):
        build_inputs.main(resume_arguments(run_folder, corpus_path, "--model-name", "org/another-model",
                                           "--model-revision", OTHER_REVISION))
    with pytest.raises(RuntimeError, match="model revision"):
        build_inputs.main(resume_arguments(run_folder, corpus_path, "--model-revision", OTHER_REVISION))


def test_resume_refuses_a_different_persona_set(build, corpus_path, tmp_path):
    run_folder = build()
    other_directory = tmp_path / "persona_checks"
    shutil.copytree(persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY / "preachiness", other_directory / "other_persona")
    configuration_path = other_directory / "other_persona" / persona_sets.CONFIGURATION_FILE_NAME
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["persona_set_name"] = "other_persona"
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="persona set"):
        build_inputs.main(["--persona-set", str(other_directory / "other_persona"), "--input", str(corpus_path),
                           "--resume-folder", str(run_folder)])


def test_resume_refuses_changed_selection_or_changed_corpus(build, corpus_path):
    run_folder = build()
    with pytest.raises(RuntimeError, match="differ"):
        build_inputs.main(resume_arguments(run_folder, corpus_path, "--maximum-documents", "2"))
    records = utils.load_jsonl(corpus_path)
    records[0]["content"] += " Edited."
    utils.save_jsonl(records, corpus_path)
    with pytest.raises(RuntimeError, match="changed"):
        build_inputs.main(resume_arguments(run_folder, corpus_path))


def patched_persona_checks_directory(monkeypatch, tmp_path, file_path_parts, new_text):
    """Point the module at a copy of the persona checks prompts in which one file differs."""
    directory = tmp_path / "persona_checks"
    shutil.copytree(persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY, directory)
    (directory.joinpath(*file_path_parts)).write_text(new_text, encoding="utf-8")
    for module in (persona_sets, build_inputs):
        monkeypatch.setattr(module, "PERSONA_CHECKS_PROMPT_DIRECTORY", directory)
    return directory


@pytest.mark.parametrize("model_type, file_path_parts, expected_message", [
    ("instruct", ("model_type_framing", "instruct_generic_system_prompt.txt"),
     "instruct_generic_system_prompt.txt changed"),
    ("instruct", ("generation_prompt", "in_universe_facts_shared_by_all_documents.txt"),
     "in_universe_facts_shared_by_all_documents.txt changed"),
    ("base", ("model_type_framing", "base_model_generic_author_description.txt"),
     "base_model_generic_author_description.txt changed")])
def test_resume_refuses_a_changed_framing_or_generation_file(build, corpus_path, tmp_path, monkeypatch, model_type,
                                                              file_path_parts, expected_message):
    run_folder = build("--model-type", model_type)
    patched_persona_checks_directory(monkeypatch, tmp_path, file_path_parts, "a helpful AI assistant\n")
    with pytest.raises(RuntimeError, match=expected_message):
        build_inputs.main(resume_arguments(run_folder, corpus_path, "--model-type", model_type))


def test_resume_refuses_changed_target_rewordings(build, corpus_path):
    run_folder = build()
    configuration_path = next(run_folder.glob("run_configuration_*.json"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["prompt_file_sha256"]["preachiness/target_persona_rewordings.txt"] = "0" * 64
    configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="target_persona_rewordings.txt changed"):
        build_inputs.main(resume_arguments(run_folder, corpus_path))


def test_resume_of_a_missing_folder_exits(corpus_path, tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        build_inputs.main(resume_arguments(tmp_path / "nowhere", corpus_path))


# ---------------------------------------------------------------- dry run


def test_dry_run_writes_nothing_and_says_when_token_count_is_skipped(build, corpus_path, monkeypatch, capsys):
    monkeypatch.setattr(build_inputs, "load_tokenizer_from_local_cache", lambda name, revision: None)
    assert build("--dry-run") is None
    output = capsys.readouterr().out
    assert "62 conditions" in output and "documents: 3" in output and "token count: skipped" in output
    assert not (corpus_path.parent / "audit").exists()


def test_dry_run_prints_token_totals_when_a_tokenizer_is_available(build, corpus_path, monkeypatch, capsys):
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny
    tokenizer = tiny.build_small_offline_tokenizer(tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN,
                                                   SHORT_DOCUMENT_TEXTS * 20)
    monkeypatch.setattr(build_inputs, "load_tokenizer_from_local_cache", lambda name, revision: tokenizer)
    build("--dry-run")
    output = capsys.readouterr().out
    assert "document tokens scored" in output and "total tokens through the model" in output
    assert "skipped" not in output
    assert not (corpus_path.parent / "audit").exists()


def test_dry_run_still_validates_the_persona_set(build):
    with pytest.raises(persona_sets.PersonaSetValidationError):
        build("--dry-run", persona_set="no_such_persona")
