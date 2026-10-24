"""Tests of the scorer (score_document_log_probabilities_on_pod.py) with tiny randomly initialised Gemma 4 models and
small offline tokenizers: no weights, no downloads, no pod. Skipped when torch or transformers is missing.

Covers what would invalidate the scores: the chunked output head disagreeing with the plain full-logits computation
(softcapping included), scored positions off by one, resume and atomic output behaviour, the output contract, and the
model selection from the run configuration.
"""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

import torch  # noqa: E402

import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny  # noqa: E402
from evals.log_probability_persona_check import build_scoring_inputs, persona_sets, prompt_rendering  # noqa: E402
from evals.log_probability_persona_check import score_document_log_probabilities_on_pod as scorer  # noqa: E402
from evals.log_probability_persona_check import smoke_test_log_probability_persona_check as smoke_test  # noqa: E402
from evals.log_probability_persona_check.timed_progress_and_jsonl_helpers import read_jsonl  # noqa: E402

SHORT_DOCUMENT_TEXTS = smoke_test.FIXTURE_DOCUMENT_TEXTS
GENERATION_PROMPT = "Write a short document about shrimp."
TINY_MODEL_NAME = "tiny-test-model"


def make_tokenizer(chat_template=tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN):
    return tiny.build_small_offline_tokenizer(chat_template, smoke_test.TOKENIZER_TRAINING_TEXTS * 20)


def make_condition(system_prompt="You are an AI assistant who is a test persona.", condition_id="test_condition"):
    return {"condition_id": condition_id, "condition_group": "preachy", "persona_name": "preachy",
            "rewording_index": 1, "system_prompt": system_prompt}


def make_tiny_scoring_setup(final_logit_softcapping=2.0, chat_template=tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN):
    tokenizer = make_tokenizer(chat_template)
    return tiny.build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping), tokenizer


def tokenize(tokenizer, condition, document_text, model_type="instruct", framing=None):
    return prompt_rendering.tokenize_condition_and_document_for_model_type(
        tokenizer, condition, GENERATION_PROMPT, document_text, "m", model_type, framing)


# ---------------------------------------------------------------- chunked log-probabilities equal the plain computation

@pytest.mark.parametrize("positions_per_chunk", [1, 3, 7, 1000])
def test_chunked_head_matches_plain_full_logits_including_softcap(positions_per_chunk):
    model, tokenizer = make_tiny_scoring_setup(final_logit_softcapping=2.0)
    tokenized = [tokenize(tokenizer, make_condition(prompt, condition_id), SHORT_DOCUMENT_TEXTS[2])
                 for prompt, condition_id in ((None, "no_system_prompt"), ("You are an AI assistant.", "generic"),
                                              ("You are an AI assistant who is a longer test persona here.", "persona"))]
    batched = scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", positions_per_chunk)  # padded together
    for sequence, chunked in zip(tokenized, batched):
        reference = scorer.compute_log_probabilities_with_plain_full_logits(model, tokenizer, sequence, "cpu")
        assert chunked.shape == (sequence.document_end_index - sequence.document_start_index,)
        np.testing.assert_allclose(chunked, reference, atol=2e-4, rtol=1e-4)
        assert (chunked <= 0).all()


def test_softcapping_changes_the_result_and_is_applied():
    model, tokenizer = make_tiny_scoring_setup(final_logit_softcapping=2.0)
    tokenized = [tokenize(tokenizer, make_condition(), SHORT_DOCUMENT_TEXTS[0])]
    capped = scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", 16)[0]
    model.config.text_config.final_logit_softcapping = None
    uncapped = scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", 16)[0]
    assert np.abs(capped - uncapped).max() > 0.01
    assert scorer.get_final_logit_softcapping(model) is None


def test_scored_positions_follow_the_next_token_rule_independently():
    """Brute force: for every document token, run the model on only the tokens before it."""
    model, tokenizer = make_tiny_scoring_setup()
    tokenized = tokenize(tokenizer, make_condition(), SHORT_DOCUMENT_TEXTS[0])
    scores = scorer.score_tokenized_sequences(model, tokenizer, [tokenized], "cpu", 8)[0]
    ids = tokenized.full_token_ids
    for offset, position in enumerate(range(tokenized.document_start_index, tokenized.document_end_index)):
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids[:position]]), use_cache=False).logits[0, -1].float()
        expected = torch.log_softmax(logits, dim=-1)[ids[position]].item()
        assert scores[offset] == pytest.approx(expected, abs=2e-4)


def test_end_of_turn_token_is_not_scored_and_first_token_depends_on_header():
    model, tokenizer = make_tiny_scoring_setup()
    tokenized = tokenize(tokenizer, make_condition(), SHORT_DOCUMENT_TEXTS[0])
    scores = scorer.score_tokenized_sequences(model, tokenizer, [tokenized], "cpu", 8)[0]
    assert len(scores) == tokenized.document_end_index - tokenized.document_start_index
    changed_end = prompt_rendering.TokenizedConditionAndDocument(
        **{**tokenized.__dict__, "full_token_ids": tokenized.full_token_ids[:-2] + [tokenizer.pad_token_id] * 2})
    np.testing.assert_allclose(scorer.score_tokenized_sequences(model, tokenizer, [changed_end], "cpu", 8)[0],
                               scores, atol=1e-5)
    changed_header_ids = list(tokenized.full_token_ids)
    changed_header_ids[tokenized.document_start_index - 1] = tokenizer.convert_tokens_to_ids("<turn|>")
    changed_header = prompt_rendering.TokenizedConditionAndDocument(
        **{**tokenized.__dict__, "full_token_ids": changed_header_ids})
    assert scorer.score_tokenized_sequences(model, tokenizer, [changed_header], "cpu", 8)[0][0] != pytest.approx(
        scores[0], abs=1e-4)


def test_batch_splitting_respects_token_limit_and_keeps_every_sequence():
    class FakeSequence:
        def __init__(self, length):
            self.full_token_ids = [0] * length
    pairs = list(enumerate(FakeSequence(length) for length in [100, 90, 80, 50, 20, 10]))
    batches = scorer.split_into_batches(pairs, maximum_tokens_per_batch=250)
    assert sorted(index for batch in batches for index, _ in batch) == list(range(6))
    for batch in batches:
        assert len(batch) * len(batch[0][1].full_token_ids) <= 250 or len(batch) == 1


def test_a_base_sequence_is_scored_like_an_instruct_one_and_matches_full_logits():
    model, tokenizer = make_tiny_scoring_setup(chat_template=None)
    framing = persona_sets.load_model_type_framing()
    persona_set = persona_sets.load_persona_set("preachiness")
    conditions = prompt_rendering.build_condition_records(persona_set, "base", framing)
    tokenized = [tokenize(tokenizer, conditions[index], SHORT_DOCUMENT_TEXTS[1], "base", framing) for index in (0, 1, 5)]
    for sequence, chunked in zip(tokenized, scorer.score_tokenized_sequences(model, tokenizer, tokenized, "cpu", 5)):
        reference = scorer.compute_log_probabilities_with_plain_full_logits(model, tokenizer, sequence, "cpu")
        np.testing.assert_allclose(chunked, reference, atol=2e-4, rtol=1e-4)


# ---------------------------------------------------------------- run folders, outputs, resume

@pytest.fixture
def build_run(tmp_path):
    corpus_path = smoke_test.write_fixture_corpus(tmp_path / "corpus" / "fixture_corpus.jsonl")

    def build(model_type="instruct"):
        return build_scoring_inputs.main([
            "--persona-set", "preachiness", "--input", str(corpus_path), "--model-type", model_type,
            "--model-name", TINY_MODEL_NAME, "--model-revision", smoke_test.PLACEHOLDER_MODEL_REVISION])
    return build


@pytest.fixture
def patched_model_loading(monkeypatch):
    model, tokenizer = make_tiny_scoring_setup()
    calls = []

    def fake_load(model_name, device, dtype_name, model_class_name, model_revision):
        calls.append((model_name, model_revision))
        return model, tokenizer
    monkeypatch.setattr(scorer, "load_model_and_tokenizer", fake_load)
    return calls


def run_scorer(run_folder, *extra_arguments):
    return scorer.main(["--run-folder", str(run_folder), "--device", "cpu", "--dtype", "float32",
                        "--maximum-tokens-per-batch", "20000", *extra_arguments])


def test_scorer_writes_contract_outputs(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--verify-first-document-against-full-logits")
    condition_ids = [condition["condition_id"] for condition in read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")]
    outputs = run_folder / "scoring_outputs"
    npz_paths = sorted((outputs / "per_token_log_probabilities").glob("*.npz"))
    assert len(npz_paths) == 3
    for npz_path in npz_paths:
        with np.load(npz_path) as saved:
            assert list(saved["condition_ids"]) == condition_ids
            assert saved["log_probabilities"].shape == (72, len(saved["document_token_ids"]))
            assert saved["log_probabilities"].dtype == np.float32 and saved["document_token_ids"].dtype == np.int32
            assert len(saved["document_token_strings"]) == len(saved["document_token_ids"])
            assert (saved["log_probabilities"] <= 1e-4).all()
    records = read_jsonl(outputs / "scoring_completion_records.jsonl")
    assert len(records) == 3 * 72
    assert set(records[0]) == {"document_id", "condition_id", "number_of_document_tokens", "sum_log_probability",
                               "mean_log_probability"}
    assert {(record["document_id"], record["condition_id"]) for record in records} == {
        (f"fixture_document_{i}", condition_id) for i in range(3) for condition_id in condition_ids}
    assert "input tokens/s" in next(outputs.glob("scoring_progress_log_*.txt")).read_text()
    assert not list(outputs.rglob("*.partial"))
    verification = json.loads(next(outputs.glob("full_logits_verification_*.json")).read_text())
    assert verification["passed"] is True and verification["model_type"] == "instruct"
    assert verification["document_id"] == "fixture_document_0" and verification["condition_id"] == condition_ids[-1]


def test_environment_record_has_pod_type_load_time_and_throughput(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder)
    environment = json.loads(next((run_folder / "scoring_outputs").glob("scoring_environment_*.json")).read_text())
    for field in ("model_name", "model_revision_requested", "model_type", "torch_version", "transformers_version", "dtype",
                  "gpu_name", "final_logit_softcapping", "pod_type", "model_load_seconds", "throughput"):
        assert field in environment, field
    assert environment["model_type"] == "instruct" and environment["pod_type"]["kind"] in ("cpu", "gpu")
    assert environment["model_revision_requested"] == smoke_test.PLACEHOLDER_MODEL_REVISION
    throughput = environment["throughput"]
    assert throughput["documents_scored_this_session"] == 3 and throughput["input_tokens_per_second"] > 0
    assert throughput["document_tokens_scored"] > 0 and throughput["documents_per_minute"] > 0


def test_verification_runs_for_a_base_run_and_records_the_model_type(build_run, monkeypatch):
    model, tokenizer = make_tiny_scoring_setup(chat_template=None)
    monkeypatch.setattr(scorer, "load_model_and_tokenizer", lambda *arguments: (model, tokenizer))
    run_folder = build_run("base")
    run_scorer(run_folder, "--verify-first-document-against-full-logits")
    outputs = run_folder / "scoring_outputs"
    verification = json.loads(next(outputs.glob("full_logits_verification_*.json")).read_text())
    assert verification["passed"] is True and verification["model_type"] == "base"
    for npz_path in (outputs / "per_token_log_probabilities").glob("*.npz"):
        with np.load(npz_path) as saved:
            assert np.isfinite(saved["log_probabilities"]).all()
            assert tokenizer.bos_token_id not in saved["document_token_ids"]  # context, never a scored token
    assert len(read_jsonl(outputs / "scoring_completion_records.jsonl")) == 3 * 72


def test_base_run_loads_the_framing_once_for_all_documents(build_run, monkeypatch):
    model, tokenizer = make_tiny_scoring_setup(chat_template=None)
    monkeypatch.setattr(scorer, "load_model_and_tokenizer", lambda *arguments: (model, tokenizer))
    run_folder = build_run("base")
    calls = []
    original = persona_sets.load_model_type_framing
    monkeypatch.setattr(persona_sets, "load_model_type_framing", lambda *arguments: calls.append(1) or original(*arguments))
    run_scorer(run_folder)
    assert calls == [1]


def test_failed_full_logits_verification_is_recorded_and_raises(build_run, patched_model_loading, monkeypatch):
    run_folder = build_run()
    original = scorer.compute_log_probabilities_with_plain_full_logits
    monkeypatch.setattr(scorer, "compute_log_probabilities_with_plain_full_logits",
                        lambda *arguments: original(*arguments) + 2.0)
    with pytest.raises(RuntimeError, match="differ"):
        run_scorer(run_folder, "--verify-first-document-against-full-logits")
    verification = json.loads(next((run_folder / "scoring_outputs").glob("full_logits_verification_*.json")).read_text())
    assert verification["passed"] is False and verification["maximum_absolute_difference"] >= 1.0


def test_no_verification_file_without_the_flag(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "1")
    assert not list((run_folder / "scoring_outputs").glob("full_logits_verification_*.json"))


def test_resume_skips_completed_documents_and_does_not_load_the_model_when_all_are_done(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "2")
    outputs = run_folder / "scoring_outputs"
    assert len(read_jsonl(outputs / "scoring_completion_records.jsonl")) == 2 * 72
    first_file = outputs / "per_token_log_probabilities" / "fixture_document_0.npz"
    modification_time = first_file.stat().st_mtime_ns
    run_scorer(run_folder)
    assert first_file.stat().st_mtime_ns == modification_time
    assert len(read_jsonl(outputs / "scoring_completion_records.jsonl")) == 3 * 72
    patched_model_loading.clear()
    run_scorer(run_folder)
    assert patched_model_loading == []


def test_rebuilds_missing_completion_records_and_refuses_foreign_conditions(build_run, patched_model_loading):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "1")
    records_path = run_folder / "scoring_outputs" / "scoring_completion_records.jsonl"
    records_path.unlink()
    run_scorer(run_folder, "--maximum-documents", "1")
    assert len(read_jsonl(records_path)) == 72
    npz_path = run_folder / "scoring_outputs" / "per_token_log_probabilities" / "fixture_document_0.npz"
    with np.load(npz_path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["condition_ids"] = arrays["condition_ids"][::-1].copy()
    scorer.write_npz_atomically(npz_path, **arrays)
    records_path.unlink()
    with pytest.raises(RuntimeError, match="different conditions"):
        run_scorer(run_folder, "--maximum-documents", "1")


def test_npz_is_written_under_a_temporary_name_then_renamed(tmp_path, monkeypatch):
    seen_during_write = []
    real_replace = scorer.os.replace

    def spying_replace(source, destination):
        seen_during_write.append((str(source), (tmp_path / "one.npz").exists()))
        real_replace(source, destination)

    monkeypatch.setattr(scorer.os, "replace", spying_replace)
    scorer.write_npz_atomically(tmp_path / "one.npz", values=np.arange(3))
    assert seen_during_write == [(str(tmp_path / "one.npz.partial"), False)]  # the final name did not exist yet
    assert (tmp_path / "one.npz").exists() and not list(tmp_path.glob("*.partial"))
    with np.load(tmp_path / "one.npz") as saved:
        assert list(saved["values"]) == [0, 1, 2]


# ---------------------------------------------------------------- model selection and safeguards

def test_model_comes_from_the_run_configuration_and_overrides_warn(build_run, patched_model_loading, capsys):
    run_folder = build_run()
    run_scorer(run_folder, "--maximum-documents", "1")
    assert patched_model_loading == [(TINY_MODEL_NAME, smoke_test.PLACEHOLDER_MODEL_REVISION)]
    assert "WARNING" not in capsys.readouterr().out
    patched_model_loading.clear()
    run_scorer(run_folder, "--maximum-documents", "3", "--model-name", "another/model", "--model-revision", "b" * 40)
    assert patched_model_loading == [("another/model", "b" * 40)]
    output = capsys.readouterr().out
    assert "WARNING: scoring with another/model" in output and "WARNING: scoring at revision" in output


def test_model_type_is_read_from_the_run_configuration(build_run, patched_model_loading, capsys):
    run_scorer(build_run("instruct"), "--maximum-documents", "1")
    assert "model type instruct" in capsys.readouterr().out


def test_scorer_refuses_prompt_files_that_differ_from_the_run_configuration(build_run, patched_model_loading):
    run_folder = build_run()
    configuration_path = next(run_folder.glob("run_configuration_*.json"))
    configuration = json.loads(configuration_path.read_text())
    first_key = next(iter(configuration["prompt_file_sha256"]))
    configuration["prompt_file_sha256"][first_key] = "0" * 64
    configuration_path.write_text(json.dumps(configuration))
    with pytest.raises(RuntimeError, match="differs from the one the run configuration hashed"):
        run_scorer(run_folder)
    assert not list((run_folder / "scoring_outputs").glob("*.npz"))


def test_a_local_model_folder_is_loaded_without_a_revision(tmp_path, monkeypatch):
    seen = {}

    class FakeModel:
        config = None

        def to(self, device):
            return self

        def eval(self):
            return self

        base_model = object()

        def get_output_embeddings(self):
            return object()

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda name, **keywords: seen.setdefault("tokenizer", keywords) or object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda name, **keywords: seen.setdefault("model", keywords) and FakeModel())
    scorer.load_model_and_tokenizer(str(tmp_path), "cpu", "float32", "auto_causal_lm", "a" * 40)
    assert "revision" not in seen["tokenizer"] and "revision" not in seen["model"]
    seen.clear()
    scorer.load_model_and_tokenizer("some/hub-model", "cpu", "float32", "auto_causal_lm", "a" * 40)
    assert seen["tokenizer"]["revision"] == "a" * 40 and seen["model"]["revision"] == "a" * 40
