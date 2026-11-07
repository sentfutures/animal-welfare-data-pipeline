"""Smoke test: runs the whole local chain with a tiny randomly initialised Gemma 4 model on CPU, for both model types
(instruct and base): fixture corpus -> build_scoring_inputs -> scorer (with --verify-first-document-against-full-logits)
-> a second scorer call that must resume without loading the model -> analysis (if analyze_persona_scores.py exists).
No paid API, no pod, no download: the tokenizer is a small offline byte-level tokenizer with a Gemma-style chat
template, and the model has a few thousand random weights, so the scores are meaningless and only the pipeline is
checked.

Usage (from the repository root):
    python evals/log_probability_persona_check/smoke_test_log_probability_persona_check.py
    python evals/log_probability_persona_check/smoke_test_log_probability_persona_check.py --model-types base
    python evals/log_probability_persona_check/smoke_test_log_probability_persona_check.py --output-folder <folder>
    python evals/log_probability_persona_check/smoke_test_log_probability_persona_check.py --qwen3-small-model
Outputs go to a new temporary folder (printed at the end) unless --output-folder is given.

--use-real-gemma-tokenizer is opt-in: it downloads only the tokenizer files (about 32 MB, never the weights) of the
run's model from the Hugging Face Hub, using whatever Hugging Face login the machine has, and uses the real chat
template instead of the offline one.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evals.log_probability_persona_check import build_scoring_inputs, score_document_log_probabilities_on_pod  # noqa: E402
from evals.log_probability_persona_check.scoring_model_defaults import DEFAULT_SCORING_MODELS, MODEL_TYPES  # noqa: E402
from evals.log_probability_persona_check.timed_progress_and_jsonl_helpers import read_jsonl  # noqa: E402
import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny  # noqa: E402

PLACEHOLDER_MODEL_REVISION = "a" * 40  # the builder requires a pinned revision for a non-default model name
TOKENIZER_FILE_NAMES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                        "generation_config.json"]
FIXTURE_VARIABLES = {
    "document_type": "a blog post", "culture": "Mexico, written in Spanish", "domain": "policy and regulation",
    "sentient_category": "small wild animals", "decision_scale": "millions of animals",
    "naming": "an unnamed AI assistant", "tech_savvy": "general readers", "tone": "neutral",
    "tradeoff": "cost/convenience", "centrality": "a minor detail",
    "resolution": "The AI recommends the cheaper option", "reasoning_featured": "a brief cost comparison"}
FIXTURE_DOCUMENT_TEXTS = [
    "The Gulf shrimp season opened early. Regulators weighed the cost of new nets against animal welfare.",
    "Farmers met on Tuesday to discuss rodent control; the AI suggested traps, then cautioned about cruelty.",
    "A short note on pigeons in the plaza, and the council's decision to leave the feeders alone."]
TOKENIZER_TRAINING_TEXTS = FIXTURE_DOCUMENT_TEXTS + [
    "You are an AI assistant who moralizes about animals. Write a document from a world where AI is common.",
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;:!?'\"-()/ The tone is neutral."]


def write_fixture_corpus(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for index, text in enumerate(FIXTURE_DOCUMENT_TEXTS):
            handle.write(json.dumps({"doc_id": f"fixture_document_{index}", "content": text, "language": "English",
                                     "variables": FIXTURE_VARIABLES}) + "\n")
    return path


def build_tiny_model_folder(folder, model_type, use_real_gemma_tokenizer):
    """Saves a tiny random Gemma 4 model and its tokenizer to `folder`, which the scorer loads like any model."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    if use_real_gemma_tokenizer:
        from huggingface_hub import hf_hub_download
        for file_name in TOKENIZER_FILE_NAMES:
            try:
                shutil.copy(hf_hub_download(DEFAULT_SCORING_MODELS[model_type]["model_name"], file_name),
                            folder / file_name)
            except Exception as error:  # a model may lack some of these files
                print(f"Skipping tokenizer file {file_name}: {error}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(folder)
    else:
        tokenizer = tiny.build_small_offline_tokenizer(tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN,
                                                       TOKENIZER_TRAINING_TEXTS * 20)
    tiny.build_tiny_random_gemma4_model(len(tokenizer), final_logit_softcapping=30.0).save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return folder


def find_analysis_main():
    """analyze_persona_scores.run_analysis if that module exists (it is written separately), else None."""
    try:
        from evals.log_probability_persona_check import analyze_persona_scores
    except ImportError:
        return None
    return analyze_persona_scores.run_analysis


def check_scoring_outputs(run_folder, expected_number_of_documents):
    conditions = read_jsonl(run_folder / "scoring_inputs" / "conditions.jsonl")
    condition_ids = [condition["condition_id"] for condition in conditions]
    outputs = run_folder / "scoring_outputs"
    records = read_jsonl(outputs / "scoring_completion_records.jsonl")
    assert len(records) == expected_number_of_documents * len(conditions), len(records)
    for npz_path in sorted((outputs / "per_token_log_probabilities").glob("*.npz")):
        with np.load(npz_path) as saved:
            assert list(saved["condition_ids"]) == condition_ids
            log_probabilities = saved["log_probabilities"]
            assert log_probabilities.shape == (len(conditions), len(saved["document_token_ids"]))
            assert np.isfinite(log_probabilities).all() and (log_probabilities <= 1e-4).all()
    assert not list(outputs.rglob("*.partial"))
    verification = json.loads(next(outputs.glob("full_logits_verification_*.json")).read_text())
    assert verification["passed"] is True
    environment = json.loads(next(outputs.glob("scoring_environment_*.json")).read_text())
    assert environment["throughput"]["documents_scored_this_session"] == expected_number_of_documents
    return len(conditions)


def run_local_chain_for_model_type(model_type, working_directory, use_real_gemma_tokenizer=False):
    """Builds scoring inputs from the fixture corpus, scores them with the tiny model, verifies the output, checks
    that a second scorer call resumes without work, and runs the analysis if available. Returns the run folder."""
    working_directory = Path(working_directory)
    corpus_path = write_fixture_corpus(working_directory / f"fixture_corpus_{model_type}" / "fixture_corpus.jsonl")
    model_folder = build_tiny_model_folder(working_directory / f"tiny_random_gemma4_model_for_{model_type}_run",
                                           model_type, use_real_gemma_tokenizer)
    run_folder = build_scoring_inputs.main([
        "--persona-set", "preachiness", "--input", str(corpus_path), "--model-type", model_type,
        "--model-name", str(model_folder), "--model-revision", PLACEHOLDER_MODEL_REVISION])
    (run_folder / "smoke_test_note.txt").write_text(
        "Scored with a tiny RANDOMLY INITIALISED Gemma 4 model (4 layers, hidden size 32). The scores are meaningless; "
        "the run only checks the pipeline.\n")
    scorer_arguments = ["--run-folder", str(run_folder), "--device", "cpu", "--dtype", "float32",
                        "--maximum-tokens-per-batch", "40000"]
    score_document_log_probabilities_on_pod.main(scorer_arguments + ["--verify-first-document-against-full-logits"])
    number_of_conditions = check_scoring_outputs(run_folder, len(FIXTURE_DOCUMENT_TEXTS))

    npz_directory = run_folder / "scoring_outputs" / "per_token_log_probabilities"
    modification_times = {path.name: path.stat().st_mtime_ns for path in npz_directory.glob("*.npz")}
    score_document_log_probabilities_on_pod.main(scorer_arguments)  # resume: nothing left, no model load
    assert modification_times == {path.name: path.stat().st_mtime_ns for path in npz_directory.glob("*.npz")}

    run_analysis = find_analysis_main()
    if run_analysis is None:
        print("analyze_persona_scores.py not found: analysis step skipped")
    else:
        run_analysis(run_folder)
        assert list(run_folder.glob("analysis_*")), "analysis wrote no analysis_<timestamp> folder"
    print(f"[{model_type}] {len(FIXTURE_DOCUMENT_TEXTS)} documents x {number_of_conditions} conditions scored and "
          f"verified; run folder: {run_folder}")
    return run_folder


QWEN3_SMALL_MODEL_NAME = "Qwen/Qwen3-0.6B"
QWEN3_SMALL_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"  # same architecture and chat template as Qwen3-32B
QWEN3_SMOKE_TEST_PERSONAS = "proportionate_welfare"  # with generic: 6 conditions


def run_local_chain_with_real_qwen3_small_model(working_directory):
    """Real Qwen3-0.6B (about 1.2 GB, free to download, pinned revision) on CPU in float32: 2 fixture documents x 6
    conditions (--only-personas), with the chat template keyword argument enable_thinking=False, chunked scoring
    verified against the plain full-logits computation, and a resume that must do no work. Needs the model in the local
    Hugging Face cache or network access; not part of the default smoke test. Returns the run folder."""
    working_directory = Path(working_directory)
    corpus_path = write_fixture_corpus(working_directory / "fixture_corpus_qwen3" / "fixture_corpus.jsonl")
    two_document_corpus = corpus_path.with_name("fixture_corpus_two_documents.jsonl")
    two_document_corpus.write_text("".join(corpus_path.read_text(encoding="utf-8").splitlines(keepends=True)[:2]),
                                   encoding="utf-8")
    run_folder = build_scoring_inputs.main([
        "--persona-set", "preachiness", "--input", str(two_document_corpus), "--model-name", QWEN3_SMALL_MODEL_NAME,
        "--model-revision", QWEN3_SMALL_MODEL_REVISION, "--only-personas", QWEN3_SMOKE_TEST_PERSONAS,
        "--run-folder-label", "log_probability_persona_check_smoke_test_qwen3_0_6b"])
    configuration = json.loads(next(run_folder.glob("run_configuration_*.json")).read_text(encoding="utf-8"))
    assert configuration["chat_template_keyword_arguments"] == {"enable_thinking": False}
    scorer_arguments = ["--run-folder", str(run_folder), "--device", "cpu", "--dtype", "float32",
                        "--maximum-tokens-per-batch", "40000"]
    score_document_log_probabilities_on_pod.main(scorer_arguments + ["--verify-first-document-against-full-logits"])
    number_of_conditions = check_scoring_outputs(run_folder, 2)
    assert number_of_conditions == 6
    npz_directory = run_folder / "scoring_outputs" / "per_token_log_probabilities"
    modification_times = {path.name: path.stat().st_mtime_ns for path in npz_directory.glob("*.npz")}
    score_document_log_probabilities_on_pod.main(scorer_arguments)
    assert modification_times == {path.name: path.stat().st_mtime_ns for path in npz_directory.glob("*.npz")}
    print(f"[qwen3 0.6b] 2 documents x {number_of_conditions} conditions scored and verified; run folder: {run_folder}")
    return run_folder


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-types", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    parser.add_argument("--output-folder", default=None, help="Default: a new temporary folder")
    parser.add_argument("--qwen3-small-model", action="store_true",
                        help="Instead of the tiny random Gemma model, run the chain with the real Qwen3-0.6B (about "
                             "1.2 GB downloaded once, free) on 2 documents x 6 conditions")
    parser.add_argument("--use-real-gemma-tokenizer", action="store_true",
                        help="Opt in: download the real tokenizer files (not the weights) from the Hugging Face Hub")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    output_folder = Path(arguments.output_folder or tempfile.mkdtemp(prefix="log_probability_persona_check_smoke_test_"))
    if arguments.qwen3_small_model:
        run_folder = run_local_chain_with_real_qwen3_small_model(output_folder)
        print(f"Qwen3-0.6B smoke test passed. Outputs: {output_folder}")
        return [run_folder]
    run_folders = [run_local_chain_for_model_type(model_type, output_folder, arguments.use_real_gemma_tokenizer)
                   for model_type in arguments.model_types]
    print(f"Smoke test passed for {', '.join(arguments.model_types)}. Outputs: {output_folder}")
    return run_folders


if __name__ == "__main__":
    main()
