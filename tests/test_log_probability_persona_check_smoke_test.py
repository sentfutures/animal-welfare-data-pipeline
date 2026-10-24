"""The smoke test and the local rehearsal as tests: the whole chain (fixture corpus, scoring inputs, scorer on CPU with a
tiny random model, verification, resume, analysis if present) for both model types. Downloading the real Gemma
tokenizer is opt-in (`--use-real-gemma-tokenizer` on the script) and never happens here."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from evals.log_probability_persona_check import rehearse_pilot_pipeline_locally_without_paid_calls as rehearsal  # noqa: E402
from evals.log_probability_persona_check import smoke_test_log_probability_persona_check as smoke_test  # noqa: E402


@pytest.mark.parametrize("model_type", ["instruct", "base"])
def test_smoke_test_chain_runs_for_each_model_type(tmp_path, model_type):
    run_folder = smoke_test.run_local_chain_for_model_type(model_type, tmp_path)
    assert run_folder.is_relative_to(tmp_path)
    assert (run_folder / "scoring_outputs" / "scoring_completion_records.jsonl").is_file()


def test_real_tokenizer_download_is_opt_in():
    assert smoke_test.parse_arguments([]).use_real_gemma_tokenizer is False
    assert smoke_test.parse_arguments(["--use-real-gemma-tokenizer"]).use_real_gemma_tokenizer is True


def test_rehearsal_covers_both_model_types_and_the_launcher_dry_run(tmp_path, capsys):
    run_folders = rehearsal.main(["--output-folder", str(tmp_path)])
    assert set(run_folders) == {"instruct", "base"}
    output = capsys.readouterr().out
    assert output.count("DRY RUN: no RunPod calls are made.") == 2
