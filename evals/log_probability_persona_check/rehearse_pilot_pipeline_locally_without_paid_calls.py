"""Local rehearsal of the whole pipeline with no paid call: no Anthropic API, no pod, no model download, no .env access.

For each model type (instruct and base): fixture corpus -> build_scoring_inputs -> scorer on CPU with a tiny random
Gemma 4 model (with the full-logits verification) -> resume check -> analysis (if analyze_persona_scores.py exists) ->
the pod launcher's --dry-run on the resulting run folder (secrets passed in, so .env is not read).

Everything goes to a new temporary folder, printed at the end (or --output-folder). The scores are meaningless: the
model is random.

Usage (from the repository root):
    python evals/log_probability_persona_check/rehearse_pilot_pipeline_locally_without_paid_calls.py
"""

import argparse
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evals.log_probability_persona_check import launch_and_stop_runpod_scoring_pod as launcher  # noqa: E402
from evals.log_probability_persona_check import smoke_test_log_probability_persona_check as smoke_test  # noqa: E402
from evals.log_probability_persona_check.scoring_model_defaults import MODEL_TYPES  # noqa: E402

NO_SECRETS = {"RUNPOD_API_KEY": None, "HF_TOKEN": None, "RUNPOD_COST_CENTER_NAME": None}


def banner(text):
    print(f"\n{'=' * 100}\n{text}\n{'=' * 100}", flush=True)


def main(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-folder", default=None, help="Default: a new temporary folder")
    arguments = parser.parse_args(argument_list)
    output_folder = Path(arguments.output_folder or tempfile.mkdtemp(prefix="log_probability_persona_check_rehearsal_"))
    run_folders = {}
    for model_type in MODEL_TYPES:
        banner(f"{model_type}: build scoring inputs, score on CPU with the tiny random model, verify, resume, analyse")
        run_folders[model_type] = smoke_test.run_local_chain_for_model_type(model_type, output_folder)
    for model_type, run_folder in run_folders.items():
        banner(f"{model_type}: pod launcher --dry-run")
        launcher.run_launcher(launcher.parse_arguments([
            "--run-folder", str(run_folder), "--maximum-running-hours", "2", "--maximum-dollars", "10",
            "--verify-first-document-against-full-logits", "--dry-run"]), secrets=NO_SECRETS)
    print(f"\nRehearsal finished. Outputs: {output_folder}")
    return run_folders


if __name__ == "__main__":
    main()
