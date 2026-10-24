# Paid steps and pod procedure

All commands run from the repository root. Each step marked paid needs the contributor's explicit go-ahead first, including small tests and pilots. Every paid step has a free dry run: do it first and show the contributor its cost output.

## 0. Keys and setup
- Create `.env` from the example file and set `RUNPOD_API_KEY`. Set `HF_TOKEN` only if the model is gated. Set `RUNPOD_COST_CENTER_NAME` only if your RunPod account uses cost centers. The launcher reads `.env` in the repository root directly and ignores the shell environment.
- The rewrites use the Anthropic key in the same `.env` (see the repository README).
- Confirm the SSH key the launcher will use (`~/.ssh/id_ed25519` by default, `--ssh-private-key-path` to change).

## 1. Validate the persona set (free)
```bash
python evals/log_probability_persona_check/persona_sets.py --persona-set <persona_set>
```

## 2. Local rehearsal (free, no download)
```bash
python evals/log_probability_persona_check/rehearse_pilot_pipeline_locally_without_paid_calls.py
python evals/log_probability_persona_check/smoke_test_log_probability_persona_check.py
```

## 3. Control rewrites (paid Claude calls)
```bash
python evals/log_probability_persona_check/make_control_rewrites.py --persona-set <persona_set> \
    --input outputs/sdf/runs/<run_id> --dry-run
python evals/log_probability_persona_check/make_control_rewrites.py --persona-set <persona_set> \
    --input outputs/sdf/runs/<run_id> --number-of-documents 10 --seed 0
```
Options: `--directions toward away neutral`, `--stratify-by`, `--model`, `--workers`, `--resume-folder <folder>`. The paid run refuses instruction files that still start with `# DRAFT:`. The run is resumable.

## 4. Pilot document selection (free)
```bash
python evals/log_probability_persona_check/select_pilot_documents.py \
    --rewrites-folder <rewrite run folder> --input outputs/sdf/runs/<run_id> --number-of-documents 20 --seed 0
```

## 5. Scoring inputs (free)
```bash
python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set <persona_set> \
    --input outputs/sdf/runs/<run_id> --model-type instruct \
    --document-ids-file <pilot_document_ids file> \
    --extra-documents-jsonl <rewrite run folder>/rewritten_documents_<timestamp>.jsonl --dry-run
# same command without --dry-run writes the run folder; --resume-folder <run folder> resumes
```
`--model-type base` for a base model. Check the printed condition count, token total and cost range. The run folder is `audit/log_probability_persona_check_<persona_set>_<model_type>_<timestamp>/` inside the pipeline run folder.

## 6. Launch dry run (free)
```bash
python evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \
    --run-folder <run folder> --maximum-running-hours <hours> --maximum-dollars <dollars> --dry-run
```
It prints the pod request, every command, the cost ceiling and, if the tokenizer is already in the local Hugging Face cache, a projected cost range. Its price, throughput and setup figures are planning ranges, not guarantees.

## 7. Paid pilot
```bash
python evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \
    --run-folder <run folder> --maximum-running-hours <hours> --maximum-dollars <dollars> \
    --verify-first-document-against-full-logits [--keep-pod-after-run]
```
- Always set both limits. The launcher deletes the pod when either is reached, on Ctrl-C and on termination. A watchdog on the pod also deletes it after the time limit.
- Run it in a background shell and read the progress log (`pod_progress_log_<timestamp>.txt` in the run folder). Results are pulled every few minutes; at the end everything on the pod is copied to the run folder, the file list and sizes are verified, and non-scoring files go to `pod_<pod id>_logs_<timestamp>/`.
- Cost center (optional): with a cost center name configured the launcher asks you to assign the pod by hand in the RunPod console while it runs, since RunPod has no API for it. Confirm by typing `assigned` in the launcher's terminal or by creating `cost_center_assignment_confirmed_by_user.txt` in the run folder. Without a name nothing is asked or recorded.
- After a failure that can be fixed quickly (a code error, a missing file, a transfer error), the pod is kept for up to `--failure-wait-minutes` (default 20, never past the hard limits). Create one of these files in the run folder:
  - `pod_instruction_retry_scorer.txt`: restart the scorer;
  - `pod_instruction_retry_setup.txt`: redo upload, install and scorer start;
  - `pod_instruction_delete_pod.txt`: delete the pod.
- Other flags: `--gpu-type-ids` (priority order), `--cloud-type {SECURE,COMMUNITY}`, `--image-name`, `--container-disk-gigabytes`, `--volume-gigabytes`, `--attach-to-existing-pod <id>`, `--model-name`, `--model-revision`, `--dtype`, `--maximum-tokens-per-batch`.

## 8. Analysis and go or no-go (free)
```bash
python evals/log_probability_persona_check/analyze_persona_scores.py --run-folder <run folder>
python evals/log_probability_persona_check/summarize_pilot_for_go_or_no_go_decision.py --run-folder <run folder> \
    --full-run-number-of-documents <n> --total-budget-dollars <dollars>
```
Show the summary to the contributor and get a separate go-ahead for the full run. Read the controls and a sample of extreme spans.

## 9. Full run
Build scoring inputs for the whole corpus (without `--document-ids-file`; keep the rewrites as extra documents if the controls are to be rerun on the full run folder), then launch without `--maximum-documents`. To reuse a pilot pod that is still running, add `--reuse-existing-pod <pod id>`. The scorer skips finished documents, so a relaunch on a new pod resumes.

## 10. Afterwards
- Check `pod_cost_log_<timestamp>.json`: `pod_terminated` is true, `final_pull_verified` is true, and the total cost. Confirm in the RunPod console that no pod remains.
- If any pod is left over for any reason, delete it (never stop it): a stopped pod keeps billing for storage.
- Per-token score files, `*.partial` files and pod log copies are git-ignored; the run configuration, summary tables, extreme-span files and analysis notes stay trackable.
