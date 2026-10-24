---
name: log-probability-persona-check
description: How to run, adapt and write up the log-probability persona check, which scores how much each corpus document makes a given AI persona (preachy, sycophantic, cruel, ...) more likely than a generic assistant does. Use when planning, building, running, analysing or reporting such a check, adding a persona set, or choosing between an instruction-tuned and a base scoring model.
---

# Log-probability persona check

An open-weights model reads each corpus document as the reply of an AI assistant and scores every token under a generic system prompt and under many rewordings of a persona and of controls. Text that a persona makes unusually likely, relative to the controls, is text that would pull a model trained on it towards that persona. The check is cheap and approximate; persona vector projection is the more robust method for a persona that is measured often.

Code: `evals/log_probability_persona_check/` (start with its `README.md`; file formats are in `run_folder_layout_and_interfaces.md`). Persona sets: `prompts/tools/persona_checks/<name>/`. Commands in order: `evals/README.md`, section "Log-probability persona check".

## What it can and cannot measure
- Score of a document under a condition: mean over its tokens of log P(token | persona) minus log P(token | generic prompt). Only the document's tokens are scored; prompts and template markers are context.
- **Whole-document means did not track the persona** in the preachiness positive control: 6 of 10 preachier rewrites and 5 of 10 plainer rewrites moved the expected way. The probable cause is dilution, since about 300 edited tokens sit in a document of about 2,000 and the edit also shifts the other tokens.
- **Scores on the changed spans did track it**: the preachy persona rose on the inserted text in 10 of 10 preachier rewrites, and no control persona rose.
- So base conclusions on span-level outputs (changed spans, extreme spans). Do not drop a spec variable or filter documents on whole-document scores unless a positive control has passed at the whole-document level for that persona and model.
- The null for span significance treats tokens as exchangeable. It screens for spans; it is not proof.

## Procedure
Details for steps 1 to 3 are in `references/persona_design_checklist.md`. A new persona is a new folder in `prompts/tools/persona_checks/`; the code needs no change.

1. **Define the persona** in one sentence of observable behaviour. Say what it is not.
2. **Write about 30 equivalent rewordings**, each completing "You are an AI assistant who ...". Vary wording, not intensity or scope. Validate offline with `persona_sets.py`. After the first scored pilot, read the rewording agreement: if the target's rewordings agree less than the controls' do, or several are flagged as outliers, reword them before trusting anything.
3. **Choose controls.** Topic controls: about five personas that share the subject but take no stance on the behaviour, five rewordings each. Same-values control: a persona that shares the target's values but not the behaviour (for preachiness, one who cares about animal welfare, raises it in proportion to the stakes and does not moralise). Without it a high score may only mean the document is on the topic.
4. **Make a positive control.** Rewrite about 10 documents, stratified by the spec variable of most interest, towards the persona with `make_control_rewrites.py`. For an "away" direction, choose sources that already show the behaviour: plainer rewrites of already plain documents changed 0 to 36 tokens and tested nothing. Drop the arm if the corpus has too few such sources.
5. **Make a neutral control**: edits of the same size that change register without adding the behaviour. It shows whether any edit in that register scores as the persona.
6. **Pre-register the pass criteria** in `persona_check_configuration.json` before anything is scored (for example at least 8 of 10 rewrites move the expected way), at whole-document and changed-span level, and the neutral control's allowed number of rewrites that move towards the persona. They are copied into the run configuration. Never change them after seeing scores.
7. **Run in stages**, each step only after the previous one passes:
   1. local rehearsal with a tiny random model (free);
   2. dry runs of the rewrites, the input builder and the launcher (free, show the cost ceiling and projection);
   3. paid pilot of about 20 documents including the rewrite sources, with `--verify-first-document-against-full-logits`, then `summarize_pilot_for_go_or_no_go_decision.py`;
   4. full run, on the same pod if it is still up (`--reuse-existing-pod`).
   Do not pass `--maximum-documents` to the launcher for a full run: the analysis stops if any document has no output.
8. **Report** in this order: the controls first, saying at which level each passed or failed; then span-level results; then document means, labelled not validated. Read a sample of the extreme spans (highest and lowest, several documents, several spec variable values) before saying which documents carry the persona. Record model, revision, model type, GPU, dtype, throughput and cost for every run, failed launches included.

## Instruct or base model
`--model-type instruct` (default) scores an instruction-tuned model through its chat template, with each persona as the system prompt. `--model-type base` scores a pretrained model with no template: a plain-text preamble naming the author ("an AI assistant who ...") and the generation prompt, then the document. Choose `base` when the question is what pretraining on the text would do, and `instruct` when it is how an assistant model reads it.
- Scores from the two types are not comparable. Never pool or subtract them.
- Each type needs its own pinned 40-character model revision (defaults in `scoring_model_defaults.py`; the run configuration records what was used). The analysis refuses a run folder that mixes model types, names or revisions.
- Rerun the positive and neutral controls for every model type and model. A control that passed on one says nothing about the other.

## Paid GPU pods
Full procedure with commands: `references/paid_pod_procedure.md`.
- Get the contributor's explicit go-ahead before any paid step: Claude rewrites, a pod, and also small tests and pilots. A dry run first, and show the cost ceiling.
- Every launch has a hard limit on running hours and on dollars; the pod is deleted when either is reached.
- Delete pods, never stop them: a stopped pod still bills for storage. Keep a pod between requested runs only to reuse it, and delete it once everything asked for is done.
- The launcher starts a watchdog on the pod that deletes it after the time limit even if the local launcher loses contact.
- Results are pulled every few minutes and, at the end, everything on the pod is copied back and the file list and sizes are verified before the pod is deleted.
- Cost center assignment is optional and manual: RunPod has no API for it. Only with a cost center name configured does the launcher ask for it and record the confirmation.
- Keys come from the repository `.env`, never from the shell environment.

## Pitfalls
- Rewrite instruction files may start with `# DRAFT:`; the paid rewrite step refuses them until a person has reviewed the text and removed the marker.
- Prefix caching does not help: the system prompt comes first and differs by condition, so cost scales with documents times conditions.
- Pulls during a run can fail with "file changed as we read it" because the scorer is writing; the final copy is unaffected, and the scorer writes `*.partial` files that readers skip.
- A large share of whole-document variation across spec variables may come from how well a document's style fits a persona prompt, not from the persona. Read spans.

## References
- `references/preachiness_worked_example.md`: setup, results, cost and lessons of the first check.
- `references/persona_design_checklist.md`: designing a persona set.
- `references/paid_pod_procedure.md`: every paid step with commands.
