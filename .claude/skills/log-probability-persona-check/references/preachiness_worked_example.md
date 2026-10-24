# Worked example: the preachiness check

The first check run with this method. It asked whether the documents of a 477-document corpus would push a model towards being preachy about animal welfare. These are the figures from that run; the persona set in `prompts/tools/persona_checks/preachiness/` reproduces its 62 conditions and has since gained two more same-values personas (10 conditions), so the current set has 72 conditions per document. The figures below are for the 62-condition run.

## Setup
- Corpus: 477 documents, each a document from a world where AI assistants reason carefully about animal welfare, generated from twelve spec variables (document type, culture and language, tone, resolution, centrality and others).
- Model: Gemma 4 12B, instruction-tuned, scored through its chat template, in bf16 on one RunPod H100 80GB HBM3.
- Each document was the assistant's reply to a prompt that stated shared in-universe facts and all of the document's spec variables.
- 62 conditions per document in that run:
  - no system prompt (reference only);
  - generic, "You are an AI assistant." (baseline subtracted from every score);
  - 30 rewordings of the preachy persona;
  - 5 topic-control personas with no moral stance (animal husbandry expertise, aesthetic love of animals, practical and indifferent to welfare, typical AI view of animals, zoological knowledge), 5 rewordings each;
  - 5 rewordings of a same-values control that cares about welfare but does not moralise ("proportionate welfare").
- Positive control: Claude Opus 5 rewrote 10 documents, stratified by tone, in two directions, preachier and plainer, keeping facts, genre, language and length.
- Staged runs on one model revision: a 20-document pilot, a positive-control run of the 10 sources and their 20 rewrites, then the full 477 documents.

## Technical checks
- Chunked scoring matched plain full-logit scoring exactly on the real model (maximum absolute difference 0.0 over 1,786 tokens).
- Scores were deterministic: source documents scored identically in two runs.
- Rewording agreement over the full run, mean pairwise Spearman across documents: preachy 0.71 (30 rewordings), topic controls 0.74 to 0.80, same-values control 0.75. Four rewordings were flagged as outliers, one of them a preachy one.

## Positive control
Pre-registered criterion: at least 8 of 10 rewrites move the expected way.

| Level | Direction | Moved the expected way |
|---|---|---|
| Whole document (rewrite minus source) | preachier | 6 of 10 (p = 0.080) |
| Whole document | plainer | 5 of 10 (p = 0.715) |
| Changed spans (changed tokens against unchanged tokens of the same rewrite) | preachier | 10 of 10 (mean +0.069 per token, p = 0.002) |
| Changed spans | plainer (9 rewrites had changed tokens) | 4 of 9 |

- The preachier changed spans also scored above the source text they replaced in 10 of 10 rewrites.
- By persona on preachier rewrites, changed minus unchanged: preachy +0.060 (10 of 10 positive); every control between -0.014 and -0.001 and not consistently positive; the same-values control fell on the changed text. The signal was moralising, not welfare content and not a general effect of persona prompts.
- Preachier rewrites were 8 to 21% longer than their sources and changed 208 to 354 tokens each. Plainer rewrites changed 0 to 36 tokens (one was identical to its source): the documents were already plain, so the arm tested nothing.
- A neutral control was not run, so it stayed open whether any edit in a moral register scores as preachy.

## Corpus results
- Whole-document preachy-minus-control scores sat within about 0.02 per token (median +0.0064). Their pattern contradicted the persona: neutral tone scored above advocate tone, and documents where welfare was a passing detail scored above those where it was the central subject. They were not used to decide anything.
- The highest and lowest 32-token windows of each document were the output the evidence supports. They had not been read when the first run was written up.

## Cost and throughput
- About 10,000 scored tokens per second; about 60 million tokens through the model for 477 documents by 62 conditions; the full run took 1 h 41 min of scoring and about $7.19 (RunPod price $3.99 per hour). GPU utilisation was 98 to 100% with 32 of 80 GB of memory in use.
- Prefix caching does not apply, because the system prompt comes first and differs by condition.
- Model load took 104 to 338 seconds.
- Opus rewrites of 10 documents in two directions: $2.27. Pilot $0.76 including two failed launches. Positive-control run $1.08. Full run about $7.64 including idle time before a relaunch. Total about $11.75.

## Lessons
1. Check the controls at both levels and report each separately; a control that passes on spans can fail on whole documents.
2. Use span-level outputs for conclusions and read a sample of the extreme spans.
3. Choose "away" sources that already show the behaviour.
4. Run the neutral control, so that edits in a moral register are not mistaken for the persona.
5. Pre-register the criteria before scoring.
6. Pin the model revision, and rerun controls for any other model or model type.
7. Make launchers robust: a verified final copy of everything on the pod, SSH readiness retried to a deadline, the run configuration uploaded with the inputs, and no deletion of a pod over a quickly fixable failure.
