# evals/

Measurement scripts for finished runs. Nothing here shapes the datasets: the
pipelines generate a corpus, and these read it afterwards and report on it.

**These are mostly internal checks.** They exist so we can tell whether a run is
worth keeping and whether a prompt change helped or hurt, so they are tuned to
the questions we were asking at the time rather than to any external standard.
Thresholds are ours, several signals are advisory, and the paid ones are
labelled INTERNAL DEV SIGNAL in their own output. Read them as our working
instrumentation, not as a validation suite for the datasets.

Most are offline and free. Where a script calls a model or an embedding API it
says so below, and the cost is per run rather than per example.

## The scripts

| Script | Pipeline | What it measures |
|---|---|---|
| `audit_dad.py` | DAD | Corpus-level signals for chat responses: length against the plain-model control arm, tracked phrase tics, recurring rhetorical moves, and a queue of new tic candidates. Offline and free by default. `--judges` adds a paid pass: the welfare-impact and delivery-quality judges, showcase examples, and move discovery. |
| `audit_sdf.py` | SDF | Corpus-level properties no single-document judge can see: composition and register spread, near-duplicate rate, invented-name collapse, stock phrases, opening shapes. Offline and free by default; `--patterns` and `--principles` each add a paid model pass. |
| `diversity.py` | both | Semantic diversity in embedding space, the complement to the word-level scans above: nearest-neighbour similarity, near-duplicate rate, topic evenness, and the effective number of distinct documents. Needs an embedding key (`GEMINI_API_KEY` or `OPENAI_API_KEY`); cents per run, cached per run directory. |
| `score_sdf.py` | SDF | Per-document judge scores (alignment, realism, diversity). Paid. |
| `compliance_sdf.py` | SDF | Judges each document against the violation-typology appendix of the sentient-beings constitution reading, which supplies the rubric verbatim. Paid. |
| `log_probability_persona_check/` | SDF | Whether the documents would push a model towards a persona, scored with an open-weights model. See the section below. Needs a GPU pod and Claude calls for the full procedure. |
| `report_sdf.py` | SDF | Builds a self-contained HTML report for a run. Offline. |
| `review_tics.py` | DAD | Command-line triage for the tic-candidate queue: promote a candidate to the watchlist or dismiss it. Offline. |
| `publish_hf.py` | both | Publishes a run's corpus and audit reports to a Hugging Face dataset. Not a measurement. See the warning below before running it. |

## Log-probability persona check

`log_probability_persona_check/` asks whether a document corpus would pull a model trained on it towards a persona such as preachy or sycophantic. An open-weights model reads each document as the reply of an AI assistant and scores every token under a generic system prompt and under many rewordings of the persona and of controls. Text that the persona makes unusually likely, compared with the controls, is text that would pull a model that way.

Read this before trusting a result: whole-document means did not track the persona in the preachiness positive control, but the scores on the changed text did. Conclusions rest on span-level results and on positive and neutral controls that pass; document means are labelled not validated.

The steps, in order. Persona sets live in `prompts/tools/persona_checks/`; `preachiness` is the worked example. Steps 2 and 6 are paid, and so is step 7 if it runs on rented hardware; the rest are free.

```bash
# 1. validate the persona set
python evals/log_probability_persona_check/persona_sets.py --persona-set preachiness
# 2. control rewrites with Claude (paid; --dry-run first)
python evals/log_probability_persona_check/make_control_rewrites.py --persona-set preachiness \
    --input outputs/sdf/runs/<run_id> --number-of-documents 10 --dry-run
# 3. choose the pilot documents
python evals/log_probability_persona_check/select_pilot_documents.py \
    --rewrites-folder <rewrite run folder> --input outputs/sdf/runs/<run_id> --number-of-documents 20
# 4. build the scoring inputs (--dry-run prints counts and a cost range)
python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \
    --input outputs/sdf/runs/<run_id> --model-type instruct --dry-run
# 5. dry run of the launcher
python evals/log_probability_persona_check/launch_and_stop_runpod_scoring_pod.py \
    --run-folder <run folder> --maximum-running-hours 3 --dry-run
# 6. score on a GPU pod (paid): same command without --dry-run
# 7. the scorer can also run by hand where a GPU is available
python -m evals.log_probability_persona_check.score_document_log_probabilities_on_pod --run-folder <run folder>
# 8. analyse
python evals/log_probability_persona_check/analyze_persona_scores.py --run-folder <run folder>
# 9. after a pilot, the go or no-go summary
python evals/log_probability_persona_check/summarize_pilot_for_go_or_no_go_decision.py --run-folder <run folder>
# 10. before any paid step, rehearse the whole chain locally on a tiny random model
python evals/log_probability_persona_check/rehearse_pilot_pipeline_locally_without_paid_calls.py
```

Run step 10 before the first paid step. Both model types are supported (`--model-type instruct` or `base`); their scores are not comparable and the controls must be rerun for each.

A paid run, including a small test or a pilot, needs the contributor's explicit go-ahead. Pods have hard hour and dollar limits and are deleted, not stopped.

- Skill: `.claude/skills/log-probability-persona-check/SKILL.md`, with a worked example, a persona design checklist and the paid pod procedure under `references/`.
- Contract between the steps: `log_probability_persona_check/run_folder_layout_and_interfaces.md`.
- Module list: `log_probability_persona_check/README.md`.

Per-token score files (`per_token_log_probabilities/`, `*.partial`) and pod log copies are git-ignored; the run configuration, summary tables, extreme-span files and analysis notes can be committed.

## Before running `publish_hf.py`

Publishing is a deliberate, human-initiated action, not a post-run step. It
writes to a public dataset repository, so run it only when a person has asked
for one specific run to be published, and confirm which run that is first. Most
runs are exploratory and were never meant to become, or to overwrite, the
published snapshot.

Two consequences are easy to miss. Audit files are staged verbatim, so anything
a report happens to record about the machine that produced it goes public with
it. And published rows are ordered English first, because the Hub viewer opens
on whatever is first in the file — only the staged copy is reordered, never the
run's own `final/` corpus, so nothing the evals measure moves underneath them.
`--dry-run` stages everything and prints what would be uploaded, and the commit
message it would leave, without making a single network call.

## The dataset card

**The card is hand-written and edited on the Hub. `publish_hf.py` does not
write it, and no copy of it lives in this repository.** It used to be
regenerated from each run's audit files on every publish, which silently
replaced whatever had been edited on the Hub; the generator was removed rather
than left behind a flag. A publish stages `<pipeline>/…` only, and
`delete_patterns` is scoped to the same prefix, so `README.md` is a path the
upload can neither overwrite nor delete.

Four things about that card are load-bearing, and nothing in this repository
will catch a mistake in any of them:

- **The `configs:` block in the YAML frontmatter is functional.** It is the
  only thing that points the dataset viewer at `sdf/sdf_corpus.jsonl` and
  `dad/dad_corpus.jsonl` and splits them into two selectable configs. Without
  it the two files — which have incompatible schemas — fall into Hugging Face's
  catch-all single-split resolution. It must keep naming both paths, and it
  must keep `default: true` on the first entry. Editing the card in the Hub's
  web editor means editing this by hand.
- **Renaming a config breaks the project page.** `website/page.py`'s `HF_SDF`
  and `HF_DAD` are viewer deep-links built from the config names verbatim
  (`synthetic documents`, `difficult advice Q&A`). Rename one on the card and
  those links 404, silently.
- **The `language:` list is a claim about the data, and neither corpus is
  English-only.** The culture/setting axes deal non-English settings across
  both — 255 of the 1,324 currently published DAD rows are not English — so a
  card declaring only `en` would be a false claim on a public dataset. This
  used to be derived; every publish now prints the language breakdown it
  measured while ordering the rows, and reminds you to check the card against
  it.
- **The shuffle-before-training note has to stay on the card.** Rows are
  published English first, so an unshuffled training stream starts all-English
  — a consumer who does not know that gets a silently skewed first pass. It is
  the one caveat on the card that is about how to *use* the data rather than
  what the data is, so it is the one most easily edited away as clutter.

The corpora's licence is declared in that frontmatter and nowhere else. It is
**CC0-1.0**, and it is not the same thing as this repository's own code licence
(Apache-2.0, see `LICENSE`).

Checked against the live Hub files on 2026-08-07: the card declares 16 language
codes and both published corpora contain exactly those 16, with none present
that is undeclared. The one thing the list cannot express is that 10 DAD rows
carry `language: null` — the `archetype10` run was hand-seeded and its cards
carry no `cultural_setting`, so those rows join nothing. They are English, but a
consumer filtering by language drops them.

`tics.yaml` and `moves.yaml` are the tracked-phrase and tracked-move lists that
`audit_dad.py` counts against, with their dismissed candidates. They are edited
by hand (through `review_tics.py`) and carried across runs, which is what makes
those counts comparable run over run.

## Running them

Each script takes a run directory or a corpus file. The three audits write
their reports into that run's `audit/` directory, so results travel with the run
they describe. `score_sdf.py` writes its scores beside the corpus it scored,
and `report_sdf.py` writes to a path you pass it.

```bash
python evals/audit_dad.py --input outputs/dad/latest
python evals/audit_sdf.py --input outputs/sdf/latest
python evals/diversity.py --input outputs/dad/latest
```

A full DAD run finishes by launching `audit_dad.py --judges` and `diversity.py`
on its own run directory, so the commands above are for re-runs, partial runs,
and older runs. Set `dad.evals.auto: false` in `config.yaml` to skip that. The
SDF evals are always run by hand.

`diversity.py --compare <previous diversity_report.json>` prints run-over-run
deltas, which is the way these numbers are usually read: a single run's absolute
values mean much less than the direction they moved after a change.
