# evals/ — notes for agents

What each eval measures and how to run it is in `evals/README.md`. This file holds what an agent
needs before touching `publish_hf.py`. **Only run it when a human explicitly asks for a specific
run to be published, and confirm the run's provenance with them first.**

## What `publish_hf.py` stages and how

The warning in `evals/README.md` ("Before running `publish_hf.py`") and the safety rule in the root
`CLAUDE.md` are the rules; this is the detail, moved here from the root `CLAUDE.md`. The target is the public dataset repository
`sentientfutures/animal-welfare-training-claude`. That one repository holds **both**
corpora as separate Hugging Face *configs* (each gets its own selector in the dataset
viewer), staged under per-pipeline subdirectories, `sdf/` and `dad/`, each with its corpus
jsonl, `run_manifest.json` and `audit/`. Consequences worth knowing before running it:

- **Both configs publish the same column philosophy: the content a reader came for, then
  supporting metadata, then the widest lineage column, then the id, trailing everything,
  last.** SDF is re-keyed by `reorder_sdf_corpus` into `content` · `description` ·
  `language` · `type_name` · `type_id` · `register` · `scores` · `variables` ·
  `variable_value_ids` (only on runs that record it; absent on older runs and the published
  dataset so far) · `doc_id`. It is no longer a verbatim `shutil.copy2`, since the
  pipeline's own write order (the score step, `sdf_pipeline/layer5_score_and_final_selection/score.py`) buried `content`, the actual document text,
  behind `doc_id` and four other metadata fields. DAD is rewritten per record by
  `flatten_dad_corpus` into `user_prompt` · `assistant_response` · `language` · `variables` ·
  `example_gid`. `variables` is the one nested column on each side and the counterpart of the
  other corpus's own: the whole hand the scenario or document was dealt, so a reader can tell
  which slice of the matrix a row came from rather than only which language it is in. It
  sits second-to-last on both, ahead of only the id, because it is the widest cell and the
  text column(s) are what a visitor came to read; the id trails even `variables` because it
  is pure lineage and join bookkeeping (still used internally to look up
  `language`/`variables`; see `flatten_dad_corpus`), not something a reader needs in front of
  the content. A field `reorder_sdf_corpus` does not recognise — a legacy field like
  `subtype_id`/`role` predating the current SDF schema — is appended after the known columns
  rather than dropped: this reorders columns, it never selects them, so an older committed
  run missing `variables`/`description`/`type_name` still publishes every field it has.
- **`language` and `variables` are both joined off `step3/rewrites.jsonl`, in one read**
  (`dad_dealt_cards` → `{example_gid: cards}`, with `dad_languages` derived from it), so they
  can never describe different records. Joined rather than read off the corpus record even
  though the record now carries `variables` itself: every committed run predates that field,
  so the join is what gives the column to the published 1,324 rows without rewriting a line
  of committed output. Both follow the same two rules — omitted entirely when nothing
  resolves (a column null down every row reads as broken), null on a single row that does
  not join (a visible gap is honest, an invented value is not). Their "nothing resolved"
  gates are **separate and per-run**, because they are not the same claim: `archetype10`
  deals no `cultural_setting`, so it earns no `language` column, but it has cards worth
  publishing under `variables`.
- `delete_patterns` is scoped to `<pipeline>/audit/*`. A bare `audit/*` would delete the
  *sibling* dataset's audit files on every publish.
- **Published rows are ordered English first**, because the Hub viewer opens on whatever is
  first in the file and both corpora are written in matrix-deal order (the live SDF config
  opened on Spanish, DAD on Hindi). **Only the staged copy is reordered — never the run's
  own `final/` corpus**, which the stride-sampling evals (`compliance_sdf.py`,
  `audit_sdf.py`, `diversity.py`, `score_sdf.py`) read and which the near-duplicate removal
  step of SDF layer 5 (`layer5_score_and_final_selection/near_duplicate_removal.py`) already ordered greedily. It is a
  stable binary partition, not a sort by language: English block first, everything else
  after, each in the run's own order. The pass sorts raw lines, so the Hub file's ROW order
  is a permutation of whatever it was staged as going into this pass. Both corpora are
  re-keyed before staging now (`reorder_sdf_corpus`, `flatten_dad_corpus`), so the Hub file
  is no longer a raw-text permutation of the run's own `final/*.jsonl` the way
  `diff <(sort hub.jsonl) <(sort final.jsonl)` could once check for SDF — the equivalence
  check is now over parsed, sorted records (same records, same values, nothing added or
  dropped, only column and row order changed), not raw text. A run whose language cannot be
  read (no `language` field; a DAD run with no `step3/rewrites.jsonl` or no
  `cultural_setting` on its cards, e.g. `archetype10`) is published in the order it was
  written, and stdout says so. DAD rows carry a `language` column derived the same way — the
  language the scenario was **dealt**, null where a row does not join. Publishing prints the
  resulting language breakdown, which is now the only thing telling a publisher what the
  hand-maintained card ought to declare — **neither corpus is English-only**, and a card
  that says otherwise is a false claim on a public dataset.
- **Tags are repo-wide**, so prefix them per dataset (`sdf-v1-…`, `dad-v1-…`). The
  pre-multi-config `v1-fullscale-500-opus5` tag predates this convention. `_create_tag`
  passes `exist_ok=True` and **never moves an existing tag**, so a revision pinned by an old
  tag keeps the old row order for ever — publishing a reordered corpus at a pinned revision
  needs a **new** tag.
- `--dry-run` makes zero network calls: it stages, prints what would be uploaded and the
  commit message it would leave, and stops. (It therefore also skips the `git fetch` the
  merge check would otherwise do, and says `origin/main` may be stale.)
- `--staging-dir` is wiped before staging, so the script refuses any existing non-empty
  directory it did not create (marker `.publish_hf_staging`), in `--dry-run` too.
- **An unmerged publish warns and asks, and is recorded in the Hub commit message.** Before
  staging, the script checks whether the current `HEAD` and the run's own `git_commit` are
  reachable from `origin/main` (`merge_state`, in `evals/publish_hf.py`, over
  `shared/utils.py`'s). If either isn't — or can't be verified — it prints what's unmerged
  and requires a typed `yes`; with no TTY it exits telling you to pass `--allow-unmerged`.
  Proceeding appends `_unmerged_summary`'s clause to the commit message, naming each
  unverified run with the branch it was **generated** on and its commit, and separately the
  branch it was **published** from. That was a stamp on the generated card until the card
  became hand-edited; a commit message is the better home anyway, since nobody can edit it
  afterwards. A **dirty tree at run time is context, never a trigger** — every run so far has
  been dirty, and a warning that fires on all of them is one people learn to ignore. This is
  a guardrail against accidents, not an access control: the write token is on contributors'
  laptops, so anyone can bypass the script entirely — which is exactly why the Hub's own
  history, not the terminal, carries the record.

Credentials for the Hub token are described in `shared/README.md`.

```bash
# Stages final/{sdf,dad}_corpus.jsonl (English rows first; DAD also flattened to
# example_gid/language/user_prompt/assistant_response) + run_manifest.json +
# audit/*.{json,jsonl,html}
# into <pipeline>/ (report_content.json excluded — editorial, already baked into
# corpus_report.html) and uploads exactly that. The dataset card is NOT written:
# it is hand-edited on the Hub. Requires a Hub token with write access to the
# target repo/org, one time (`huggingface-cli login`, or HF_TOKEN in .env);
# --dry-run stages + prints what would be uploaded with no network calls.
# An unmerged run prompts for confirmation first (--allow-unmerged skips the
# prompt; the commit message records it either way).
REPO=sentientfutures/animal-welfare-training-claude
python evals/publish_hf.py --input outputs/sdf/latest --repo-id $REPO --dry-run
python evals/publish_hf.py --input outputs/sdf/runs/<run_id> --repo-id $REPO \
    --tag sdf-v1-<run-label>
python evals/publish_hf.py --input outputs/dad/runs/<run_id> --repo-id $REPO \
    --tag dad-v1-<run-label>
```
