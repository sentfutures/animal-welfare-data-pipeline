#!/usr/bin/env python3
"""Publish a run's final corpus + audit reports as a Hugging Face dataset.

One repo holds BOTH pipelines' corpora as separate HF "configs" (each gets its
own selector in the dataset viewer), so a run is staged under its own
per-pipeline directory rather than at the repo root:

    README.md            <- the dataset card: hand-written, edited on the Hub
    sdf/  sdf_corpus.jsonl, run_manifest.json, audit/*
    dad/  dad_corpus.jsonl, run_manifest.json, audit/*

Each dataset dir holds the final corpus jsonl, run_manifest.json for
provenance, and (if present) every audit/*.{json,jsonl,html} file — globbed
rather than named so a future run's eval additions/omissions are picked up or
skipped automatically. Republishing a run clears only ITS OWN
`<pipeline>/audit/*` on the Hub (delete_patterns), so a file only the previous
run of that pipeline produced can't linger — while the sibling pipeline's data
is never touched.

THIS SCRIPT NEVER WRITES THE DATASET CARD. README.md is authored and edited by
hand on the Hub, and it is deliberately absent from the staging directory:
upload_folder overwrites every path it finds, so leaving the card unstaged is
what protects it, and delete_patterns is scoped per pipeline so it never
matches the card either. This script used to regenerate the card from the audit
files on every publish; that silently replaced every hand-edit, so the
generator was removed rather than left behind a flag.

Three things follow from the card living only on the Hub:

  * Its YAML frontmatter carries the `configs:` block, which is the ONLY thing
    pointing the viewer at sdf/sdf_corpus.jsonl and dad/dad_corpus.jsonl and
    naming the two configs. It is functional, not decorative, and nothing here
    regenerates it — see "The dataset card" in evals/README.md for what it must
    contain and what breaks if a config is renamed.
  * Its `language:` list is hand-maintained too. NEITHER corpus is
    English-only: the culture/setting axes deal non-English settings across
    both, which is why this module computes a language breakdown at staging
    time (see order_english_first) and prints it. That print is now the only
    thing telling a publisher what the card ought to declare.
  * The corpora's licence is declared there and nowhere else. (The pipeline's
    own code licence is this repo's Apache-2.0 LICENSE. They are separate, and
    README.md says so.)

Everything a card would have quoted is uploaded as data regardless:
run_manifest.json (or manifests/<run_id>.json for a combined corpus) and every
audit/* file the run produced. report_content.json is the one exception — the
editorial input among the audit files (curated excerpts/translations and
report-section prose, read by evals/report_sdf.py). It is excluded from the
upload because it is already fully baked into corpus_report.html, so nothing in
it would be invisible to a Hub visitor.

Tags are repo-wide, so with more than one dataset in the repo they should be
prefixed per pipeline (`sdf-v1-…`, `dad-v1-…`) to stay unambiguous.

Usage:
  REPO=sentientfutures/animal-welfare-training-claude
  python evals/publish_hf.py --input outputs/sdf/latest --repo-id $REPO --dry-run
  python evals/publish_hf.py --input outputs/sdf/runs/<run_id> --repo-id $REPO \
      --tag sdf-v1-fullscale-500-opus5
  python evals/publish_hf.py --input outputs/dad/runs/<run_id> --repo-id $REPO \
      --tag dad-v1-archetypes-40

Requires a Hugging Face token with write access to the target repo/org:
either ``HF_TOKEN`` in .env (checked first) or a one-time ``huggingface-cli
login`` (its cached token is the fallback) — --dry-run needs neither and
makes no network calls.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import utils
# The one regex that reads a writing language out of a dealt culture/setting
# card ("China, written in Mandarin Chinese, ..." -> "Mandarin Chinese").
# Imported rather than copied: it is a contract with the wording in
# prompts/{sdf,dad}/variables.txt, and a second copy drifts the first time
# someone rewords a value. evals/ already reaches into pipeline packages for
# pure helpers this way (audit_dad.py, diversity.py both import from
# dad_pipeline.id_registry). Cheap: compose_prompts pulls only shared.matrix
# and shared.entity_pools, and entity_pools imports faker lazily inside a
# function, so nothing heavy loads.
from sdf_pipeline.compose_prompts import derive_language
# The offline composer, not step1_dilemmas: see dad_dealt_cards on why this
# script must not reach anything that pulls shared.api -> anthropic.
from dad_pipeline.compose_scenarios import dealt_variables

load_dotenv()

CORPUS_FILENAMES = ("sdf_corpus.jsonl", "dad_corpus.jsonl")

# A sidecar this script no longer writes. It held the title/subtitle a
# generated card used as its section heading, and only existed because the card
# was rebuilt whole on every publish. `sdf/card_meta.json` is still on the Hub
# from the last such publish; the delete pattern in main() clears it.
LEGACY_CARD_META_FILENAME = "card_meta.json"

# Ownership marker for a staging directory this script created. stage_run
# WIPES its staging root, and a mistyped --staging-dir (a sibling run under
# outputs/*/runs/, say) is an uncommitted, unrecoverable, $50-500 loss — so a
# non-empty directory is only deletable when this marker proves we made it.
STAGING_MARKER = ".publish_hf_staging"
# What a staging root created by an OLDER version of this script can
# legitimately hold: the two per-pipeline dirs, plus README.md from back when a
# card was staged. Accepting that shape keeps existing local staging dirs
# reusable without a manual delete. README.md stays in the set for exactly that
# reason — a staging dir this version creates never contains one.
STAGING_LEGACY_ENTRIES = {"README.md", "sdf", "dad"}

# Both spellings the corpora actually use: the full name derive_language
# produces (written into every SDF record by sdf_pipeline/layer5_score.py) and
# the bare ISO code four early SDF runs carry instead. evals/audit_sdf.py
# already accepts both when it filters to English-only documents.
ENGLISH_SPELLINGS = frozenset({"en", "english"})


def resolve_corpus_file(input_arg: str) -> tuple[Path, str]:
    """Return (run_dir, corpus_filename) for an SDF or DAD run directory."""
    run_dir = Path(input_arg)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a run directory: {run_dir}")
    for name in CORPUS_FILENAMES:
        if (run_dir / "final" / name).exists():
            return run_dir, name
    raise SystemExit(
        f"No final/sdf_corpus.jsonl or final/dad_corpus.jsonl under {run_dir}"
    )


def is_english(language: str | None) -> bool:
    """True only for a language value we can READ as English.

    Unlike evals/audit_sdf.py's English filter this does NOT default a missing
    value to English. There a wider net only over-counts a measurement; here
    the front of the published file is a promise, so a row whose language we
    cannot read belongs BEHIND the English block rather than in front of it. A
    corpus where no row at all is readable is handled once and separately, by
    order_english_first declining to reorder.
    """
    return str(language or "").strip().lower() in ENGLISH_SPELLINGS


def dad_dealt_cards(run_dir: Path) -> dict[str, dict]:
    """{example_gid: the cards its scenario was dealt} for one DAD run, or {}
    when the run cannot be read.

    Both published DAD columns that the final corpus predates — `language` and
    `variables` — are derived from this one map, so they describe the same
    records or neither of them does. Each step3/rewrites.jsonl record holds the
    example_gid that reaches the published row AND the scenario_cards it was
    dealt.

    ONE hop, not two: the same cards also sit on step1/dilemmas.jsonl, but
    step3 carries them next to the id the published row keeps, so the join
    needs one file and one key. Checked against the step1 route on all five
    committed runs — the same answer for all 1,324 records, no misses either
    way.
    """
    path = run_dir / "step3" / "rewrites.jsonl"
    if not path.exists():
        return {}
    cards_by_gid: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = record.get("example_gid")
            if not gid:
                continue
            # `annotation` is the pre-rename spelling of scenario_cards.
            # dad_pipeline/step1_dilemmas.py's dealt_cards() is the canonical
            # accessor, deliberately not imported: it pulls shared.api ->
            # anthropic into a script that makes no model calls. (The
            # projection dealt_variables applies below has no such dependency,
            # which is why it lives in the offline composer.)
            cards = record.get("scenario_cards") or record.get("annotation") or {}
            cards_by_gid[gid] = cards if isinstance(cards, dict) else {}
    return cards_by_gid


def dad_languages(cards_by_gid: dict[str, dict]) -> dict[str, str]:
    """{example_gid: language} for one run's dealt cards, or {} when the run
    carries no language evidence at all.

    Final DAD records carry no language field, but the run does: a scenario's
    `cultural_setting` card names the writing language ("China, written in
    Mandarin Chinese, with Chinese idioms and references"). derive_language
    reads that clause; the unmarked ~65% slice stores null, has no clause, and
    falls through to English — which is what those prompts are.

    This is the language the scenario was DEALT, not one detected from the
    text. Validated by script on the pinned run: none of the rows it calls
    English carry CJK/Devanagari/Arabic/Hangul, so the error direction is a
    non-English row landing behind the block, never an unreadable row landing
    in front of it.

    Returns {} when no record carries a cultural_setting at all (a pre-matrix
    run such as archetype10). The caller then leaves the corpus order alone
    rather than declaring every row English on absent evidence — which is why
    this gate stays PER RUN, and why it cannot be folded into
    dad_dealt_cards: such a run still has cards worth publishing under
    `variables`, it just has no language among them.
    """
    languages: dict[str, str] = {}
    any_dealt = False
    for gid, cards in cards_by_gid.items():
        setting = cards.get("cultural_setting")
        if setting:
            any_dealt = True
        languages[gid] = derive_language(setting or "")
    return languages if any_dealt else {}


def order_english_first(corpus_path: Path, language_of) -> dict[str, int] | None:
    """Rewrite a staged corpus jsonl with its English rows first, returning the
    {language: count} breakdown measured on the way — or None when it declined
    to reorder, leaving the file exactly as staged.

    Sorts the RAW LINES. Each line is parsed only to read its language, and the
    original line — whatever byte form it already has by the time this
    function runs — is written back unchanged, so this pass itself never
    reorders a key, reformats a float, or re-escapes non-ASCII — json.dumps
    defaults to ensure_ascii=True, which would turn most of a non-English
    corpus into \\uXXXX escapes (flatten_dad_corpus and reorder_sdf_corpus both
    already pass ensure_ascii=False for exactly that reason, upstream of this
    function). The published rows are therefore always a permutation of
    whatever this function was handed. Both corpora are re-keyed before
    staging now, so that is no longer byte-identical to the run's own
    final/*.jsonl — but it is still value-identical: the same records, the
    same values, only column order and (here) row order changed. Checkable by
    comparing parsed-and-sorted records rather than raw text.

    A STABLE binary partition, not a sort by language: English rows first in
    their original order, then every other row in its original order. Sorting
    by language name would make every prefix of the file monolingual rather
    than just the first screen — worse for anyone streaming the corpus without
    shuffling, and no better in the viewer.

    Declines (returns None, file untouched) on a blank line, a line that is not
    a JSON object, or a corpus where no row's language could be read at all.
    Row order is cosmetic, so an old run we cannot measure is published in the
    order it was written rather than aborting a publish over it.

    Line terminators are normalised: every written line ends in \\n. That is a
    correctness requirement, not tidying — a source whose last line lacked a
    newline would otherwise glue two records onto one line once that line moved
    out of last place.
    """
    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    tagged: list[tuple[str, str | None]] = []
    counts: dict[str, int] = {}
    for line in lines:
        if not line.strip():
            return None
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict):
            return None
        language = language_of(record)
        tagged.append((line, language))
        if language:
            counts[language] = counts.get(language, 0) + 1
    if not counts:
        return None
    # sorted() is stable, so a two-valued key partitions the list without
    # disturbing the order within either side.
    tagged.sort(key=lambda pair: not is_english(pair[1]))
    # One write of the whole joined string: a failure part way through must not
    # leave a half-written corpus where a complete one was staged.
    corpus_path.write_text("\n".join(line for line, _ in tagged) + "\n",
                           encoding="utf-8")
    return counts


# Priority order for published SDF columns: the document text and its brief
# (what a reader came for), then the short scalar metadata, then the widest
# column (variables), then doc_id last — pure lineage/join bookkeeping that
# trails everything else, including variables, the same content-then-
# metadata-then-lineage-then-id shape flatten_dad_corpus uses for the DAD
# config, so both configs read the same way on the Hub.
SDF_COLUMN_ORDER = ["content", "description", "language", "type_name",
                    "type_id", "register", "scores", "variables", "doc_id"]


def reorder_sdf_corpus(src: Path, dst: Path) -> int:
    """Write the published form of an SDF corpus: the same records as src, one
    JSON object per line, with each object's keys reordered to
    SDF_COLUMN_ORDER instead of sdf_pipeline/layer5_score.py's write order.

    Every SDF_COLUMN_ORDER key present on a record is emitted in that order;
    anything else on the record — a legacy field like subtype_id or role that
    predates the current schema, or a future field this list hasn't caught up
    to — is appended afterward in its original relative order. Nothing is ever
    dropped: this reorders columns, it does not select them, so an older
    committed run missing variables/description/type_name still publishes
    every field it has, just reordered.

    ensure_ascii=False for the same reason flatten_dad_corpus uses it: the
    default would turn most of a non-English document into \\uXXXX escapes.

    A line that isn't a parseable JSON object (blank, malformed) is written
    through unchanged rather than aborting the whole publish — column order is
    cosmetic, the same reasoning order_english_first already applies to row
    order, just decided per line here since there is no reason for one bad
    line to block reordering the rest of the file.

    Never writes to src — only dst — so the run's own final/sdf_corpus.jsonl
    is untouched; only the staged/published copy is reordered. Returns the
    number of lines written.
    """
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            record = None
            if stripped:
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    record = parsed
            if record is None:
                fout.write(line if line.endswith("\n") else line + "\n")
            else:
                row = {k: record[k] for k in SDF_COLUMN_ORDER if k in record}
                row.update((k, v) for k, v in record.items()
                          if k not in SDF_COLUMN_ORDER)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def flatten_dad_corpus(src: Path, dst: Path, languages: dict[str, str] | None = None,
                       cards: dict[str, dict] | None = None,
                       append: bool = False) -> int:
    """Write the published form of a DAD corpus: one flat record per example
    (user_prompt, assistant_response, language, variables, example_gid) instead
    of the training format's messages array, so the Hub viewer shows one
    readable column per field with no role/content nesting.

    The two text columns lead because they are what a visitor came to read.
    `language` and `variables` follow for the same reason in reverse — a short
    language cell and the one nested column cost almost nothing behind two
    wide text columns, but would push those columns off the viewer's first
    screen sitting in front of them. `example_gid` trails everything,
    including `variables` — it is pure lineage/join bookkeeping (still used
    internally, below, to look up `language` and `variables`), not something
    a reader needs in front of the content.

    `language` is the language the scenario was DEALT (see dad_languages), not
    one detected from the text, and it is emitted only when the map is
    non-empty — a run we cannot measure would otherwise grow a column that is
    null on every row and reads as broken. A single row that does not join
    carries null rather than a guessed "English": a visible gap is honest, an
    invented value is not.

    `variables` is the one nested column and the DAD counterpart of the SDF
    corpus's own: the whole hand this example's scenario was dealt, so a reader
    can tell which slice of the matrix a prompt/response pair came from rather
    than only which language it is in. It follows `language`'s two rules for
    the same reasons — emitted only when something resolves, null on a row that
    does not join — and sits second-to-last, ahead of only `example_gid`,
    because it is the widest cell on the row.

    Both are joined off the run rather than read from the corpus record, even
    though the record now carries `variables` itself: every committed run
    predates that field, and one source means the column cannot mean different
    things on either side of the change.

    That this column is worth carrying, while the old source_run column was
    not, is not a contradiction: unlike a run id repeated down every row,
    language carries information no other column holds — which is exactly what
    a reader needs once the corpus is ordered English-first and a balanced
    sample has to be rebuilt.

    Deliberately carries NO per-row run column, even when several runs are
    concatenated into one published corpus (append=True for every run after
    the first). Row-to-run attribution comes from the repo instead:
    example_gid is globally unique and content-keyed via the git-tracked
    dad/id_registry.json, so `git grep <gid> -- outputs/dad/runs` resolves any
    published row to exactly one committed run dir. Which runs went into a
    combined corpus is recorded by the per-run manifests staged under
    manifests/. A repeated run_id string on every row bought nothing that trace
    doesn't already give, and it dominated the viewer's first screen.

    The run's own final/dad_corpus.jsonl keeps the SFT chat shape — only the
    staged copy is flattened. A record without a user+assistant string pair
    aborts the publish rather than uploading a mangled row. Returns the number
    of records written.
    """
    n = 0
    with open(src, encoding="utf-8") as fin, \
         open(dst, "a" if append else "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            by_role: dict = {}
            for msg in record.get("messages") or []:
                if isinstance(msg, dict):
                    by_role.setdefault(msg.get("role"), msg.get("content"))
            if not (isinstance(by_role.get("user"), str)
                    and isinstance(by_role.get("assistant"), str)):
                rid = record.get("example_gid") or record.get("record_id") \
                    or f"line {line_no}"
                raise SystemExit(
                    f"{src}: record {rid} has no user+assistant message pair "
                    f"— refusing to publish a mangled row"
                )
            gid = record.get("example_gid")
            row = {}
            row["user_prompt"] = by_role["user"]
            row["assistant_response"] = by_role["assistant"]
            if languages:
                row["language"] = languages.get(gid)
            if cards:
                row["variables"] = dealt_variables(cards[gid]) if gid in cards else None
            row["example_gid"] = gid
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def stage_run(run_dirs: list[Path], corpus_name: str, staging_dir: Path,
              pipeline_tag: str) -> dict:
    """Copy the publishable subset of run dir(s) into staging_dir/<pipeline_tag>/.

    The per-pipeline subdirectory is what lets one repo hold both corpora as
    separate HF configs. Returns a manifest dict of what was staged, used to
    log what was (or, under --dry-run, would have been) uploaded.

    With ONE run dir the layout is the original single-run shape
    (run_manifest.json + audit/*). With several (DAD only — enforced in
    main()), the flattened corpora are concatenated into one jsonl and the
    per-run files move under run-scoped paths so they can't collide:
    manifests/<run_id>.json and audit/<run_id>/*. Those per-run manifests are
    the combined corpus's provenance record — the rows themselves carry no run
    column (see flatten_dad_corpus).
    """
    # Refuse a --staging-dir that equals or contains any run dir, OR either of
    # the two specific subtrees this function reads from (final/, audit/):
    # rmtree below would delete data we're about to read before the copy even
    # runs. Checking only run_dir itself isn't enough — a --staging-dir
    # pointing at run_dir/final or run_dir/audit directly (an easy typo, since
    # those are real, well-known subdirectory names on every run) would slip
    # past a run_dir-only check while still destroying the corpus or audit
    # reports.
    staging_real = staging_dir.resolve()
    for run_dir in run_dirs:
        for guarded, label in (
            (run_dir.resolve(), "the run directory"),
            ((run_dir / "final").resolve(), "the run's final/ directory"),
            ((run_dir / "audit").resolve(), "the run's audit/ directory"),
        ):
            if guarded.is_relative_to(staging_real):
                raise SystemExit(
                    f"--staging-dir {staging_dir} equals or contains {label} ({guarded}) "
                    f"— refusing to delete it. Pick a --staging-dir outside the run."
                )

    # Wipe first: a reused --staging-dir (e.g. re-running after fixing a typo'd
    # --input) must reflect only THIS run — otherwise leftover files from an
    # earlier invocation ride along into upload_folder silently mixed with
    # this run's data. But only if we own it: a non-empty directory this
    # script didn't create (no STAGING_MARKER, and not a pre-marker legacy
    # staging shape) is refused rather than wiped — this also makes --dry-run
    # non-destructive, since stage_run runs before the --dry-run early return.
    if staging_dir.exists():
        if not staging_dir.is_dir():
            raise SystemExit(f"--staging-dir {staging_dir} exists and is not a directory.")
        entries = sorted(p.name for p in staging_dir.iterdir())
        ours = (
            not entries                                   # empty: the common `mkdir` case
            or STAGING_MARKER in entries                   # staged by this script
            or (set(entries) <= STAGING_LEGACY_ENTRIES     # staged before the marker existed
                and any((staging_dir / p).is_dir() for p in ("sdf", "dad")))
        )
        if not ours:
            raise SystemExit(
                f"--staging-dir {staging_dir} already exists, is not empty, and "
                f"carries no {STAGING_MARKER} marker, so this script did not create "
                f"it — refusing to delete its {len(entries)} entry/entries "
                f"({', '.join(entries[:5])}...). Staging WIPES this directory. Pass "
                f"a new path, an empty directory, or one this script staged into."
            )
        shutil.rmtree(staging_dir)
    utils.ensure_dir(staging_dir)
    (staging_dir / STAGING_MARKER).write_text(
        "Created by evals/publish_hf.py. This directory is wiped and re-staged on "
        "every publish; nothing here is authoritative.\n", encoding="utf-8")
    # Everything for this run lives under <staging>/<pipeline_tag>/ so the
    # sibling pipeline can occupy its own sibling directory in the same repo.
    dataset_dir = utils.ensure_dir(staging_dir / pipeline_tag)
    multi = len(run_dirs) > 1
    staged: dict = {"pipeline": pipeline_tag, "corpus_file": None,
                    "manifest_file": None, "audit_files": [], "n_docs": 0,
                    "runs": [], "languages": None}

    # Built BEFORE the loop, because the flatten writes the joined columns as
    # it goes. Merged across runs since a combined corpus is one published
    # file; a gid repeated across runs means byte-identical content (the ids
    # are content-keyed via dad_pipeline/id_registry.py), so whichever run
    # wins the merge carries the same answer. The language is derived per run,
    # not from the merged map: its "no evidence, no column" gate is a statement
    # about one run's cards, and a run that carries none must not be handed a
    # language because a sibling run had some.
    dad_language_map: dict[str, str] = {}
    dad_cards_map: dict[str, dict] = {}
    if corpus_name == "dad_corpus.jsonl":
        for run_dir in run_dirs:
            run_cards = dad_dealt_cards(run_dir)
            dad_cards_map.update(run_cards)
            dad_language_map.update(dad_languages(run_cards))

    corpus_dst = dataset_dir / corpus_name
    for i, run_dir in enumerate(run_dirs):
        manifest = _load_json(run_dir / "run_manifest.json") or {}
        run_id = manifest.get("run_id") or run_dir.name

        corpus_src = run_dir / "final" / corpus_name
        if corpus_name == "dad_corpus.jsonl":
            n = flatten_dad_corpus(corpus_src, corpus_dst, dad_language_map,
                                   dad_cards_map, append=(i > 0))
        else:
            n = reorder_sdf_corpus(corpus_src, corpus_dst)
        staged["n_docs"] += n
        staged["runs"].append({"run_id": run_id, "n_docs": n})

        manifest_src = run_dir / "run_manifest.json"
        if manifest_src.exists():
            if multi:
                dst = utils.ensure_dir(dataset_dir / "manifests") / f"{run_id}.json"
            else:
                dst = dataset_dir / "run_manifest.json"
                staged["manifest_file"] = "run_manifest.json"
            shutil.copy2(manifest_src, dst)

        audit_src = run_dir / "audit"
        if audit_src.is_dir():
            audit_dst = utils.ensure_dir(
                dataset_dir / "audit" / run_id if multi else dataset_dir / "audit")
            # *.jsonl too: evals/audit_dad.py writes audit/tic_candidates.jsonl
            # and audit/reason_failures.jsonl for DAD runs — a fixed
            # *.json/*.html pattern silently dropped both.
            for pattern in ("*.json", "*.jsonl", "*.html"):
                for f in sorted(audit_src.glob(pattern)):
                    if f.name == "report_content.json":
                        continue  # editorial input, already baked into corpus_report.html
                    shutil.copy2(f, audit_dst / f.name)
                    staged["audit_files"].append(
                        f"{run_id}/{f.name}" if multi else f.name)

    # AFTER the run loop, so a combined DAD corpus is partitioned across the
    # WHOLE published file rather than once per run. Per-run partitioning would
    # only put English first if run_dirs[0] happened to hold enough English
    # rows — on the real input order the viewer's first screen would still turn
    # non-English part way down. Run order survives inside each language block
    # (the sort is stable), and nothing published reports row ranges — the
    # per-run manifests carry counts — so provenance stays true either way.
    # Both pipelines now publish the language on the row itself — SDF writes it
    # upstream, DAD gets it from the flatten above — so the ordering reads the
    # very column a visitor sees. That also makes the decline path fall out for
    # free: a run with no measurable language emits no column, every row reads
    # None, and order_english_first leaves the file alone.
    staged["languages"] = order_english_first(
        corpus_dst, lambda record: record.get("language"))

    staged["corpus_file"] = corpus_name
    return staged


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _create_repo(repo_id: str) -> None:
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def _upload_folder(folder_path: str, repo_id: str, commit_message: str,
                   delete_patterns: list[str]) -> str:
    from huggingface_hub import HfApi
    return HfApi().upload_folder(
        folder_path=folder_path, repo_id=repo_id, repo_type="dataset",
        commit_message=commit_message,
        # delete_patterns: republishing a run must not leave a PREVIOUS run's
        # audit files (e.g. one with realism_ablation.json alongside a later
        # one without it) lingering next to the new corpus. The caller
        # scopes this to the pipeline being published — a bare "audit/*" would
        # delete the SIBLING pipeline's audit files on every publish.
        delete_patterns=delete_patterns,
        # huggingface_hub only ignores .git* and .cache/huggingface by
        # default — a dotfile marker is NOT covered, so without this the
        # staging bookkeeping marker would ship as dataset content.
        ignore_patterns=[STAGING_MARKER],
    )


def _create_tag(repo_id: str, tag: str) -> None:
    from huggingface_hub import HfApi
    # exist_ok: a retried publish with the same --tag (e.g. after fixing a
    # typo'd --input) must not die here after the corpus has already been
    # re-uploaded — that would leave the run in a partially-completed state.
    HfApi().create_tag(repo_id=repo_id, tag=tag, repo_type="dataset", exist_ok=True)


def merge_state(run_commit: str | None, *, fetch: bool = True) -> dict:
    """Seam over utils.merge_state, so tests can pin a run's merge status rather
    than depending on whatever branch the developer happens to be on."""
    return utils.merge_state(run_commit, fetch=fetch)


def _unmerged_summary(stamp: dict) -> str:
    """One-line description of an unmerged stamp, for the Hub commit message.

    Names each run's branch and commit, not just its id. This is the only place
    an unmerged publish is recorded now, so it has to carry what a reader needs
    to go and look: the id says which rows, the branch and commit say which
    code. A run's own branch is where the data was GENERATED — distinct from
    publish_branch, the checkout that uploaded it, which is stated separately
    because conflating them would credit the corpus to whatever happens to be
    checked out at publish time.
    """
    parts = []
    if runs := stamp.get("runs"):
        named = ", ".join(
            f"{r.get('run_id') or 'unknown'} "
            f"(branch {r.get('branch') or 'unknown'}, "
            f"commit {r.get('commit') or 'unknown'})"
            for r in runs)
        parts.append(f"unmerged run(s): {named}")
    if stamp.get("publish_branch"):
        parts.append(f"published from unmerged branch {stamp['publish_branch']}")
    return "; ".join(parts) or "unmerged"


def check_merged(run_dirs: list[Path], *, dry_run: bool,
                 allow_unmerged: bool) -> dict | None:
    """Pre-flight provenance gate. Returns a stamp describing what is NOT backed
    by merged code, or None when everything checks out:

        {"publish_branch": <branch, only if HEAD isn't verified merged>,
         "runs": [{"run_id", "branch", "commit"}, ...]}

    Every input run is checked separately, and the stamp NAMES each unverified
    one. A combined corpus is only as merged as its least-merged run, and a row
    can be traced to the run — and therefore the code — that produced it, via
    the repo lookup example_gid supports (see flatten_dad_corpus). Naming each
    run is what connects that trace to a merge verdict; collapsing them into
    one would leave a reader able to identify a row's run but not whether that
    run's code was reviewed.

    Deliberately a warning-plus-confirmation rather than a refusal. The HF write
    token lives on contributors' laptops, so a hard block wouldn't prevent an
    unmerged publish — it would push it out of this script, which is the only
    thing that records provenance at all. What makes the check stick is that
    _unmerged_summary writes the stamp into the Hub COMMIT MESSAGE, not this
    terminal warning. The dataset card would be the more visible place, but it
    is hand-edited on the Hub and this script no longer writes it — and a
    warning someone can quietly edit away is not a record. A commit message
    cannot be edited after the fact.

    A dirty tree at run time is reported as context but is never itself a
    trigger: every real run so far has been dirty, and a warning that fires on
    every run is one people learn to type straight past.
    """
    # One fetch for the whole publish: merge_state would otherwise hit the
    # network once per run dir, and every check compares against the same
    # origin/main anyway.
    checked = []
    for i, run_dir in enumerate(run_dirs):
        manifest = _load_json(run_dir / "run_manifest.json") or {}
        state = merge_state(manifest.get("git_commit"),
                            fetch=(i == 0 and not dry_run))
        checked.append((run_dir, manifest, state))

    # head_merged describes the checkout doing the publishing, so it is the same
    # for every run — read it off the first.
    head_state = checked[0][2]
    unverified = [(rd, m, s) for rd, m, s in checked
                  if s["run_commit_merged"] is not True]
    if head_state["head_merged"] is True and not unverified:
        return None

    reasons = []
    if head_state["head_merged"] is False:
        ahead = head_state["ahead"]
        reasons.append(
            f"the current branch `{head_state['branch']}` has "
            f"{f'{ahead} commit(s)' if ahead else 'commits'} "
            f"not in {utils.MAIN_REF}")
    for run_dir, manifest, state in unverified:
        if state["run_commit_merged"] is not False:
            continue
        dirty = manifest.get("git_dirty_files")
        dirty_note = ""
        if dirty:
            dirty_note = f", plus {len(dirty)} uncommitted file(s) at run time"
        elif manifest.get("git_dirty"):
            dirty_note = ", plus uncommitted changes at run time"
        reasons.append(
            f"run {manifest.get('run_id') or run_dir.name} was generated from "
            f"commit {state['run_commit']}, which is not in "
            f"{utils.MAIN_REF}{dirty_note}")

    # "Not merged" only when something is definitely not merged. When every
    # check came back unknown, say THAT — overstating it teaches people the
    # warning is inaccurate, which is how a guardrail loses its authority.
    subject = "This run" if len(run_dirs) == 1 else "This publish"
    headline = (f"{subject} has NOT been merged into main." if reasons else
                f"{subject}'s provenance could NOT be verified against main.")
    bar = "=" * 68
    print(f"\n{bar}", file=sys.stderr)
    print(f"  {headline}", file=sys.stderr)
    for reason in reasons:
        print(f"    - {reason}", file=sys.stderr)
    # Notes explain why something is UNKNOWN; printed as caveats rather than
    # mixed in with the findings, which would read as reasons for the verdict.
    # Deduplicated: with several runs the same caveat (a stale origin/main, say)
    # would otherwise repeat once per run.
    for note in dict.fromkeys(n for _, _, s in checked for n in s["notes"]):
        print(f"    (note: {note})", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Publishing anyway records this in the Hub commit message, publicly.",
          file=sys.stderr)
    print("  If this is meant to be a canonical snapshot, merge your pull "
          "request first", file=sys.stderr)
    print("  and re-run this on main.", file=sys.stderr)
    print(f"{bar}\n", file=sys.stderr)

    # Attribute the branch each run was GENERATED on, not the one it happens to
    # be published from — they differ, and the claim being recorded is about the
    # code behind the corpus. Only v3+ manifests record it, so fall back to the
    # live checkout for every run predating that.
    stamp: dict = {
        "runs": [
            {"run_id": m.get("run_id") or rd.name,
             "branch": m.get("git_branch") or s["branch"],
             "commit": s["run_commit"]}
            for rd, m, s in unverified
        ],
    }
    # A merged run can still be published from an unmerged checkout, which says
    # nothing about any individual run — so it is recorded separately.
    if head_state["head_merged"] is not True:
        stamp["publish_branch"] = head_state["branch"]
    if dry_run:
        # Nothing is published, so there is nothing to confirm — but the
        # preview still shows the stamp this run would carry.
        return stamp
    if allow_unmerged:
        print("Proceeding: --allow-unmerged was passed.", file=sys.stderr)
        return stamp
    if not sys.stdin.isatty():
        # A prompt nobody can see would hang an agent, a pipe, or a CI job
        # forever. Make the bypass an explicit, greppable flag instead.
        raise SystemExit(
            "Refusing to publish an unmerged run without confirmation. Re-run "
            "interactively, or pass --allow-unmerged to publish anyway.")
    try:
        answer = input("Type 'yes' to publish anyway: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nAborted.")
    if answer != "yes":
        raise SystemExit("Aborted — nothing was published.")
    return stamp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a run's final corpus + audit reports as a Hugging Face dataset."
    )
    parser.add_argument("--input", required=True, nargs="+",
                        help="Run directory (SDF or DAD). Several DAD run dirs "
                             "publish as ONE combined corpus, with each run's "
                             "manifest staged under manifests/; SDF takes "
                             "exactly one.")
    parser.add_argument("--repo-id", required=True,
                        help="e.g. sentientfutures/animal-welfare-training-claude")
    parser.add_argument("--tag", default=None,
                        help="Tag to create on the upload commit. Tags are repo-wide, so "
                             "prefix per pipeline (sdf-v1-..., dad-v1-...) once the repo "
                             "holds more than one dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage locally and print what would be uploaded; "
                             "make no Hub API calls")
    parser.add_argument("--staging-dir", default=None,
                        help="Where to stage files (default: a temp dir). Staging wipes "
                             "this directory, so it must be new, empty, or one a "
                             "previous publish staged into.")
    parser.add_argument("--allow-unmerged", action="store_true",
                        help="Publish even though this run's code is not in "
                             "origin/main, without the interactive "
                             "confirmation. The Hub commit message still "
                             "records it as an unmerged publish")
    args = parser.parse_args()

    resolved = [resolve_corpus_file(p) for p in args.input]
    run_dirs = [r[0] for r in resolved]
    corpus_names = {r[1] for r in resolved}
    if len(corpus_names) > 1:
        raise SystemExit("All --input run dirs must belong to the same pipeline "
                         f"(got {sorted(corpus_names)})")
    corpus_name = corpus_names.pop()
    if corpus_name == "sdf_corpus.jsonl" and len(run_dirs) > 1:
        # SDF has no cross-run stable id (doc_id is per-run, unlike DAD's
        # globally unique example_gid), so a concatenation would leave rows no
        # way to be traced back to a run — not even through the repo.
        raise SystemExit("Combined publishing is DAD-only; pass one SDF run dir.")
    if len(set(run_dirs)) != len(run_dirs):
        raise SystemExit("Duplicate --input run dirs would double their rows "
                         "in the combined corpus.")
    pipeline_tag = "sdf" if corpus_name == "sdf_corpus.jsonl" else "dad"

    # Before staging (which wipes a directory) and before any Hub call, so an
    # aborted publish leaves nothing behind. Runs in --dry-run too: a preview
    # that hid the warning would be the wrong preview.
    unmerged = check_merged(run_dirs, dry_run=args.dry_run,
                            allow_unmerged=args.allow_unmerged)

    import contextlib
    import tempfile

    if args.staging_dir:
        # Explicitly requested — never ours to delete.
        staging_ctx = contextlib.nullcontext(args.staging_dir)
    elif args.dry_run:
        # --dry-run's whole point is to let a human inspect the staged output
        # afterward, so this directory must outlive the process.
        staging_ctx = contextlib.nullcontext(tempfile.mkdtemp(prefix="publish_hf_"))
    else:
        staging_ctx = tempfile.TemporaryDirectory()

    with staging_ctx as tmp:
        staging_dir = Path(tmp) if args.staging_dir else Path(tmp) / "staged"
        staged = stage_run(run_dirs, corpus_name, staging_dir, pipeline_tag)

        run_names = ", ".join(r["run_id"] for r in staged["runs"])
        print(f"Staged {pipeline_tag}/{corpus_name} ({staged['n_docs']} records "
              f"from {len(staged['runs'])} run(s): {run_names}), "
              f"{len(staged['audit_files'])} audit file(s): {', '.join(staged['audit_files']) or '(none)'}")
        # Say what the ordering pass did, either way. A publish whose rows
        # could not be measured looks identical from the outside otherwise,
        # and the whole point of the pass is what a visitor lands on.
        if staged["languages"]:
            by_count = sorted(staged["languages"].items(),
                              key=lambda kv: (-kv[1], kv[0]))
            n_en = sum(n for name, n in by_count if is_english(name))
            others = ", ".join(f"{name} {n}" for name, n in by_count
                               if not is_english(name))
            # Rows whose language could not be read sort behind the English
            # block as well, but they are NOT evidence of a non-English
            # corpus — they are evidence of a run that recorded no card
            # (archetype10 is the committed example). Counting them into the
            # language tally would misreport both numbers, so name them apart.
            unmeasured = staged["n_docs"] - sum(n for _, n in by_count)
            note = (f"  Ordered English first: {n_en} of {staged['n_docs']} "
                    f"rows lead, then {others or '(none)'}")
            if unmeasured:
                note += (f"; {unmeasured} row(s) carry no recorded language "
                         f"and sort behind the block too")
            print(note)
            if others:
                # The card's `language:` list used to be derived from exactly
                # this breakdown. It is hand-maintained now, and a card that
                # declares English over a corpus that is 19% not is a false
                # claim on a public dataset — so the numbers a publisher needs
                # are put in front of them at the moment they'd have to act.
                print("  The card's `language:` list is hand-maintained on the "
                      "Hub — check it still covers the above.")
        else:
            print("  Not reordered: no readable language on these records — "
                  "publishing in the order the run wrote them.")

        # run_names (plural) is main's combined-publish naming; the unmerged
        # marker rides on the end of it rather than replacing it. This is where
        # an unmerged publish is recorded durably — see check_merged — so it is
        # built before the --dry-run return and previewed there. A preview that
        # showed the terminal warning but not the record it would leave behind
        # would be showing the half that doesn't last.
        commit_message = f"Publish {pipeline_tag}: {run_names}"
        if unmerged:
            # Visible in the repo's commit history, not just this terminal.
            commit_message += f" ({_unmerged_summary(unmerged)})"

        if args.dry_run:
            print(f"\n--dry-run: no Hub API calls made. Staged at {staging_dir} "
                  f"(left on disk for inspection).")
            print(f"Would commit as: {commit_message}")
            print("The dataset card is not staged, here or on a real publish: "
                  "it is hand-written and edited on the Hub.")
            return

        _create_repo(args.repo_id)

        print("  Card: leaving the Hub's README.md as it is — it is "
              "hand-written and edited there.")

        commit = _upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            commit_message=commit_message,
            # Scoped to THIS pipeline — a bare "audit/*" would delete the
            # sibling's audit files on every publish. README.md matches none of
            # these patterns and is never staged, so the Hub's card is neither
            # overwritten nor deleted.
            #
            # card_meta.json is a sidecar this script no longer writes (it fed
            # the removed card generator). Nothing is staged at that path any
            # more, so the deletion is no longer suppressed by an add and the
            # next publish of a pipeline clears its orphan — sdf/ has one on the
            # Hub, dad/ does not. Drop this pattern once that publish happens.
            # run_manifest.json and manifests/* are both listed so a publish
            # that switches layout (single-run <-> combined) clears the OTHER
            # layout's manifest file(s) — upload_folder drops any deletion
            # whose path is also being added, so the layout actually staged
            # always survives its own pattern.
            delete_patterns=[f"{pipeline_tag}/audit/*",
                             f"{pipeline_tag}/run_manifest.json",
                             f"{pipeline_tag}/manifests/*",
                             f"{pipeline_tag}/{LEGACY_CARD_META_FILENAME}"],
        )
        if args.tag:
            _create_tag(args.repo_id, args.tag)
        print(f"\nPublished to https://huggingface.co/datasets/{args.repo_id}")
        print(f"Commit: {commit}")
        if args.tag:
            print(f"Tag: {args.tag}")


if __name__ == "__main__":
    main()
