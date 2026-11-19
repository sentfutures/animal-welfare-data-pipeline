#!/usr/bin/env python3
"""Publish a run's final corpus and audit reports as a Hugging Face dataset, staged under ``sdf/`` or ``dad/`` so one repo holds both as separate configs.
This script never writes the dataset card (hand-edited on the Hub, and it holds the ``configs:`` block); the rules are in ``evals/CLAUDE.md``.
Usage: ``python evals/publish_hf.py --input outputs/sdf/latest --repo-id $REPO --dry-run`` (``--dry-run`` makes no network calls); tags are repo-wide, so prefix them per pipeline."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import utils
from shared.api_keys import (add_api_key_source_flags, apply_api_key_source_flags,
                             hide_environment_value_unless_environment_is_allowed, load_optional_api_key)
# The one regex that reads a writing language out of a dealt culture/setting card; imported, not copied, because it is a contract with the wording in prompts/{sdf,dad}/variables.txt.
# Cheap to import: compose_prompts pulls only shared.matrix and shared.entity_pools.
from sdf_pipeline.shared_step_machinery.compose_prompts import derive_language
# The offline composer, not step1_dilemmas: see dad_dealt_cards on why this
# script must not reach anything that pulls shared.api -> anthropic.
from dad_pipeline.compose_scenarios import dealt_variables


CORPUS_FILENAMES = ("sdf_corpus.jsonl", "dad_corpus.jsonl")

# Sidecar this script does not write; the delete pattern in main() clears any
# copy left on the Hub.
LEGACY_CARD_META_FILENAME = "card_meta.json"

# Ownership marker for a staging directory this script created: stage_run wipes its staging root, so a non-empty directory is deletable only when the marker proves we made it.
STAGING_MARKER = ".publish_hf_staging"
# Entries an existing staging root may hold and still be reusable without a
# manual delete: the two per-pipeline dirs, plus README.md. A staging dir
# created by this script never contains README.md.
STAGING_LEGACY_ENTRIES = {"README.md", "sdf", "dad"}

# Both spellings the corpora use: the full name derive_language writes into SDF records, and the bare ISO code four early SDF runs carry (audit_sdf.py accepts both).
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
    """True only for a language value that reads as English; unlike ``audit_sdf.py`` a missing value is NOT English, since the front of the published file is a promise."""
    return str(language or "").strip().lower() in ENGLISH_SPELLINGS


def dad_dealt_cards(run_dir: Path) -> dict[str, dict]:
    """``{example_gid: the cards its scenario was dealt}`` for one DAD run (from step3/rewrites.jsonl), or {} when it cannot be read.
    The published ``language`` and ``variables`` columns both derive from this one map."""
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
            # `annotation` is the old spelling of scenario_cards. dad_pipeline/step1_dilemmas.py's dealt_cards() is not imported: it pulls shared.api -> anthropic into a script that makes no model calls.
            cards = record.get("scenario_cards") or record.get("annotation") or {}
            cards_by_gid[gid] = cards if isinstance(cards, dict) else {}
    return cards_by_gid


def dad_languages(cards_by_gid: dict[str, dict]) -> dict[str, str]:
    """``{example_gid: language}`` read from each scenario's ``cultural_setting`` card (unmarked scenarios are English), or {} when no record has one.
    This is the language dealt, not detected; the gate stays per run so a run without languages keeps its corpus order."""
    languages: dict[str, str] = {}
    any_dealt = False
    for gid, cards in cards_by_gid.items():
        setting = cards.get("cultural_setting")
        if setting:
            any_dealt = True
        languages[gid] = derive_language(setting or "")
    return languages if any_dealt else {}


def order_english_first(corpus_path: Path, language_of) -> dict[str, int] | None:
    """Rewrite a staged corpus jsonl with English rows first (a stable partition of the raw lines, each written back unchanged except a normalised ``\\n`` ending), returning the language counts.
    Declines (None, file untouched) on a blank or non-object line, or when no row's language is readable."""
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


# Published SDF column priority: the text and its brief, short scalar metadata, the widest column (variables), then doc_id (lineage); the same shape as flatten_dad_corpus.
SDF_COLUMN_ORDER = ["content", "description", "language", "type_name",
                    "type_id", "register", "scores", "variables", "variable_value_ids",
                    "doc_id"]


def reorder_sdf_corpus(src: Path, dst: Path) -> int:
    """Write the published SDF corpus: ``src``'s records with keys in SDF_COLUMN_ORDER (other keys appended in order, none dropped), ``ensure_ascii=False``; returns the lines written.
    A line that does not parse is written through unchanged; ``src`` is never modified."""
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
    """Write the published DAD corpus: one flat record per example (user_prompt, assistant_response, language, variables, example_gid), with no per-row run column.
    ``language`` and ``variables`` come from the run's dealt cards, null on a row that does not join; a record without a user and assistant string aborts. Returns the number written."""
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
    """Copy the publishable subset of the run dir(s) into ``staging_dir/<pipeline_tag>/``; returns a manifest dict of what was staged.
    Several run dirs (DAD only) are concatenated into one corpus, with per-run files under ``manifests/<run_id>.json`` and ``audit/<run_id>/*``."""
    # Refuse a --staging-dir that equals or contains a run dir, its final/ or its audit/: the rmtree below would delete data about to be read.
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

    # Wipe first, so a reused --staging-dir reflects only this run; but only a directory this script created (STAGING_MARKER or a legacy staging shape) is wiped, otherwise it is refused.
    # That also keeps --dry-run non-destructive, since stage_run runs before its early return.
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

    # Built before the loop, because the flatten writes the joined columns as it goes; merged across runs (a repeated gid is byte-identical content).
    # Language is derived per run, not from the merged map: its "no evidence, no column" gate is about one run's cards.
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

    # After the run loop, so a combined corpus is partitioned once across the whole file (the sort is stable, so run order survives inside each language block).
    # Both pipelines publish language on the row, so a run with no measurable language emits no column and order_english_first leaves the file alone.
    staged["languages"] = order_english_first(
        corpus_dst, lambda record: record.get("language"))

    staged["corpus_file"] = corpus_name
    return staged


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_hugging_face_token() -> None:
    """``HF_TOKEN`` is optional (the cached ``huggingface-cli login`` token is the fallback); in the default mode an environment value is removed unless the .env file supplied it."""
    if load_optional_api_key("HF_TOKEN") is None:
        hide_environment_value_unless_environment_is_allowed("HF_TOKEN")
    hide_environment_value_unless_environment_is_allowed("HUGGING_FACE_HUB_TOKEN")


def _create_repo(repo_id: str) -> None:
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def _upload_folder(folder_path: str, repo_id: str, commit_message: str,
                   delete_patterns: list[str]) -> str:
    from huggingface_hub import HfApi
    return HfApi().upload_folder(
        folder_path=folder_path, repo_id=repo_id, repo_type="dataset",
        commit_message=commit_message,
        # delete_patterns: republishing must not leave a previous run's audit files next to the new corpus; the caller scopes it to the published pipeline.
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
    """One-line description of an unmerged stamp for the Hub commit message, naming each run's branch and commit and, separately, the checkout that published it."""
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
    """Pre-flight provenance gate: ``{"publish_branch", "runs": [{"run_id", "branch", "commit"}]}`` naming everything NOT backed by merged code, or None.
    A warning plus confirmation, not a refusal; the stamp goes in the Hub commit message. A dirty tree is context, never a trigger."""
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
    # Notes explain why something is unknown, so print them as caveats apart from the findings, deduplicated across runs.
    for note in dict.fromkeys(n for _, _, s in checked for n in s["notes"]):
        print(f"    (note: {note})", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Publishing anyway records this in the Hub commit message, publicly.",
          file=sys.stderr)
    print("  If this is meant to be a canonical snapshot, merge your pull "
          "request first", file=sys.stderr)
    print("  and re-run this on main.", file=sys.stderr)
    print(f"{bar}\n", file=sys.stderr)

    # Attribute the branch each run was generated on, not the publishing checkout; only v3+ manifests record it, so older runs fall back to the live checkout.
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
    add_api_key_source_flags(parser)
    args = parser.parse_args()
    apply_api_key_source_flags(args)

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

    if not args.dry_run:
        _load_hugging_face_token()

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
            # Rows with an unreadable language sort behind the English block but are evidence of a run that recorded no card (archetype10), not of a non-English corpus, so name them apart.
            unmeasured = staged["n_docs"] - sum(n for _, n in by_count)
            note = (f"  Ordered English first: {n_en} of {staged['n_docs']} "
                    f"rows lead, then {others or '(none)'}")
            if unmeasured:
                note += (f"; {unmeasured} row(s) carry no recorded language "
                         f"and sort behind the block too")
            print(note)
            if others:
                # The card's `language:` list is hand-maintained, so put the numbers a publisher needs in front of them when they must act (a card declaring English over a 19% non-English corpus is a false claim).
                print("  The card's `language:` list is hand-maintained on the "
                      "Hub — check it still covers the above.")
        else:
            print("  Not reordered: no readable language on these records — "
                  "publishing in the order the run wrote them.")

        # run_names is main's combined-publish naming; the unmerged marker is appended. This is where an unmerged publish is durably recorded, so it is built before the --dry-run return and previewed there.
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
            # Scoped to this pipeline (a bare "audit/*" would delete the sibling's files); README.md matches no pattern and is never staged.
            # Both manifest patterns are listed so a layout switch (single-run <-> combined) clears the other layout's manifest; a path also being added survives its own pattern.
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
