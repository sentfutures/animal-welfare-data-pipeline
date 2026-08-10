"""Tests for evals/publish_hf.py — staging and the Hub-upload chokepoints
(stubbed via the stub_hf fixture; never touches huggingface_hub or the
network).

There is nothing here about building a dataset card, because this script no
longer builds one: the card is hand-written and edited on the Hub. What that
leaves is a contract worth testing in its own right — a publish must never
stage or delete README.md — and it is pinned in TestMainEndToEnd.
"""

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from evals import publish_hf
from dad_pipeline.compose_scenarios import DEALT_CARD_FIELDS

COMPLIANCE = {"judged": 100, "clean_documents": 98, "clean_frac": 0.98}
CARD_FIDELITY = {
    "judged": 99,
    "honoured": {"document_type": 96, "resolution": 65, "centrality": 84},
    "by_card_frac": {"document_type": 0.97, "resolution": 0.657, "centrality": 0.848},
}
DIVERSITY = {"n_records": 477, "vendi": {"score": 34.45, "ratio": 0.0722},
             "mean_pairwise_cosine": 0.3709}
REALISM = {"n": 78, "layer5_mean": 8.487, "blind_same_rubric_mean": 5.782, "mean_drop": 2.705}
VENDI_CURVE = {"proj": {"1000": {"power": 50.1, "log": 48.2}, "5000": {"power": 90.4, "log": 85.1}}}
AUDIT_REPORT = {
    "n_docs": 477,
    "composition": {"language": {
        "English": 139, "Spanish": 48, "Mandarin Chinese": 57, "Urdu": 10,
    }},
}
REPORT_CONTENT = {"title": "SDF corpus audit — 477 documents", "subtitle": "A test subtitle."}
MANIFEST = {
    "run_id": "2026-07-25_15-57_fullscale-500-opus5",
    "label": "fullscale-500-opus5",
    "git_commit": "4abd78b",
    # Shaped like the real published run: the top-level model is the default,
    # while the stage that matters was overridden to Opus.
    "model": "claude-sonnet-5",
    "config": {"backend": "claude_code",
               "sdf": {"rewrite_model": "claude-opus-5", "n_prompts": 500}},
}

KNOWN_AUDIT_FILES = {
    "audit_report.json": AUDIT_REPORT,
    "compliance_report.json": COMPLIANCE,
    "card_fidelity_report.json": CARD_FIDELITY,
    "diversity_report.json": DIVERSITY,
    "realism_ablation.json": REALISM,
    "vendi_curve.json": VENDI_CURVE,
    "report_content.json": REPORT_CONTENT,
}

# A fully-merged verdict — what merge_state returns on a clean main checkout.
MERGED_STATE = {
    "branch": "main", "head": "4abd78b", "head_merged": True, "ahead": 0,
    "run_commit": "4abd78b", "run_commit_merged": True,
    "fetched": True, "notes": [],
}


def unmerged_state(branch="declan/wip", commit="deadbee", ahead=3, **over):
    """merge_state's verdict for a run whose code never reached origin/main."""
    return {**MERGED_STATE, "branch": branch, "head": commit, "ahead": ahead,
            "head_merged": False, "run_commit": commit,
            "run_commit_merged": False, **over}


@pytest.fixture(autouse=True)
def _default_merged(monkeypatch):
    """Pin merge_state to "merged" for every test in this module.

    Without this the real helper runs, and the suite's result would depend on
    the branch the developer happens to be on — green on main, and blocked on a
    typed confirmation everywhere else. Tests that exercise the guard override
    this with their own monkeypatch.
    """
    monkeypatch.setattr(publish_hf, "merge_state",
                        lambda commit, fetch=True: dict(MERGED_STATE))


def marked_setting(language):
    """A cultural_setting card in the exact wording derive_language parses.

    English is the UNMARKED slice in prompts/dad/variables.txt — ~65% of a run
    is dealt "no particular location or culture" and stored as null — so an
    English row gets no card, and derive_language falls through to English on
    the empty string. Mirroring that here is what makes
    test_the_unmarked_cultural_setting_slice_is_treated_as_english real rather
    than a tautology over a value the pipeline never writes.
    """
    if language == "English":
        return None
    return f"Somewhere, written in {language}, with local idioms and references"


def make_run_dir(tmp_path, pipeline="sdf", docs=3, audit_files=None, manifest=MANIFEST,
                  include_html=True, extra_audit_files=None, run_name=None,
                  languages=None, gid_start=0):
    """Build a fake run directory with the given audit files present.

    audit_files=None means "all six known + report_content.json + html";
    pass a subset of KNOWN_AUDIT_FILES' keys to omit others. run_name lets a
    combined-publish test build several distinct run dirs under one tmp_path.

    languages=None (the default) builds a run with NO language data at all —
    an SDF corpus whose records have no `language` key, and a DAD run with no
    step3/rewrites.jsonl. That is the ordering pass's decline path, which is
    what keeps every pre-existing test in this module asserting the behaviour
    it was written for. Pass a list of language names to cycle over the records
    and put the reorder in play.

    gid_start offsets the DAD example_gids. Two runs default to the SAME gids,
    which is deliberate for the tests that prove rows are not distinguished by
    id; a test that joins per-row language across runs needs distinct ones,
    because two runs disagreeing about one gid's language is a state the real
    pipeline cannot produce (gids are content-keyed via
    dad_pipeline/id_registry.py, so a shared gid means identical content).
    """
    run_dir = tmp_path / "runs" / (run_name or "2026-07-25_15-57_fullscale-500-opus5")
    final = run_dir / "final"
    final.mkdir(parents=True)
    corpus_name = "sdf_corpus.jsonl" if pipeline == "sdf" else "dad_corpus.jsonl"

    def language_at(i):
        return languages[i % len(languages)] if languages else None

    if pipeline == "sdf":
        lines = []
        for i in range(docs):
            record = {"doc_id": f"d{i}", "content": f"document {i}"}
            if languages:
                record["language"] = language_at(i)
            lines.append(json.dumps(record))
    else:
        lines = [json.dumps({
            "record_id": f"r{i}", "example_gid": f"E-{gid_start + i:04d}",
            "response_gid": f"R-{gid_start + i:04d}",
            "messages": [
                {"role": "user", "content": f"user prompt {i}"},
                {"role": "assistant", "content": f"assistant response {i}"},
            ],
        }) for i in range(docs)]
        if languages:
            # Shaped like the real file: publish_hf joins the published row's
            # example_gid straight to this record's, and reads the language off
            # the cards it was dealt.
            step3 = run_dir / "step3"
            step3.mkdir(parents=True)
            (step3 / "rewrites.jsonl").write_text("\n".join(
                json.dumps({"record_id": f"r{i}",
                            "example_gid": f"E-{gid_start + i:04d}",
                            "scenario_cards": {
                                "cultural_setting": marked_setting(language_at(i))}})
                for i in range(docs)) + "\n", encoding="utf-8")
    (final / corpus_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if manifest is not None:
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    names = list(KNOWN_AUDIT_FILES) if audit_files is None else audit_files
    if names or include_html or extra_audit_files:
        audit = run_dir / "audit"
        audit.mkdir()
        for name in names:
            (audit / name).write_text(json.dumps(KNOWN_AUDIT_FILES[name]), encoding="utf-8")
        if include_html:
            (audit / "corpus_report.html").write_text("<html>report</html>", encoding="utf-8")
        for name, content in (extra_audit_files or {}).items():
            (audit / name).write_text(json.dumps(content), encoding="utf-8")

    return run_dir, corpus_name


def _tag_for(corpus_name):
    return "sdf" if corpus_name.startswith("sdf") else "dad"


def _stage(run_dir, corpus_name, staging_dir):
    """stage_run for the pipeline implied by corpus_name, plus the per-pipeline
    dataset dir it staged into."""
    tag = _tag_for(corpus_name)
    run_dirs = run_dir if isinstance(run_dir, list) else [run_dir]
    staged = publish_hf.stage_run(run_dirs, corpus_name, staging_dir, tag)
    return staged, staging_dir / tag


class TestResolveRunDir:
    def test_sdf_run(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path, pipeline="sdf")
        resolved, corpus_name = publish_hf.resolve_corpus_file(str(run_dir))
        assert resolved == run_dir
        assert corpus_name == "sdf_corpus.jsonl"

    def test_dad_run(self, tmp_path):
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad")
        _, corpus_name = publish_hf.resolve_corpus_file(str(run_dir))
        assert corpus_name == "dad_corpus.jsonl"

    def test_missing_corpus_raises(self, tmp_path):
        empty = tmp_path / "runs" / "empty-run"
        empty.mkdir(parents=True)
        with pytest.raises(SystemExit):
            publish_hf.resolve_corpus_file(str(empty))

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            publish_hf.resolve_corpus_file(str(tmp_path / "nope"))


class TestStageRun:
    def test_stages_corpus_manifest_and_audit_files(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)

        assert staged["corpus_file"] == corpus_name
        assert staged["pipeline"] == "sdf"
        assert staged["n_docs"] == 3
        assert staged["manifest_file"] == "run_manifest.json"
        # everything lands under <pipeline>/, never at the staging root — that
        # subdir is what lets one repo hold both corpora as separate configs
        assert dataset_dir == staging_dir / "sdf"
        assert (dataset_dir / corpus_name).exists()
        assert (dataset_dir / "run_manifest.json").exists()
        assert not (staging_dir / corpus_name).exists()
        assert not (staging_dir / "run_manifest.json").exists()
        # report_content.json is editorial and must never be staged/uploaded
        assert "report_content.json" not in staged["audit_files"]
        assert not (dataset_dir / "audit" / "report_content.json").exists()
        assert set(staged["audit_files"]) == {
            "audit_report.json", "compliance_report.json", "card_fidelity_report.json",
            "diversity_report.json", "realism_ablation.json", "vendi_curve.json",
            "corpus_report.html",
        }

    def test_no_audit_dir_is_fine(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["audit_files"] == []

    def test_no_manifest_is_fine(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, manifest=None, audit_files=[],
                                            include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["manifest_file"] is None

    def test_unknown_audit_file_is_staged_anyway(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, audit_files=[], include_html=False,
            extra_audit_files={"custom_eval.json": {"foo": "bar"}},
        )
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["audit_files"] == ["custom_eval.json"]

    def test_jsonl_audit_files_are_staged(self, tmp_path):
        """Regression: the glob only matched *.json/*.html, silently dropping
        evals/audit_dad.py's audit/tic_candidates.jsonl and
        audit/reason_failures.jsonl for DAD runs."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", audit_files=[], include_html=False,
            extra_audit_files={
                "tic_candidates.jsonl": {"phrase": "so I've been thinking"},
                "reason_failures.jsonl": {"reason": "example"},
            },
        )
        staging_dir = tmp_path / "staged"
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert set(staged["audit_files"]) == {"tic_candidates.jsonl", "reason_failures.jsonl"}
        assert (dataset_dir / "audit" / "tic_candidates.jsonl").exists()

    def test_reused_staging_dir_is_cleared_of_stale_files(self, tmp_path):
        """Regression: a --staging-dir reused across two invocations (e.g. after
        fixing a typo'd --input) must reflect only the LATEST run — leftover
        files from an earlier call must not ride along into the upload.

        Under the per-pipeline layout the staging ROOT is wiped, so an earlier
        run's whole `<pipeline>/` dir goes with it. That's intended: the sibling
        dataset's files stay on the Hub untouched — upload_folder only writes
        paths present in the staged folder — so nothing depends on whatever
        happens to be left in a local staging dir."""
        staging_dir = tmp_path / "staged"

        run_a, corpus_a = make_run_dir(tmp_path / "a", audit_files=["compliance_report.json"])
        _, dir_a = _stage(run_a, corpus_a, staging_dir)
        assert (dir_a / "audit" / "compliance_report.json").exists()

        run_b, corpus_b = make_run_dir(
            tmp_path / "b", pipeline="dad", audit_files=["audit_report.json"], include_html=False,
        )
        staged_b, dir_b = _stage(run_b, corpus_b, staging_dir)

        assert staged_b["audit_files"] == ["audit_report.json"]
        # run A's entire pipeline dir is gone, corpus and audit alike
        assert not dir_a.exists()
        assert not (staging_dir / "sdf").exists()
        assert (dir_b / corpus_b).exists()

    def test_staging_dir_equal_to_run_dir_is_rejected(self, tmp_path):
        """Regression: rmtree(staging_dir) must never fire before verifying
        staging_dir doesn't equal or contain run_dir — otherwise a mistyped
        --staging-dir pointing back at --input deletes the run being
        published before it can even be copied."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir, "sdf")
        # the run must survive the rejected attempt intact
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_that_contains_run_dir_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir.parent, "sdf")
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_equal_to_run_final_is_rejected(self, tmp_path):
        """Regression: the run_dir-only check missed this — a --staging-dir
        pointing directly at run_dir/final (an easy typo, since 'final' is a
        real, well-known subdirectory name on every run) slipped past it,
        and rmtree then deleted the corpus before it could be copied."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir / "final", "sdf")
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_equal_to_run_audit_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, run_dir / "audit", "sdf")
        assert (run_dir / "audit" / "audit_report.json").exists()

    def test_staging_dir_nested_inside_run_dir_is_allowed(self, tmp_path):
        """The reverse nesting is safe as long as it doesn't overlap final/
        or audit/ specifically — deleting an unrelated subdir of run_dir
        doesn't touch run_dir's own data, and colocating the staged output
        with the run is a plausible deliberate choice."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = run_dir / "hf_staging"
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert staged["corpus_file"] == corpus_name
        assert (run_dir / "final" / corpus_name).exists()

    def test_staging_dir_pointing_at_a_sibling_run_is_rejected(self, tmp_path):
        """A --staging-dir that typo'd onto a DIFFERENT run dir (a very
        plausible mistake — both live under outputs/*/runs/) must be refused
        rather than wiped, or the sibling run's corpus/manifest are gone."""
        run_a, corpus_a = make_run_dir(tmp_path, run_name="run-a")
        run_b, _ = make_run_dir(tmp_path, run_name="run-b")
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_a], corpus_a, run_b, "sdf")
        assert (run_b / "final" / "sdf_corpus.jsonl").exists()
        assert (run_b / "run_manifest.json").exists()

    def test_populated_foreign_staging_dir_is_rejected(self, tmp_path):
        """A non-empty --staging-dir with no marker and no legacy staging
        shape wasn't created by this script — refuse rather than wipe it."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
        (foreign / "subdir").mkdir()
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, foreign, "sdf")
        assert (foreign / "notes.txt").exists()
        assert (foreign / "subdir").exists()

    def test_empty_pre_existing_staging_dir_is_accepted(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staging_dir.mkdir()
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert (dataset_dir / corpus_name).exists()

    def test_tool_created_staging_dir_carries_the_marker_and_is_reusable(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        _stage(run_dir, corpus_name, staging_dir)
        assert (staging_dir / publish_hf.STAGING_MARKER).exists()
        # a second call must not be refused just because it now owns the dir
        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert (dataset_dir / corpus_name).exists()

    def test_legacy_staging_dir_without_the_marker_is_still_reused(self, tmp_path):
        """A staging dir created by a pre-marker version of this script has no
        STAGING_MARKER, but its shape (README.md + sdf/ and/or dad/ dirs) is
        recognizable — don't force a manual delete on every existing local
        staging dir just because the script grew a marker file."""
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_dir = tmp_path / "staged"
        staging_dir.mkdir()
        (staging_dir / "README.md").write_text("old card", encoding="utf-8")
        legacy_sdf = staging_dir / "sdf"
        legacy_sdf.mkdir()
        (legacy_sdf / "sdf_corpus.jsonl").write_text("stale", encoding="utf-8")

        staged, dataset_dir = _stage(run_dir, corpus_name, staging_dir)
        assert (dataset_dir / corpus_name).exists()
        # the stale legacy content is gone, replaced by this run's corpus
        assert (dataset_dir / corpus_name).read_text(encoding="utf-8") != "stale"

    def test_staging_dir_that_is_a_file_is_rejected(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path)
        staging_file = tmp_path / "staged"
        staging_file.write_text("not a directory", encoding="utf-8")
        with pytest.raises(SystemExit):
            publish_hf.stage_run([run_dir], corpus_name, staging_file, "sdf")
        assert staging_file.is_file()
        assert staging_file.read_text(encoding="utf-8") == "not a directory"


class TestFlattenDadCorpus:
    def test_staged_dad_records_are_flat_columns(self, tmp_path):
        """The published copy shows one column per field (example_gid,
        user_prompt, assistant_response) — no messages array, no role keys.

        Rows are looked up BY GID rather than by position: this test is about
        the shape of a record, and the ordering tests own row order. Asserting
        both here would mean a deliberate reordering change failed as a shape
        regression."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        assert staged["n_docs"] == 2
        lines = (dataset_dir / corpus_name).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == 2
        by_gid = {rec["example_gid"]: rec for rec in records}
        assert set(by_gid) == {"E-0000", "E-0001"}
        for i in range(2):
            rec = by_gid[f"E-{i:04d}"]
            # No language or variables column here: this run has no
            # step3/rewrites.jsonl to join, so both are omitted rather than
            # nulled down every row.
            assert set(rec) == {"example_gid", "user_prompt", "assistant_response"}
            assert rec["user_prompt"] == f"user prompt {i}"
            assert rec["assistant_response"] == f"assistant response {i}"

    def test_published_rows_carry_no_run_column(self, tmp_path):
        """Row-to-run attribution is the repo's job (git grep on the globally
        unique example_gid), not a repeated run_id on every row."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        raw = (dataset_dir / corpus_name).read_text(encoding="utf-8")
        assert "source_run" not in raw
        assert MANIFEST["run_id"] not in raw

    def test_local_training_corpus_is_left_untouched(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        src = run_dir / "final" / corpus_name
        before = src.read_text(encoding="utf-8")
        _stage(run_dir, corpus_name, tmp_path / "staged")
        assert src.read_text(encoding="utf-8") == before
        assert "messages" in before  # the SFT chat shape stays on disk

    def test_an_sdf_corpus_with_no_language_field_keeps_row_order(self, tmp_path):
        """Records with no `language` key are unmeasurable, so the row-order
        pass declines and rows stay in the run's own order — which is what
        every SDF run predating layer5_score's language field gets. Columns
        are still reordered per SDF_COLUMN_ORDER regardless of that (doc_id
        moves from this fixture's leading position to trailing), so this
        compares parsed records rather than raw bytes."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="sdf", docs=2,
                                            audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        _, published = _published(dataset_dir, corpus_name)
        source = [json.loads(l) for l in (run_dir / "final" / corpus_name)
                 .read_text(encoding="utf-8").splitlines()]
        assert published == source  # same order, same values
        assert staged["languages"] is None

    def test_record_without_assistant_message_aborts(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        bad = {"example_gid": "E-9999",
               "messages": [{"role": "user", "content": "only a user turn"}]}
        (run_dir / "final" / corpus_name).write_text(
            json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="E-9999"):
            _stage(run_dir, corpus_name, tmp_path / "staged")

    def test_non_ascii_content_is_not_escaped(self, tmp_path):
        """The corpus is multilingual; published rows must keep native script
        readable, not \\uXXXX escapes."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        rec = {"example_gid": "E-0001", "messages": [
            {"role": "user", "content": "鶏の福祉について"},
            {"role": "assistant", "content": "丁寧に考えます"},
        ]}
        (run_dir / "final" / corpus_name).write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        raw = (dataset_dir / corpus_name).read_text(encoding="utf-8")
        assert "鶏の福祉について" in raw


class TestReorderSdfCorpus:
    """Direct tests of reorder_sdf_corpus, the column-order counterpart of
    flatten_dad_corpus for the SDF config — SDF is no longer staged via a
    verbatim shutil.copy2. make_run_dir's own SDF fixture writes doc_id
    first, which SDF_COLUMN_ORDER no longer matches (doc_id trails, as pure
    lineage/join bookkeeping), so these tests build records directly to
    exercise the actual reordering."""

    FULL_RECORD = {
        "doc_id": "matrix_000042", "type_id": "policy_memo",
        "type_name": "policy_memo", "language": "English",
        "register": "clinical, dispassionate",
        "variables": {"document_type": "policy_memo",
                      "tone": "clinical, dispassionate"},
        "description": "A policy memo weighing procurement options.",
        "content": "This memo evaluates the welfare implications of...",
        "scores": {"alignment": 8, "realism": 7, "spec_conformance": 9,
                  "notes": "solid"},
    }

    def test_full_schema_record_is_reordered_content_first(self, tmp_path):
        src = tmp_path / "final.jsonl"
        dst = tmp_path / "published.jsonl"
        src.write_text(json.dumps(self.FULL_RECORD) + "\n", encoding="utf-8")

        n = publish_hf.reorder_sdf_corpus(src, dst)

        assert n == 1
        published = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
        assert list(published) == publish_hf.SDF_COLUMN_ORDER
        assert published == self.FULL_RECORD  # same values, only key order moved

    def test_legacy_fields_are_kept_and_appended_after_known_columns(self, tmp_path):
        """Four committed SDF runs predate variables/description/type_name and
        carry subtype_id/role instead. Reordering must not drop them —
        nothing here is a column SELECTION, only an order."""
        legacy = {"doc_id": "d0", "subtype_id": "s1", "type_id": "t1",
                 "role": "advisor", "register": "warm", "language": "English",
                 "content": "hello", "scores": {"alignment": 8}}
        src = tmp_path / "final.jsonl"
        dst = tmp_path / "published.jsonl"
        src.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        publish_hf.reorder_sdf_corpus(src, dst)

        published = json.loads(dst.read_text(encoding="utf-8").splitlines()[0])
        assert list(published) == ["content", "language", "type_id", "register",
                                   "scores", "doc_id", "subtype_id", "role"]
        assert published == legacy

    def test_never_writes_to_src(self, tmp_path):
        src = tmp_path / "final.jsonl"
        dst = tmp_path / "published.jsonl"
        src.write_text(json.dumps(self.FULL_RECORD) + "\n", encoding="utf-8")
        before = src.read_text(encoding="utf-8")

        publish_hf.reorder_sdf_corpus(src, dst)

        assert src.read_text(encoding="utf-8") == before

    def test_a_malformed_line_passes_through_unchanged(self, tmp_path):
        src = tmp_path / "final.jsonl"
        dst = tmp_path / "published.jsonl"
        src.write_text(json.dumps(self.FULL_RECORD) + "\n{not json\n",
                       encoding="utf-8")

        n = publish_hf.reorder_sdf_corpus(src, dst)

        lines = dst.read_text(encoding="utf-8").splitlines()
        assert n == 2
        assert json.loads(lines[0]) == self.FULL_RECORD  # reordered, values kept
        assert lines[1] == "{not json"

    def test_non_ascii_content_survives_the_rekey(self, tmp_path):
        rec = {"doc_id": "d0", "content": "鶏の福祉について"}
        src = tmp_path / "final.jsonl"
        dst = tmp_path / "published.jsonl"
        src.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")

        publish_hf.reorder_sdf_corpus(src, dst)

        raw = dst.read_text(encoding="utf-8")
        assert "鶏の福祉について" in raw
        assert "\\u" not in raw


def _published(dataset_dir, corpus_name):
    """The staged corpus as (raw lines, parsed records)."""
    lines = (dataset_dir / corpus_name).read_text(encoding="utf-8").splitlines()
    return lines, [json.loads(line) for line in lines]


class TestEnglishFirstOrdering:
    """The Hub viewer opens on whatever row is first in the file, so English
    rows lead. Only the STAGED copy is touched — never the run's own final/
    corpus, which several evals stride-sample and which layer5's near-dup cull
    already ordered."""

    def test_english_documents_lead_the_published_sdf_corpus(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=4, audit_files=[], include_html=False,
            languages=["English", "Spanish"])
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["language"] for r in records] == [
            "English", "English", "Spanish", "Spanish"]
        assert staged["languages"] == {"English": 2, "Spanish": 2}

    def test_english_documents_keep_their_relative_order_at_the_front(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=6, audit_files=[], include_html=False,
            languages=["English", "Spanish"])
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["doc_id"] for r in records[:3]] == ["d0", "d2", "d4"]

    def test_non_english_documents_keep_their_relative_order_behind_the_english_block(
            self, tmp_path):
        """The partition is STABLE and binary — one boundary, nothing else
        moved. A secondary sort key (by language name, by length) would group
        the tail into monolingual runs too, which is worse for anyone streaming
        the corpus without shuffling. This test is what stops that being added
        quietly."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=6, audit_files=[], include_html=False,
            languages=["English", "Spanish", "Japanese"])
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        # d1/d4 Spanish and d2/d5 Japanese stay interleaved as the run wrote
        # them; grouping by language would give d1, d4, d2, d5.
        assert [r["doc_id"] for r in records[2:]] == ["d1", "d2", "d4", "d5"]

    def test_the_published_rows_are_the_run_s_own_records_reordered_not_changed(
            self, tmp_path):
        """The one assertion that catches every dropped/altered-value hazard
        at once: the published file must be a value-level PERMUTATION of the
        run's records, so nothing can have added, dropped, or changed a
        field's value on the way through. Column order is deliberately
        allowed to differ — reorder_sdf_corpus re-keys every row — so this
        compares parsed records, not raw line text (TestReorderSdfCorpus pins
        the exact column order separately)."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=6, audit_files=[], include_html=False,
            languages=["English", "Spanish", "Japanese"])
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, published = _published(dataset_dir, corpus_name)
        source = [json.loads(l) for l in (run_dir / "final" / corpus_name)
                 .read_text(encoding="utf-8").splitlines()]
        sort_key = lambda rec: sorted(rec.items())
        assert sorted(published, key=sort_key) == sorted(source, key=sort_key)
        assert [r["doc_id"] for r in published] != [r["doc_id"] for r in source]
        # and it really did reorder rows (English-first)

    def test_native_script_survives_the_reorder_unescaped(self, tmp_path):
        """json.dumps defaults to ensure_ascii=True, which would turn most of
        this corpus into \\uXXXX. Sorting raw lines is what makes that
        impossible."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=2, audit_files=[], include_html=False,
            languages=["English", "Japanese"])
        src = run_dir / "final" / corpus_name
        src.write_text("\n".join([
            json.dumps({"doc_id": "d0", "language": "Japanese",
                        "content": "鶏の福祉について"}, ensure_ascii=False),
            json.dumps({"doc_id": "d1", "language": "English",
                        "content": "about hen welfare"}),
        ]) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        raw = (dataset_dir / corpus_name).read_text(encoding="utf-8")
        assert "鶏の福祉について" in raw
        assert "\\u" not in raw

    def test_the_legacy_en_language_code_counts_as_english(self, tmp_path):
        """Four committed SDF runs write the bare ISO code instead of the full
        name. evals/audit_sdf.py already accepts both spellings."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=4, audit_files=[], include_html=False,
            languages=["Spanish", "en"])
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["language"] for r in records] == ["en", "en", "Spanish", "Spanish"]

    def test_a_corpus_that_is_already_all_english_keeps_row_order(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=3, audit_files=[], include_html=False,
            languages=["English"])
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, published = _published(dataset_dir, corpus_name)
        source = [json.loads(l) for l in (run_dir / "final" / corpus_name)
                 .read_text(encoding="utf-8").splitlines()]
        assert published == source  # same order, same values

    def test_a_malformed_corpus_line_leaves_row_order_untouched_instead_of_aborting(
            self, tmp_path):
        """Row order is cosmetic, so it degrades rather than killing a publish:
        an old run whose language cannot be read is published in the order it
        was written. The malformed line itself passes through unchanged;
        valid lines still go through the column reorder (moving doc_id from
        this fixture's leading position to trailing), which is why this
        compares parsed records rather than raw bytes."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=2, audit_files=[], include_html=False,
            languages=["Spanish", "English"])
        src = run_dir / "final" / corpus_name
        src.write_text(src.read_text(encoding="utf-8") + "{not json\n",
                       encoding="utf-8")
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        assert staged["languages"] is None
        published_lines = (dataset_dir / corpus_name).read_text(
            encoding="utf-8").splitlines()
        source_lines = src.read_text(encoding="utf-8").splitlines()
        assert published_lines[-1] == "{not json"
        assert [json.loads(l) for l in published_lines[:-1]] == \
            [json.loads(l) for l in source_lines[:-1]]

    def test_a_record_whose_language_cannot_be_read_lands_behind_the_english_block(
            self, tmp_path):
        """The front of the file is a promise, so an unreadable row sorts
        behind rather than being defaulted to English the way audit_sdf's
        wider net does."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=2, audit_files=[], include_html=False,
            languages=["English"])
        src = run_dir / "final" / corpus_name
        src.write_text("\n".join([
            json.dumps({"doc_id": "d0"}),                              # no language
            json.dumps({"doc_id": "d1", "language": "English"}),
        ]) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["doc_id"] for r in records] == ["d1", "d0"]

    def test_the_run_s_own_final_corpus_is_never_reordered(self, tmp_path):
        """Five evals stride-sample this file and layer5's greedy near-dup cull
        already ordered it. Reordering it here would silently change which
        documents those samples pick."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=4, audit_files=[], include_html=False,
            languages=["Spanish", "English"])
        src = run_dir / "final" / corpus_name
        before = src.read_text(encoding="utf-8")
        _stage(run_dir, corpus_name, tmp_path / "staged")
        assert src.read_text(encoding="utf-8") == before

    def test_staging_the_same_run_twice_produces_the_same_bytes(self, tmp_path):
        """A stable partition over deterministic input is deterministic, so
        republishing an unchanged run makes no spurious Hub commit."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="sdf", docs=6, audit_files=[], include_html=False,
            languages=["English", "Spanish", "Japanese"])
        _, first = _stage(run_dir, corpus_name, tmp_path / "staged_a")
        _, second = _stage(run_dir, corpus_name, tmp_path / "staged_b")
        assert (first / corpus_name).read_text(encoding="utf-8") == \
            (second / corpus_name).read_text(encoding="utf-8")


class TestDadLanguageJoin:
    """DAD final records carry no language, so it is joined off
    step3/rewrites.jsonl — one hop, on the example_gid the published row keeps.
    """

    def test_dad_rows_are_ordered_by_the_language_dealt_to_their_scenario(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=4, audit_files=[], include_html=False,
            languages=["English", "Spanish"])
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["example_gid"] for r in records] == [
            "E-0000", "E-0002", "E-0001", "E-0003"]
        assert staged["languages"] == {"English": 2, "Spanish": 2}

    def test_the_unmarked_cultural_setting_slice_is_treated_as_english(self, tmp_path):
        """~65% of a DAD run is dealt "no particular location or culture" and
        stored as null. Those prompts ARE English, so a null card must lead,
        not sort behind."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            languages=["Japanese", "English"])
        rewrites = json.loads(
            (run_dir / "step3" / "rewrites.jsonl").read_text(
                encoding="utf-8").splitlines()[1])
        assert rewrites["scenario_cards"]["cultural_setting"] is None

        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        _, records = _published(dataset_dir, corpus_name)
        assert [r["example_gid"] for r in records] == ["E-0001", "E-0000"]

    def test_the_join_reads_the_pre_rename_annotation_key_too(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            languages=["Spanish", "English"])
        (run_dir / "step3" / "rewrites.jsonl").write_text("\n".join([
            json.dumps({"example_gid": "E-0000",
                        "annotation": {"cultural_setting": marked_setting("Spanish")}}),
            json.dumps({"example_gid": "E-0001",
                        "annotation": {"cultural_setting": None}}),
        ]) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["example_gid"] for r in records] == ["E-0001", "E-0000"]

    def test_a_run_without_step3_rewrites_publishes_in_the_order_it_was_written(
            self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=3,
                                            audit_files=[], include_html=False)
        assert not (run_dir / "step3").exists()
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        assert staged["languages"] is None
        _, records = _published(dataset_dir, corpus_name)
        assert [r["example_gid"] for r in records] == ["E-0000", "E-0001", "E-0002"]

    def test_a_run_whose_cards_carry_no_cultural_setting_publishes_unreordered(
            self, tmp_path):
        """archetype10 is the committed example: hand-seeded, so it bypassed
        the matrix and records no setting at all. Its rows are in fact English,
        but declaring that from absent evidence is what code_state() refuses to
        do for a dirty tree, and the same reasoning applies here."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            languages=["English", "English"])
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        assert staged["languages"] is None
        _, records = _published(dataset_dir, corpus_name)
        assert [r["example_gid"] for r in records] == ["E-0000", "E-0001"]

    def test_published_dad_rows_carry_the_language_they_were_dealt(self, tmp_path):
        """SDF rows already publish a language column; this closes the gap so
        a reader can rebuild a balanced subset from a corpus that is now
        ordered English-first."""
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            languages=["English", "Mandarin Chinese"])
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert set(records[0]) == {"example_gid", "language", "user_prompt",
                                   "assistant_response", "variables"}
        assert {r["example_gid"]: r["language"] for r in records} == {
            "E-0000": "English", "E-0001": "Mandarin Chinese"}

    def test_a_run_with_no_language_data_publishes_no_language_column(self, tmp_path):
        """A column that is null on every row reads as broken; omitting it says
        the same thing honestly."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert all("language" not in r for r in records)

    def test_an_unjoined_row_carries_a_null_language_not_a_guess(self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=3, audit_files=[], include_html=False,
            languages=["Spanish", "English", "English"])
        rewrites = (run_dir / "step3" / "rewrites.jsonl")
        kept = [line for line in rewrites.read_text(encoding="utf-8").splitlines()
                if "E-0002" not in line]
        rewrites.write_text("\n".join(kept) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert {r["example_gid"]: r["language"] for r in records} == {
            "E-0000": "Spanish", "E-0001": "English", "E-0002": None}

    def test_a_row_whose_example_gid_does_not_join_lands_behind_the_english_block(
            self, tmp_path):
        run_dir, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=3, audit_files=[], include_html=False,
            languages=["Spanish", "English", "English"])
        rewrites = (run_dir / "step3" / "rewrites.jsonl")
        kept = [line for line in rewrites.read_text(encoding="utf-8").splitlines()
                if "E-0002" not in line]
        rewrites.write_text("\n".join(kept) + "\n", encoding="utf-8")
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["example_gid"] for r in records] == ["E-0001", "E-0000", "E-0002"]


def write_dealt_cards(run_dir, cards_by_gid):
    """Overwrite a DAD run's step3/rewrites.jsonl with the given dealt cards.

    make_run_dir's own step3 file carries only cultural_setting, which is all
    the language join reads; the variables column publishes the whole hand, so
    these tests deal a fuller one.
    """
    step3 = run_dir / "step3"
    step3.mkdir(parents=True, exist_ok=True)
    (step3 / "rewrites.jsonl").write_text("\n".join(
        json.dumps({"record_id": f"r{i}", "example_gid": gid,
                    "scenario_cards": cards})
        for i, (gid, cards) in enumerate(cards_by_gid.items())) + "\n",
        encoding="utf-8")


class TestDadVariablesColumn:
    """The dealt cards reach the published row as one nested column, the DAD
    counterpart of the SDF corpus's own `variables`. Same join as the language
    column, one file read, so the two cannot describe different records."""

    DEALT = {"domain": ["procurement"], "user_goal": ["drafting"],
             "taxa_category": "insect-at-scale", "user_attitude": "unaware",
             "cultural_setting": None, "conflict": "",
             # legacy write-up fields ride along on every committed run
             "claims": [], "dilemma_anatomy": {}, "moral_patients": ""}

    def test_published_rows_carry_the_cards_their_scenario_was_dealt(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        write_dealt_cards(run_dir, {"E-0000": self.DEALT, "E-0001": self.DEALT})

        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        for rec in records:
            variables = rec["variables"]
            assert variables["domain"] == ["procurement"]
            assert variables["taxa_category"] == "insect-at-scale"
            # every field present whether or not it was dealt, and "not dealt"
            # has one spelling whether the card was absent, None, or ""
            assert set(variables) == set(DEALT_CARD_FIELDS)
            assert variables["cultural_setting"] is None
            assert variables["conflict"] is None
            assert variables["length_class"] is None
            # the write-up fields are not dealt cards and are not published
            assert not {"claims", "dilemma_anatomy", "moral_patients"} & set(variables)

    def test_example_gid_trails_every_column_including_variables(self, tmp_path):
        """The two text columns are what a visitor came to read, so they
        lead; example_gid trails everything — including the 19-field
        variables struct — as pure lineage/join bookkeeping, not something a
        reader needs in front of the content."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=1,
                                            audit_files=[], include_html=False)
        # a marked setting, so the language column is in play too
        write_dealt_cards(run_dir, {"E-0000": {**self.DEALT,
                                               "cultural_setting": marked_setting("Spanish")}})

        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert list(records[0]) == ["user_prompt", "assistant_response",
                                    "language", "variables", "example_gid"]
        assert records[0]["language"] == "Spanish"

    def test_a_run_with_no_dealt_cards_publishes_no_variables_column(self, tmp_path):
        """Same rule as the language column: a column null on every row reads
        as broken, and omitting it says the same thing honestly."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert all("variables" not in rec for rec in records)

    def test_an_unjoined_row_carries_a_null_cell_not_a_guess(self, tmp_path):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        write_dealt_cards(run_dir, {"E-0000": self.DEALT})

        _, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        by_gid = {rec["example_gid"]: rec for rec in records}
        assert by_gid["E-0000"]["variables"]["taxa_category"] == "insect-at-scale"
        assert by_gid["E-0001"]["variables"] is None

    def test_a_run_with_cards_but_no_language_still_publishes_them(self, tmp_path):
        """The hand-seeded archetype10 run deals no cultural_setting, so the
        language gate declines for it — but it has cards worth publishing, and
        the two columns are gated separately for exactly that reason."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        write_dealt_cards(run_dir, {"E-0000": self.DEALT, "E-0001": self.DEALT})

        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert all("language" not in rec for rec in records)
        assert staged["languages"] is None          # nothing to order by
        assert all(rec["variables"]["taxa_category"] == "insect-at-scale"
                   for rec in records)


def _dad_manifest(run_id, backend="api", dirty=None, dirty_files=None):
    """A DAD run manifest. dirty=None omits the git_dirty fields entirely (a
    manifest predating them); pass dirty=True/False to set them, shaped as
    shared.utils records them."""
    m = {"run_id": run_id, "label": run_id.split("_", 2)[-1],
         "git_commit": "abc1234", "model": "claude-sonnet-5",
         "config": {"backend": backend,
                    "dad": {"constitution_rewrite_model": "claude-opus-5"}}}
    if dirty is not None:
        m["git_dirty"] = dirty
        m["git_dirty_files"] = list(dirty_files or [])
    return m


class TestCombinedPublish:
    def _two_runs(self, tmp_path, second_audit=False):
        run_a, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-28_17-32_pareto200",
            manifest=_dad_manifest("2026-07-28_17-32_pareto200"))
        run_b, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=3,
            audit_files=["diversity_report.json"] if second_audit else [],
            include_html=False,
            run_name="2026-07-29_23-58_archetype1000",
            manifest=_dad_manifest("2026-07-29_23-58_archetype1000"))
        return [run_a, run_b], corpus_name

    def test_corpora_without_language_data_concatenate_in_input_order(self, tmp_path):
        """Every run's rows land in the combined corpus, in --input order. The
        rows carry no run column — example_gid is what identifies them, and
        _two_runs gives both runs the same gids on purpose (each run dir is
        built independently, exactly as separate real runs would be) so the
        assertion can't accidentally lean on distinct ids.

        No run here carries language data, so the ordering pass declines and
        raw --input order is what ships. The partitioned contract is pinned
        separately by
        test_a_combined_corpus_is_partitioned_across_all_runs_not_within_each."""
        run_dirs, corpus_name = self._two_runs(tmp_path)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")

        assert staged["n_docs"] == 5
        assert [r["n_docs"] for r in staged["runs"]] == [2, 3]
        records = [json.loads(line) for line in
                   (dataset_dir / corpus_name).read_text(encoding="utf-8").splitlines()]
        assert len(records) == 5
        assert "source_run" not in records[0]
        # run_a contributed 2 rows, then run_b's 3 — docs=2 and docs=3 make the
        # user_prompt sequence restart, which is what pins the concatenation order.
        assert [r["user_prompt"] for r in records] == [
            "user prompt 0", "user prompt 1",
            "user prompt 0", "user prompt 1", "user prompt 2"]

    def _two_runs_with_languages(self, tmp_path):
        """Two runs, each dealt the SAME [English, Spanish] pattern — so a
        per-run partition and a global one give visibly different files."""
        run_a, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-28_17-32_pareto200", languages=["English", "Spanish"],
            manifest=_dad_manifest("2026-07-28_17-32_pareto200"))
        run_b, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-29_23-58_archetype1000", languages=["English", "Spanish"],
            manifest=_dad_manifest("2026-07-29_23-58_archetype1000"))
        return [run_a, run_b], corpus_name

    def test_a_combined_corpus_is_partitioned_across_all_runs_not_within_each(
            self, tmp_path):
        """The partition runs over the WHOLE published file, after the run
        loop. Partitioning per run would only put English first if run_dirs[0]
        happened to hold enough English rows — on the real five-run input order
        the viewer's first screen would still turn non-English part way down."""
        run_dirs, corpus_name = self._two_runs_with_languages(tmp_path)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        # Global: both runs' English rows, then both runs' Spanish rows.
        # Per-run would give A-en, A-es, B-en, B-es.
        assert [r["user_prompt"] for r in records] == [
            "user prompt 0", "user prompt 0", "user prompt 1", "user prompt 1"]
        assert staged["languages"] == {"English": 2, "Spanish": 2}

    def test_each_run_s_rows_keep_their_input_order_inside_each_language_block(
            self, tmp_path):
        """--input order still decides the sequence within a block; only the
        language boundary is new."""
        run_a, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=3, audit_files=[], include_html=False,
            run_name="2026-07-28_17-32_pareto200",
            languages=["English", "English", "Spanish"],
            manifest=_dad_manifest("2026-07-28_17-32_pareto200"))
        run_b, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-29_23-58_archetype1000",
            languages=["Spanish", "English"], gid_start=100,
            manifest=_dad_manifest("2026-07-29_23-58_archetype1000"))
        _, dataset_dir = _stage([run_a, run_b], corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert [r["user_prompt"] for r in records] == [
            # run A's two English rows, then run B's one, in --input order
            "user prompt 0", "user prompt 1", "user prompt 1",
            # then the Spanish rows, likewise
            "user prompt 2", "user prompt 0"]

    def test_duplicate_example_gids_across_runs_do_not_break_the_language_join(
            self, tmp_path):
        """_two_runs gives both runs the same gids on purpose. A repeated gid
        means byte-identical content — the ids are content-keyed via
        dad_pipeline/id_registry.py — so whichever run wins the dict merge
        carries the same answer, and the merge must not raise or drop rows."""
        run_dirs, corpus_name = self._two_runs_with_languages(tmp_path)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")

        _, records = _published(dataset_dir, corpus_name)
        assert len(records) == 4
        assert [r["example_gid"] for r in records] == [
            "E-0000", "E-0000", "E-0001", "E-0001"]
        assert staged["n_docs"] == 4

    def test_manifests_and_audits_are_run_scoped(self, tmp_path):
        """Several runs in one dataset dir must not collide on filenames:
        manifests land under manifests/<run_id>.json and audit files under
        audit/<run_id>/.

        These per-run manifests ARE the combined corpus's provenance record —
        the rows carry no run column, and no card is generated to tabulate
        them — so each run keeping its own file is what lets a reader tell
        which runs, on which commits, went into one corpus."""
        run_dirs, corpus_name = self._two_runs(tmp_path, second_audit=True)
        staged, dataset_dir = _stage(run_dirs, corpus_name, tmp_path / "staged")

        assert not (dataset_dir / "run_manifest.json").exists()
        assert (dataset_dir / "manifests" / "2026-07-28_17-32_pareto200.json").exists()
        assert (dataset_dir / "manifests" / "2026-07-29_23-58_archetype1000.json").exists()
        assert staged["audit_files"] == [
            "2026-07-29_23-58_archetype1000/diversity_report.json"]
        assert (dataset_dir / "audit" / "2026-07-29_23-58_archetype1000"
                / "diversity_report.json").exists()

    def test_each_staged_manifest_keeps_its_own_runs_code_state(self, tmp_path):
        """Mixed states stay named per run, not collapsed — a combined corpus
        is only as reproducible as its least-reproducible run."""
        run_a, corpus_name = make_run_dir(
            tmp_path, pipeline="dad", docs=2, audit_files=[], include_html=False,
            run_name="2026-07-28_17-32_pareto200",
            manifest=_dad_manifest("2026-07-28_17-32_pareto200",
                                   dirty=True, dirty_files=["config.yaml"]))
        run_b, _ = make_run_dir(
            tmp_path, pipeline="dad", docs=3, audit_files=[], include_html=False,
            run_name="2026-07-29_23-58_archetype1000",
            manifest=_dad_manifest("2026-07-29_23-58_archetype1000", dirty=False))
        _, dataset_dir = _stage([run_a, run_b], corpus_name, tmp_path / "staged")

        manifests = dataset_dir / "manifests"
        dirty = json.loads((manifests / "2026-07-28_17-32_pareto200.json").read_text())
        clean = json.loads((manifests / "2026-07-29_23-58_archetype1000.json").read_text())
        assert dirty["git_dirty"] is True
        assert dirty["git_dirty_files"] == ["config.yaml"]
        assert clean["git_dirty"] is False

    def test_single_run_layout_is_unchanged(self, tmp_path):
        """One --input keeps the original shape: top-level run_manifest.json,
        flat audit/, no manifests/ dir."""
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", docs=2,
                                            audit_files=[], include_html=False)
        staged, dataset_dir = _stage(run_dir, corpus_name, tmp_path / "staged")
        assert staged["manifest_file"] == "run_manifest.json"
        assert (dataset_dir / "run_manifest.json").exists()
        assert not (dataset_dir / "manifests").exists()


class TestHubApiWrappers:
    def test_create_tag_passes_exist_ok(self, monkeypatch):
        """Regression: a retried publish with the same --tag (e.g. after
        fixing a typo'd --input, the exact retry the staging-dir wipe logic
        is designed to support) must not die on a "tag already exists" error
        after the corpus has already been re-uploaded."""
        calls = []

        class FakeHfApi:
            def create_tag(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        publish_hf._create_tag("sentientfutures/sdf-corpus", "v1")
        assert calls == [{
            "repo_id": "sentientfutures/sdf-corpus", "tag": "v1",
            "repo_type": "dataset", "exist_ok": True,
        }]

    def test_upload_folder_forwards_caller_scoped_delete_patterns(self, monkeypatch):
        """Republishing a run must clear that run's own audit/ on the Hub, so
        a file only an EARLIER run produced can't linger. The pattern comes
        from the caller because it must be scoped per pipeline — see
        test_delete_patterns_are_scoped_to_the_published_pipeline."""
        calls = []

        class FakeHfApi:
            def upload_folder(self, **kwargs):
                calls.append(kwargs)
                return "fake-commit"

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        result = publish_hf._upload_folder(
            "/tmp/staged", "sentientfutures/x", "msg",
            ["sdf/audit/*", "sdf/card_meta.json"])
        assert result == "fake-commit"
        # forwarded verbatim — the wrapper adds nothing of its own
        assert calls[0]["delete_patterns"] == ["sdf/audit/*", "sdf/card_meta.json"]
        assert calls[0]["repo_id"] == "sentientfutures/x"
        assert calls[0]["folder_path"] == "/tmp/staged"

    def test_upload_folder_never_uploads_the_staging_marker(self, monkeypatch):
        """huggingface_hub only ignores .git*/.cache/huggingface by default,
        so without an explicit ignore_patterns the local ownership marker
        would ship to the Hub as dataset content."""
        calls = []

        class FakeHfApi:
            def upload_folder(self, **kwargs):
                calls.append(kwargs)
                return "fake-commit"

        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        publish_hf._upload_folder("/tmp/staged", "sentientfutures/x", "msg", [])
        assert calls[0]["ignore_patterns"] == [publish_hf.STAGING_MARKER]


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["publish_hf.py", *args])
    publish_hf.main()


class TestMainEndToEnd:
    def test_dry_run_makes_no_hub_calls(self, tmp_path, monkeypatch, stub_hf, capsys):
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/sdf-corpus", "--dry-run")
        out = capsys.readouterr().out
        assert "no Hub API calls made" in out

    def test_dry_run_refuses_a_foreign_staging_dir(self, tmp_path, monkeypatch, stub_hf):
        """The ownership check runs inside stage_run, which --dry-run calls
        before its early return — so --dry-run must refuse a foreign
        --staging-dir too, deleting nothing."""
        run_dir, _ = make_run_dir(tmp_path)
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
        stub_hf(raise_on_call=True)
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, "--input", str(run_dir),
                      "--repo-id", "sentientfutures/sdf-corpus", "--dry-run",
                      "--staging-dir", str(foreign))
        assert (foreign / "notes.txt").exists()

    def test_dry_run_without_staging_dir_leaves_files_on_disk(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """Regression: --dry-run's default staging dir used to live inside a
        tempfile.TemporaryDirectory() that self-deleted the instant main()
        returned, so the printed "Staged at <path>" was already gone by the
        time a human went to look — defeating the entire point of --dry-run.
        """
        run_dir, corpus_name = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/sdf-corpus", "--dry-run")
        out = capsys.readouterr().out

        match = re.search(r"Staged at (\S+) \(left on disk", out)
        assert match, f"expected a 'Staged at <path>' message, got:\n{out}"
        staged_path = Path(match.group(1))
        try:
            assert staged_path.is_dir()
            assert (staged_path / "sdf" / corpus_name).exists()
        finally:
            shutil.rmtree(staged_path.parent, ignore_errors=True)

    def test_dry_run_stages_no_card_and_prints_none(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """--dry-run used to build and print a card unconditionally, as an
        operator preview. There is no card to preview any more, and printing
        one would be worse than printing nothing: it would show the operator
        prose that is not on the Hub and never will be."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--dry-run", "--staging-dir", str(staging_dir))

        assert not (staging_dir / "README.md").exists()
        out = capsys.readouterr().out
        assert "pretty_name:" not in out and "config_name:" not in out
        assert "hand-written and edited on the Hub" in out

    def test_publish_calls_hf_api_with_expected_args(self, tmp_path, monkeypatch, stub_hf):
        run_dir, corpus_name = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir),
                  "--repo-id", "sentientfutures/awmtd",
                  "--tag", "sdf-v1-fullscale-500-opus5",
                  "--staging-dir", str(staging_dir))

        by_fn = {c["fn"]: c for c in calls}
        assert by_fn["create_repo"]["repo_id"] == "sentientfutures/awmtd"
        assert by_fn["upload_folder"]["repo_id"] == "sentientfutures/awmtd"
        assert by_fn["upload_folder"]["folder_path"] == str(staging_dir)
        assert by_fn["create_tag"]["tag"] == "sdf-v1-fullscale-500-opus5"

        uploaded = {str(p.relative_to(staging_dir))
                    for p in staging_dir.rglob("*") if p.is_file()}
        assert f"sdf/{corpus_name}" in uploaded
        assert "sdf/run_manifest.json" in uploaded
        assert not any("report_content.json" in u for u in uploaded)

    def test_publish_never_stages_a_card(self, tmp_path, monkeypatch, stub_hf):
        """The contract that replaced card generation, and the only thing
        protecting a card edited on the Hub.

        upload_folder overwrites every path it finds and deletes every path
        matching delete_patterns, so a README.md that is neither staged nor
        matched is one it cannot touch. Both halves are asserted: there is no
        flag or run shape that puts a card back in the staging dir, and no
        delete pattern reaches the repo root."""
        for pipeline in ("sdf", "dad"):
            run_dir, _ = make_run_dir(tmp_path / pipeline, pipeline=pipeline)
            staging_dir = tmp_path / pipeline / "staged"
            calls = stub_hf()
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                      "--staging-dir", str(staging_dir))

            assert not (staging_dir / "README.md").exists()
            upload = next(c for c in calls if c["fn"] == "upload_folder")
            assert all(p.startswith(f"{pipeline}/")
                       for p in upload["delete_patterns"])

    def test_delete_patterns_are_scoped_to_the_published_pipeline(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """THE most dangerous line in the multi-dataset change: a bare
        "audit/*" would delete the SIBLING pipeline's audit files on every
        publish, silently gutting the other dataset in the same repo."""
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad", audit_files=[], include_html=False)
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        # run_manifest.json + manifests/* are both cleared so a publish that
        # switches layout (single-run <-> combined) can't leave the other
        # layout's manifest file(s) behind; upload_folder keeps freshly staged
        # paths, so the staged layout survives its own pattern.
        assert upload["delete_patterns"] == [
            "dad/audit/*", "dad/run_manifest.json", "dad/manifests/*",
            "dad/card_meta.json"]
        # every pattern must stay under this pipeline's own prefix
        assert all(p.startswith("dad/") for p in upload["delete_patterns"])

    def test_legacy_card_meta_is_cleared_and_never_rewritten(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """card_meta.json fed the removed card generator, and `sdf/` still has
        one on the Hub from the last publish that wrote it.

        Nothing stages that path now, so upload_folder's add-wins-over-delete
        rule no longer suppresses the deletion and the next publish clears the
        orphan. Both halves matter: the run below carries a
        report_content.json (the input the sidecar used to be derived from) and
        must still write no sidecar."""
        run_dir, _ = make_run_dir(tmp_path)  # includes report_content.json
        staging_dir = tmp_path / "staged"
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert not (staging_dir / "sdf" / "card_meta.json").exists()
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert "sdf/card_meta.json" in upload["delete_patterns"]

    def test_publish_without_tag_skips_create_tag(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        calls = stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        assert "create_tag" not in [c["fn"] for c in calls]

    def test_dad_run_end_to_end(self, tmp_path, monkeypatch, stub_hf):
        run_dir, corpus_name = make_run_dir(tmp_path, pipeline="dad", audit_files=[],
                                            include_html=False)
        staging_dir = tmp_path / "staged"
        stub_hf()
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))
        assert (staging_dir / "dad" / "dad_corpus.jsonl").exists()

    @pytest.mark.parametrize("flag, value", [
        ("--regenerate-card", None),
        ("--license", "cc0-1.0"),
        ("--pretty-name", "Animal-welfare training dataset"),
    ])
    def test_the_removed_card_flags_are_gone(self, tmp_path, monkeypatch, stub_hf,
                                             flag, value):
        """These three only ever fed the card builder. They must ERROR rather
        than be silently accepted and ignored — --license especially, whose
        default would otherwise read as the published licence while declaring
        nothing at all."""
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        stub_hf(raise_on_call=True)
        extra = [flag] if value is None else [flag, value]
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, *extra, "--input", str(run_dir),
                      "--repo-id", "org/repo", "--dry-run")


class TestUnmergedGuard:
    """The pre-flight provenance gate. It warns and asks rather than refusing:
    the HF write token lives on contributors' laptops, so a hard block would
    push an unmerged publish out of this script — and out of the only place
    that records provenance at all.

    What makes it stick is the Hub COMMIT MESSAGE. It used to be a stamp on
    the generated dataset card; that card is hand-edited on the Hub now and
    this script no longer writes it, so the stamp went where an edit cannot
    reach it. Every test below asserts on the commit message for that reason.
    """

    def _unmerged(self, monkeypatch, **over):
        state = unmerged_state(**over)
        monkeypatch.setattr(publish_hf, "merge_state",
                            lambda commit, fetch=True: dict(state))
        return state

    def test_merged_run_publishes_silently(self, tmp_path, monkeypatch, stub_hf, capsys):
        """The default path must stay quiet — a warning that also fires on
        merged runs is one people learn to type straight past."""
        run_dir, _ = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir))

        err = capsys.readouterr().err
        assert "NOT been merged" not in err
        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert upload["commit_message"] == \
            "Publish sdf: 2026-07-25_15-57_fullscale-500-opus5"

    def test_non_interactive_without_flag_refuses_before_any_hub_call(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """An agent, a pipe, or a CI job has nobody to answer a prompt. It must
        exit naming the flag, and must not have touched the Hub first."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        assert "--allow-unmerged" in str(excinfo.value)

    def test_interactive_yes_proceeds(self, tmp_path, monkeypatch, stub_hf):
        run_dir, _ = make_run_dir(tmp_path)
        calls = stub_hf()
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")
        assert any(c["fn"] == "upload_folder" for c in calls)

    def test_interactive_anything_else_aborts_with_no_hub_call(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """Only the exact word publishes. 'y' is a reflex; 'yes' is a decision."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")

        with pytest.raises(SystemExit):
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")

    def test_allow_unmerged_publishes_and_stamps_everything(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """The durable half of the guard: the warning reaches the operator's
        terminal, and the Hub commit message records it where nobody can edit
        it away afterwards."""
        run_dir, _ = make_run_dir(tmp_path)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        self._unmerged(monkeypatch, branch="declan/wip", commit="deadbee")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir), "--allow-unmerged")

        err = capsys.readouterr().err
        assert "NOT been merged" in err
        assert "declan/wip" in err

        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert "unmerged run(s): 2026-07-25_15-57_fullscale-500-opus5" \
            in upload["commit_message"]

    def test_stamp_survives_a_run_with_no_audit_files(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The record must not depend on what the run happens to have produced.
        It once rode in a sidecar written only when a run had a curated
        title, so a run without report_content.json published unlabelled; the
        commit message is derived from the merge check alone."""
        run_dir, _ = make_run_dir(tmp_path, audit_files=[], include_html=False)
        calls = stub_hf()
        staging_dir = tmp_path / "staged"
        self._unmerged(monkeypatch)

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(staging_dir), "--allow-unmerged")

        upload = next(c for c in calls if c["fn"] == "upload_folder")
        assert "unmerged run(s): 2026-07-25_15-57_fullscale-500-opus5" \
            in upload["commit_message"]

    def test_dry_run_shows_the_warning_and_stamp_without_prompting(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """A preview that hid the warning would be the wrong preview — but
        --dry-run publishes nothing, so there is nothing to confirm.

        Both halves have to show: the terminal warning, and the commit message
        the publish would leave on the Hub. The record used to be previewable
        as part of the card; showing only the warning would preview the half
        that doesn't outlive the terminal."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input",
                            lambda _prompt: pytest.fail("--dry-run must not prompt"))

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--dry-run")

        captured = capsys.readouterr()
        assert "NOT been merged" in captured.err
        assert "Would commit as: Publish sdf: " in captured.out
        assert "unmerged run(s): 2026-07-25_15-57_fullscale-500-opus5 " \
               "(branch declan/wip, commit deadbee)" in captured.out

    def test_dry_run_does_not_contact_the_remote(self, tmp_path, monkeypatch, stub_hf):
        """--dry-run is documented as making zero network calls, and a git fetch
        would break that promise just as surely as a Hub call."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        seen = {}

        def record(commit, fetch=True):
            seen["fetch"] = fetch
            return dict(MERGED_STATE)

        monkeypatch.setattr(publish_hf, "merge_state", record)

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--dry-run")
        assert seen["fetch"] is False

    def test_unknown_provenance_is_treated_as_unmerged(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """An unverifiable claim is not a safe one: a manifest with no git
        commit, or a commit this clone has never seen, must warn rather than
        sail through."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf(raise_on_call=True)
        self._unmerged(monkeypatch, head_merged=True, run_commit_merged=None,
                       notes=["commit deadbee is not in this clone (never pushed?)"])
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

        with pytest.raises(SystemExit):
            _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo")

    def test_unverifiable_is_not_reported_as_definitely_unmerged(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        """Both verdicts block, but they must not read the same. Claiming a run
        is unmerged when the truth is "couldn't tell" teaches people the warning
        is inaccurate, which is how a guardrail loses its authority."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf()
        self._unmerged(monkeypatch, head_merged=True, run_commit_merged=None,
                       notes=["commit deadbee is not in this clone"])

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--allow-unmerged")
        err = capsys.readouterr().err
        assert "could NOT be verified against main" in err
        assert "has NOT been merged" not in err

    def test_notes_reach_the_operator(self, tmp_path, monkeypatch, stub_hf, capsys):
        """merge_state's plain-English reasons are the only explanation of an
        unknown verdict, so they must be printed, not swallowed."""
        run_dir, _ = make_run_dir(tmp_path)
        stub_hf()
        self._unmerged(monkeypatch, head_merged=True, run_commit_merged=None,
                       notes=["this clone has no origin/main reference to compare against"])

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--allow-unmerged")
        assert "no origin/main reference" in capsys.readouterr().err

    def test_stamp_names_the_branch_the_data_was_generated_on(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """A run generated on one branch can be published from another, and the
        commit message must keep the two straight: the RUN entry names the
        branch the data came from (v3 manifests' git_branch), while an unmerged
        checkout is reported separately as the publish branch. Collapsing them
        would let a reader think the corpus was generated by whatever happens
        to be checked out now."""
        run_dir, _ = make_run_dir(
            tmp_path,
            manifest={**MANIFEST, "git_branch": "aidan/local-only",
                      "git_commit": "cafe123"},
        )
        calls = stub_hf()
        self._unmerged(monkeypatch, branch="declan/publishing-from-here",
                       commit="cafe123")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--allow-unmerged")

        message = next(c for c in calls
                       if c["fn"] == "upload_folder")["commit_message"]
        # The run's own entry credits where the data was generated...
        assert "2026-07-25_15-57_fullscale-500-opus5 " \
               "(branch aidan/local-only, commit cafe123)" in message
        # ...and the publish branch is a separate clause, not conflated with it.
        assert "published from unmerged branch declan/publishing-from-here" \
            in message

    def test_pre_v3_manifest_falls_back_to_the_live_branch(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """No existing manifest records git_branch, so the fallback is the
        common case, not an edge case."""
        run_dir, _ = make_run_dir(tmp_path)   # MANIFEST has no git_branch
        calls = stub_hf()
        self._unmerged(monkeypatch, branch="declan/wip")

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--allow-unmerged")
        message = next(c for c in calls
                       if c["fn"] == "upload_folder")["commit_message"]
        assert "branch declan/wip" in message

    def _dad_runs(self, tmp_path, *specs):
        """Several DAD run dirs, one per (run_name, commit, branch) spec."""
        dirs = []
        for run_name, commit, branch in specs:
            rd, _ = make_run_dir(
                tmp_path, pipeline="dad", audit_files=[], include_html=False,
                run_name=run_name,
                manifest={**MANIFEST, "run_id": run_name,
                          "git_commit": commit, "git_branch": branch},
            )
            dirs.append(rd)
        return dirs

    def test_combined_publish_names_only_the_unmerged_runs(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """The point of per-run stamping: a combined corpus is only as merged as
        its least-merged run, and a reader can trace a row to its run through
        the repo (example_gid) — so the warning must say WHICH runs, or that
        trace can't tell them whether a given row's code was reviewed. The
        merged run must not be smeared with the others' warning."""
        runs = self._dad_runs(
            tmp_path,
            ("2026-07-01_10-00_merged", "aaaaaaa", "main"),
            ("2026-07-02_10-00_wip", "bbbbbbb", "aidan/wip"),
            ("2026-07-03_10-00_other", "ccccccc", "constance/other"),
        )
        staging_dir = tmp_path / "staged"
        stub_hf()

        # HEAD is clean; only the 2nd and 3rd runs' commits are unmerged.
        def per_run(commit, fetch=True):
            merged = commit == "aaaaaaa"
            return {**MERGED_STATE, "run_commit": commit,
                    "run_commit_merged": merged}

        monkeypatch.setattr(publish_hf, "merge_state", per_run)

        calls = stub_hf()
        _run_main(monkeypatch, "--input", *[str(r) for r in runs],
                  "--repo-id", "org/repo", "--staging-dir", str(staging_dir),
                  "--allow-unmerged")

        message = next(c for c in calls
                       if c["fn"] == "upload_folder")["commit_message"]
        assert "2026-07-02_10-00_wip (branch aidan/wip, commit bbbbbbb)" in message
        assert "2026-07-03_10-00_other (branch constance/other, commit ccccccc)" \
            in message
        # The merged run is not smeared with the warning — it is named in the
        # "Publish dad: ..." half of the message, never in the unmerged clause.
        unmerged_clause = message.split("unmerged run(s): ", 1)[1]
        assert "2026-07-01_10-00_merged" not in unmerged_clause
        assert "2026-07-01_10-00_merged" in message
        # HEAD was merged, so there is no publish-branch clause to add.
        assert "published from unmerged branch" not in message

    def test_combined_publish_stays_silent_when_every_run_is_merged(
        self, tmp_path, monkeypatch, stub_hf, capsys
    ):
        runs = self._dad_runs(
            tmp_path,
            ("2026-07-01_10-00_a", "aaaaaaa", "main"),
            ("2026-07-02_10-00_b", "bbbbbbb", "main"),
        )
        staging_dir = tmp_path / "staged"
        calls = stub_hf()

        _run_main(monkeypatch, "--input", *[str(r) for r in runs],
                  "--repo-id", "org/repo", "--staging-dir", str(staging_dir))

        assert "NOT been merged" not in capsys.readouterr().err
        message = next(c for c in calls
                       if c["fn"] == "upload_folder")["commit_message"]
        assert "unmerged" not in message

    def test_combined_publish_fetches_the_remote_only_once(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """merge_state fetches origin/main; doing that once per run dir would
        make a 10-run publish hit the network 10 times for the same answer."""
        runs = self._dad_runs(
            tmp_path,
            ("2026-07-01_10-00_a", "aaaaaaa", "main"),
            ("2026-07-02_10-00_b", "bbbbbbb", "main"),
            ("2026-07-03_10-00_c", "ccccccc", "main"),
        )
        stub_hf()
        fetches = []

        def record(commit, fetch=True):
            fetches.append(fetch)
            return dict(MERGED_STATE)

        monkeypatch.setattr(publish_hf, "merge_state", record)

        _run_main(monkeypatch, "--input", *[str(r) for r in runs],
                  "--repo-id", "org/repo", "--staging-dir", str(tmp_path / "s"))
        assert fetches == [True, False, False]

    def test_a_publish_only_stamps_the_pipeline_it_publishes(
        self, tmp_path, monkeypatch, stub_hf
    ):
        """Each publish's record is its own Hub commit, so a sibling's earlier
        unmerged stamp is a different commit in the same history and cannot be
        overwritten or extended by this one. The stamp used to be regenerated
        whole on every publish, which is why it needed a persisted sidecar to
        survive; a commit message needs nothing."""
        run_dir, _ = make_run_dir(tmp_path, pipeline="dad", audit_files=[],
                                  include_html=False)
        calls = stub_hf()

        _run_main(monkeypatch, "--input", str(run_dir), "--repo-id", "org/repo",
                  "--staging-dir", str(tmp_path / "staged"))

        message = next(c for c in calls
                       if c["fn"] == "upload_folder")["commit_message"]
        assert message.startswith("Publish dad:")
        assert "sdf" not in message
        assert "unmerged" not in message
