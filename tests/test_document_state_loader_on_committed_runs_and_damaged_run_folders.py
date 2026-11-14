"""The document state loader on the SDF runs committed under ``outputs/sdf/runs/`` (every layout, the
website's pinned run included) and on run folders that are damaged or are not run folders.

The expected counts below are read off each run's own files (the final corpus and the scores file with
the manifest's threshold), so they are the stage files' counts and not the loader's.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from sdf_pipeline.document_state import DocumentOutcome, RemovalReasonCode, RunLayout
from sdf_pipeline.document_state_final_corpus import derive_final_corpus_records
from sdf_pipeline.document_state_loader import (
    StageFileCorruptError, UnknownRunLayoutError, detect_run_layout, load_document_states,
    load_document_states_with_run_notes)

RUNS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "sdf" / "runs"
PINNED_WEBSITE_RUN = "2026-07-25_15-57_fullscale-500-opus5"

# run name -> (layout, documents, final, below threshold at the gate, in progress)
COMMITTED_RUNS = {
    "2026-07-01_13-49_dev": (RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX, 10, 7, 3, 0),
    "2026-07-01_14-38_english-only-test": (RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX, 10, 6, 4, 0),
    "2026-07-06_22-19_notebook-port-smoke": (RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX, 67, 67, 0, 0),
    "2026-07-08_14-59_allen_dev": (RunLayout.FIVE_LAYER_BEFORE_VARIABLES_MATRIX, 53, 47, 6, 0),
    "2026-07-11_20-06_matrix100-cli": (RunLayout.MATRIX_BEFORE_LAYER_RENUMBERING, 100, 100, 0, 0),
    "2026-07-13_13-18_al-gap-fixes-100docs": (RunLayout.MATRIX_BEFORE_LAYER_RENUMBERING, 100, 96, 2, 2),
    PINNED_WEBSITE_RUN: (RunLayout.MATRIX_BEFORE_LAYER_RENUMBERING, 500, 477, 12, 11),
}


def _records(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_committed_runs_cover_all_three_layouts_or_are_absent():
    present = {p.name for p in RUNS_ROOT.iterdir()} if RUNS_ROOT.is_dir() else set()
    assert set(COMMITTED_RUNS) <= present, "a committed run the test names is missing"
    assert PINNED_WEBSITE_RUN in COMMITTED_RUNS


@pytest.mark.parametrize("run_name", sorted(COMMITTED_RUNS))
def test_counts_per_outcome_and_reason_code_match_the_runs_own_files(run_name):
    layout, documents, final_count, below_threshold, in_progress = COMMITTED_RUNS[run_name]
    run = RUNS_ROOT / run_name
    assert detect_run_layout(run) is layout
    states = load_document_states(run)
    assert len(states) == documents
    counts = Counter((s.outcome, s.removal.reason_code if s.removal else None) for s in states)
    expected = {(DocumentOutcome.FINAL, None): final_count,
                (DocumentOutcome.REMOVED, RemovalReasonCode.BELOW_THRESHOLD_AT_GATE): below_threshold,
                (DocumentOutcome.IN_PROGRESS, None): in_progress}
    assert counts == Counter({key: n for key, n in expected.items() if n})
    assert final_count == len(_records(run / "final" / "sdf_corpus.jsonl"))
    # the gate count is the scores the manifest's threshold turns away, read off the files here
    threshold = json.loads((run / "run_manifest.json").read_text())["config"]["sdf"]["min_score_threshold"]
    scores_file = next(run.glob("layer*/scores.jsonl"))
    scored_below = [r for r in _records(scores_file)
                    if not (r["scores"]["alignment"] >= threshold and r["scores"]["realism"] >= threshold)]
    assert len(scored_below) == below_threshold
    assert {s.document_id for s in states if s.removal} == {r["doc_id"] for r in scored_below}


@pytest.mark.parametrize("run_name", sorted(COMMITTED_RUNS))
def test_corpus_derived_from_the_states_equals_the_committed_final_corpus_record_for_record(run_name):
    run = RUNS_ROOT / run_name
    states = load_document_states(run)
    derived = derive_final_corpus_records(states)
    final = _records(run / "final" / "sdf_corpus.jsonl")
    derived_by_id = {r["doc_id"]: r for r in derived}
    assert len(derived_by_id) == len(derived) and set(derived_by_id) == {r["doc_id"] for r in final}
    for record in final:
        assert derived_by_id[record["doc_id"]] == record
        assert list(derived_by_id[record["doc_id"]]) == list(record)
        assert list(derived_by_id[record["doc_id"]]["scores"]) == list(record["scores"])
    # Today's order is the score file's order, which is deal order unless the run was resumed.
    scores_file = next(run.glob("layer*/scores.jsonl"))
    scored_ids = [r["doc_id"] for r in _records(scores_file) if r["doc_id"] in derived_by_id]
    state_ids = [s.document_id for s in states if s.document_id in derived_by_id]
    if scored_ids == state_ids:
        assert [r["doc_id"] for r in derived] == [r["doc_id"] for r in final]
    else:
        assert [r["doc_id"] for r in derived] == state_ids  # deal order, which differs from today's file


def test_the_pinned_website_run_loads_with_deal_order_ids_descriptions_and_scores():
    states = load_document_states(RUNS_ROOT / PINNED_WEBSITE_RUN)
    assert [s.document_id for s in states] == [f"matrix_{n:06d}" for n in range(500)]
    final = next(s for s in states if s.is_final)
    assert final.dealt_variables["document_type"] and final.planning_notes and final.current_description
    assert [v.version_number for v in final.text_versions] == [1, 2]
    assert final.text_versions[1].critique and final.scores.alignment >= 7
    assert final.deal_round_number == 0 and final.screen_judgment is None
    assert final.latest_text_was_written_from_a_different_description is False
    unfinished = [s for s in states if s.outcome is DocumentOutcome.IN_PROGRESS]
    assert len(unfinished) == 11 and all(s.removal is None for s in unfinished)


def test_the_five_layer_runs_keep_their_legacy_fields_and_have_no_deal():
    states = load_document_states(RUNS_ROOT / "2026-07-08_14-59_allen_dev")
    assert all(s.hand_index is None and s.dealt_variables is None for s in states)
    assert {"subtype_id", "type_id", "role", "register", "language"} <= set(states[0].legacy_fields)
    assert states[0].description_versions and states[0].text_versions[1].critique


# ------------------------------------------------------------------------------ damaged and unknown folders

def _copy_run(run_name: str, destination: Path) -> Path:
    copy = destination / run_name
    shutil.copytree(RUNS_ROOT / run_name, copy, ignore=shutil.ignore_patterns("inputs", "audit", "cost_log.jsonl"))
    return copy


def test_a_cut_off_last_line_is_skipped_and_reported(tmp_path):
    run = _copy_run("2026-07-11_20-06_matrix100-cli", tmp_path)
    scores_path = run / "layer5" / "scores.jsonl"
    lines = scores_path.read_text(encoding="utf-8").splitlines()
    scores_path.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:40], encoding="utf-8")
    loaded = load_document_states_with_run_notes(run)
    assert len(loaded.states) == 100
    assert len(loaded.run_notes) == 1 and "layer5/scores.jsonl line 100" in loaded.run_notes[0]
    last = next(s for s in loaded.states if s.document_id == json.loads(lines[-1])["doc_id"])
    assert last.scores is None and last.outcome is DocumentOutcome.IN_PROGRESS


def test_a_damaged_line_in_the_middle_of_a_file_is_refused_naming_file_and_line(tmp_path):
    run = _copy_run("2026-07-11_20-06_matrix100-cli", tmp_path)
    path = run / "layer3" / "drafts.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3][:30]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(StageFileCorruptError, match=r"drafts\.jsonl line 4"):
        load_document_states(run)


def test_a_missing_optional_file_leaves_its_fields_empty(tmp_path):
    run = _copy_run("2026-07-13_13-18_al-gap-fixes-100docs", tmp_path)
    (run / "layer5" / "scores.jsonl").unlink()
    states = load_document_states(run)
    assert len(states) == 100 and all(s.scores is None and not s.is_final for s in states)
    (run / "final" / "sdf_corpus.jsonl").unlink()
    (run / "run_manifest.json").unlink()
    assert len(load_document_states(run)) == 100


def test_a_run_with_no_final_corpus_or_threshold_does_not_decide_the_gate(tmp_path):
    run = _copy_run("2026-07-13_13-18_al-gap-fixes-100docs", tmp_path)
    (run / "final" / "sdf_corpus.jsonl").unlink()
    assert not any(s.outcome is not DocumentOutcome.IN_PROGRESS for s in load_document_states(run))
    restored = _copy_run("2026-07-13_13-18_al-gap-fixes-100docs", tmp_path / "again")
    states = load_document_states(restored, gate_score_threshold=9)
    assert sum(s.removal is not None for s in states) > 2


def test_an_unknown_layout_is_refused_with_a_message_naming_the_run_folder(tmp_path):
    folder = tmp_path / "not_a_run_folder_2026-11-12"
    (folder / "something_else").mkdir(parents=True)
    with pytest.raises(UnknownRunLayoutError, match="not_a_run_folder_2026-11-12"):
        load_document_states(folder)
    with pytest.raises(UnknownRunLayoutError, match="no_such_folder"):
        load_document_states(tmp_path / "no_such_folder")


def test_the_loader_does_not_write_into_the_run_folder(tmp_path):
    run = _copy_run("2026-07-13_13-18_al-gap-fixes-100docs", tmp_path)
    before = {str(p): p.stat().st_mtime_ns for p in run.rglob("*")}
    names_before = sorted(before)
    load_document_states(run)
    assert {str(p): p.stat().st_mtime_ns for p in run.rglob("*")} == before and sorted(before) == names_before
