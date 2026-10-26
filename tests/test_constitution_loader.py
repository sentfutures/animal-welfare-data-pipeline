"""Tests for shared/constitution_loader.py against the real constitution files."""

from shared import constitution_loader


def test_load_segments_assigns_sequential_principle_ids():
    segments = constitution_loader.load_segments()
    # CLAUDE.md contract: 16 sections mapped to principle_ids 0-15
    assert len(segments) == 16
    assert [s["principle_id"] for s in segments] == list(range(16))


def test_segments_have_nonempty_titles_and_content():
    for seg in constitution_loader.load_segments():
        assert seg["section_title"].strip()
        assert seg["content"].strip()


def test_meta_principle_ids_refer_to_real_segments():
    ids = {s["principle_id"] for s in constitution_loader.load_segments()}
    assert constitution_loader.META_PRINCIPLE_IDS <= ids


def test_segment_content_is_verbatim_from_source():
    seg = constitution_loader.load_segments()[1]
    assert seg["content"] in constitution_loader.load_constitution_welfare_reading()


def test_full_constitution_joins_preamble_and_both_documents():
    full = constitution_loader.load_full_constitution()
    claude = constitution_loader.load_constitution_claude()
    reading = constitution_loader.load_constitution_welfare_reading()
    assert "joins two complementary frameworks" in full
    assert claude in full
    assert reading in full
    assert full.index(claude) < full.index(reading)


def test_constitution_with_principles_joins_preamble_constitution_and_block():
    joined = constitution_loader.load_constitution_with_principles()
    claude = constitution_loader.load_constitution_claude()
    block = constitution_loader.format_principles(constitution_loader.load_principles())
    assert claude in joined
    assert block in joined
    assert joined.index(claude) < joined.index(block)
    # the sentient-beings reading must NOT ride along
    reading = constitution_loader.load_constitution_welfare_reading()
    assert reading not in joined


def test_constitution_with_principles_honors_base_dir_and_csv_fallback(tmp_path):
    # snapshot dir with its own constitution + CSV: both are read from there
    snap = tmp_path / "constitution"
    snap.mkdir()
    (snap / "constitution_claude.md").write_text("SNAP-CLAUDE")
    (snap / constitution_loader.PRINCIPLES_FILENAME).write_text(
        "number,principle,welfare_application,constitution_excerpts\n"
        "1,SNAP-PRINCIPLE,SNAP-APPLICATION,SNAP-EXCERPT\n"
    )
    joined = constitution_loader.load_constitution_with_principles(snap)
    assert "SNAP-CLAUDE" in joined
    assert "SNAP-PRINCIPLE" in joined
    # the renamed columns must actually render — a header mismatch drops the
    # application/excerpt lines silently, not loudly
    assert "SNAP-APPLICATION" in joined
    assert 'Constitution: "SNAP-EXCERPT"' in joined

    # snapshot predating the CSV: principles fall back to the repo's live copy
    (snap / constitution_loader.PRINCIPLES_FILENAME).unlink()
    joined = constitution_loader.load_constitution_with_principles(snap)
    assert "SNAP-CLAUDE" in joined
    repo_block = constitution_loader.format_principles(constitution_loader.load_principles())
    assert repo_block in joined


def test_format_principles_renders_pre_rename_snapshot_columns():
    # Run snapshots written before the 2026-07 column renames carry the old
    # headers; the viewer re-renders them via format_principles, so the legacy
    # names must keep rendering (not silently drop the two lines).
    block = constitution_loader.format_principles(constitution_loader.parse_principles(
        "number,principle,constitution_summary,raw_text_from_constitution\n"
        "1,OLD-PRINCIPLE,OLD-SUMMARY,OLD-QUOTE\n"
    ))
    assert "OLD-PRINCIPLE" in block
    assert "OLD-SUMMARY" in block
    assert 'Constitution: "OLD-QUOTE"' in block


def test_failure_modes_load_from_snapshot_with_live_fallback(tmp_path):
    live = constitution_loader.load_failure_modes()
    assert "## A. Noticing and weighing" in live
    snap = tmp_path / "constitution"
    snap.mkdir()
    (snap / constitution_loader.FAILURE_MODES_FILENAME).write_text("SNAP-FAILURES")
    assert constitution_loader.load_failure_modes(snap) == "SNAP-FAILURES"
    # snapshot predating the checklist: fall back to the repo's live copy
    (snap / constitution_loader.FAILURE_MODES_FILENAME).unlink()
    assert constitution_loader.load_failure_modes(snap) == live


def test_failure_modes_stay_guiding_document_agnostic():
    # the checklist is shared by both pipelines and must apply when another
    # constitution is swapped in, so it names no particular guiding document
    text = constitution_loader.load_failure_modes().lower()
    assert "constitution" not in text
    assert "anthropic" not in text
