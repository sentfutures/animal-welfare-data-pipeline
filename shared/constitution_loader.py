"""Load and segment the constitution documents.

Three source files live in constitution/:
- constitution_claude.md — the original Claude constitution, verbatim.
- constitution_sentient_beings.md — the animal-welfare section-by-section
  reading, with one `## ` header per section.
- constitution_principles.csv — the distilled welfare-relevant principles
  (number, principle, welfare_application, constitution_excerpts),
  embedded as a checklist in the DAD step-3 rewrite prompt and, joined with the
  Claude constitution, in the SDF prompts (system prompt at layers 3-5,
  template variables at layer 2).

The sentient-beings reading is not sent by either pipeline's generation calls:
it is the source the principles CSV was distilled from, and it supplies the
rubric for evals/compliance_sdf.py. load_full_constitution/load_segments serve
that eval and the viewer.
"""

import csv
import io
from pathlib import Path

_CONSTITUTION_DIR = Path(__file__).parent.parent / "constitution"
_CLAUDE_PATH = _CONSTITUTION_DIR / "constitution_claude.md"
_SENTIENT_PATH = _CONSTITUTION_DIR / "constitution_sentient_beings.md"
PRINCIPLES_FILENAME = "constitution_principles.csv"

_JOIN_PREAMBLE = """\
This document joins two complementary frameworks:
Part I: The original claude constitution
Part II: A reading of the original constitution

The Part II document does not modify, extend, or rewrite Claude's Constitution. It
reads it. For each section of the constitution that bears on the treatment
of animals and other sentient beings, it quotes the section verbatim and then
explains what that section already implies, given that the constitution
already directs Claude to weigh "the welfare of animals and of all sentient
beings." Our own contributions appear only as interpretation, as labeled
heuristics, as examples, and as citations to authoritative outside documents —
never as words put into the constitution's mouth.
"""

_PRINCIPLES_JOIN_PREAMBLE = """\
This document joins two complementary parts:
Part I: The original claude constitution
Part II: The distilled welfare-relevant principles

The Part II principles do not modify, extend, or rewrite Claude's Constitution.
Each principle distills what the constitution already implies for the treatment
of animals and other sentient beings — given that the constitution already
directs Claude to weigh "the welfare of animals and of all sentient beings" —
and carries the verbatim constitution quote it distills. Nothing in Part II
puts words into the constitution's mouth.
"""

# Sections of the reading that describe the document itself rather than a
# principle usable for scenario generation (scope note, violation-typology
# appendix, closing humility note). Must track the reading's section order:
# 0 = scope, 14 = violation typology, 15 = closing note.
META_PRINCIPLE_IDS = {0, 14, 15}


def _paths(base_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve the two source files against base_dir (a run's inputs/constitution
    snapshot) or the repo's live constitution/ directory."""
    base = Path(base_dir) if base_dir else _CONSTITUTION_DIR
    return base / _CLAUDE_PATH.name, base / _SENTIENT_PATH.name


def load_constitution_claude(base_dir: str | Path | None = None) -> str:
    """Return the original Claude constitution, verbatim."""
    claude_path, _ = _paths(base_dir)
    return claude_path.read_text(encoding="utf-8")


def load_constitution_welfare_reading(base_dir: str | Path | None = None) -> str:
    """Return the sentient-beings reading of the constitution, verbatim."""
    _, sentient_path = _paths(base_dir)
    return sentient_path.read_text(encoding="utf-8")


def load_full_constitution(base_dir: str | Path | None = None) -> str:
    """Return the full constitution: join preamble + Claude constitution + reading.
    No longer sent by pipeline generation calls (SDF now uses
    load_constitution_with_principles); kept for the viewer and legacy runs."""
    return "\n---\n\n".join([
        _JOIN_PREAMBLE,
        load_constitution_claude(base_dir),
        load_constitution_welfare_reading(base_dir),
    ])


def load_constitution_with_principles(base_dir: str | Path | None = None) -> str:
    """Return the Claude constitution joined with the distilled principles block:
    join preamble + Claude constitution + formatted principles CSV. Used as the
    system prompt at SDF layers 3-5 (rewrite and scoring); SDF layer 2 embeds the
    same two texts via template variables instead. Snapshots that predate the
    principles CSV fall back to the repo's live copy, as in DAD step 3."""
    try:
        principles = load_principles(base_dir)
    except FileNotFoundError:
        principles = load_principles()
    return "\n---\n\n".join([
        _PRINCIPLES_JOIN_PREAMBLE,
        load_constitution_claude(base_dir),
        format_principles(principles),
    ])


def parse_principles(csv_text: str) -> list[dict]:
    """Parse the principles CSV: one dict per row with number, principle,
    welfare_application, constitution_excerpts (rows are keyed by header name,
    so column order in the file is free to change)."""
    return list(csv.DictReader(io.StringIO(csv_text)))


def load_principles(base_dir: str | Path | None = None) -> list[dict]:
    """Load the distilled welfare principles from base_dir (a run's
    inputs/constitution snapshot) or the repo's live constitution/."""
    base = Path(base_dir) if base_dir else _CONSTITUTION_DIR
    return parse_principles((base / PRINCIPLES_FILENAME).read_text(encoding="utf-8"))


def format_principles(principles: list[dict]) -> str:
    """Render the principles as the numbered block embedded in the DAD step-3
    rewrite prompt (re-rendered by the viewer — keep in sync). Each principle
    carries its summary and the verbatim constitution quote it distills."""
    lines = []
    for p in principles:
        lines.append(f"{p.get('number', '?')}. {p.get('principle', '').strip()}")
        # Columns renamed 2026-07 (constitution_summary -> welfare_application,
        # raw_text_from_constitution -> constitution_excerpts). The old names
        # stay as fallbacks so pre-rename run snapshots re-render faithfully
        # in the viewer.
        application = (p.get("welfare_application")
                       or p.get("constitution_summary") or "").strip()
        if application:
            lines.append(f"   {application}")
        excerpts = (p.get("constitution_excerpts")
                    or p.get("raw_text_from_constitution") or "").strip()
        if excerpts:
            lines.append(f'   Constitution: "{excerpts}"')
        lines.append("")
    return "\n".join(lines).strip()


def load_segments(base_dir: str | Path | None = None) -> list[dict]:
    """Split the sentient-beings reading on '## ' headers.

    Returns:
        List of {"section_title", "content", "principle_id"} dicts, one per
        section, with principle_id assigned in file order (0-15).
    """
    segments, current_title, current_lines = [], None, []
    for line in load_constitution_welfare_reading(base_dir).splitlines():
        if line.startswith("## "):
            if current_title and current_lines:
                segments.append({"section_title": current_title, "content": "\n".join(current_lines).strip()})
            current_title, current_lines = line[3:].strip(), []
        elif current_title is not None:
            current_lines.append(line)
    if current_title and current_lines:
        segments.append({"section_title": current_title, "content": "\n".join(current_lines).strip()})
    for i, seg in enumerate(segments):
        seg["principle_id"] = i
    return segments
