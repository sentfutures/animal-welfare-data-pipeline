"""Layer 1 screen: a batch incoherence check on the planned specifications.

Runs after every layer-1 plan has finished and before drafting. The planner
sees one deal at a time, so a contradictory specification can slip through its
own INCOHERENT gate; the screen shows the model a batch of numbered
specifications (default 10) and asks which could not be written as a realistic
document. It answers with local numbers 1..N, never prompt ids; this module
maps the numbers back. Removed specifications are written to
screen_rejections.jsonl with the model's reason and are left out downstream,
like INCOHERENT plans.

The screening logic is importable on its own: ``screen_descriptions`` takes a
list of descriptions (and optionally their dealt variables) and returns the
removals, using the same template and parser as the pipeline stage, so offline
validation scripts can reuse it unchanged.

Config (``sdf`` section): ``screen_enabled`` (default true), ``screen_model``
(falls back to ``plan_model``, then the global model), ``screen_batch_size``
(default 10), ``screen_show_variables`` (default false: the screen sees only
the descriptions, which is all that travels downstream; true also shows each
specification's dealt variables).

Checkpoint semantics (protects paid work):
- screen_batches.jsonl is written BEFORE any call, so a resumed run screens
  identical batches. Specifications planned after it was written (plans
  retried on resume) get new batches appended; existing batches never change.
- One checkpoint per batch. A malformed, truncated, or out-of-range answer is
  NOT checkpointed, so --resume retries exactly that batch.
- A batch's rejections are appended before its checkpoint, so a crash between
  the two re-screens the batch; readers de-duplicate by prompt_id.
"""

import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from shared.matrix import split_sections

STAGE_TAG = "layer1_screen"
TEMPLATE_FILENAME = "layer1_screen.txt"
BATCHES_FILENAME = "screen_batches.jsonl"
REJECTIONS_FILENAME = "screen_rejections.jsonl"
CHECKPOINT_FILENAME = "_screen_checkpoint.json"
DEFAULT_BATCH_SIZE = 10
_MAX_TOKENS = 2000  # a short list of one-line reasons; bounds api-backend cost

REMOVALS_BLOCK_RE = re.compile(r"<removals>(.*?)</removals>", re.DOTALL | re.IGNORECASE)
REMOVAL_LINE_RE = re.compile(r"^#?(\d+)\s*[:.\-–—]\s*(\S.*)$")

VARIABLES_NOTE = (
    "\nEach specification is followed by the variables it was dealt: the combination "
    "its plan was built from. Use them only as extra context for judging whether the "
    "specification fits together.\n"
)

# Values an evaluation or report flags: a variable value removed at more than
# this multiple of the corpus-wide removal rate (with at least MIN_REMOVED_TO_FLAG
# removals, so a single removal in a tiny cell is not a finding).
FLAG_RATE_MULTIPLE = 2.0
MIN_REMOVED_TO_FLAG = 2


class ScreenAnswerError(ValueError):
    """The model's screen answer was truncated, malformed, or out of range."""


@dataclass(frozen=True)
class ScreenRemoval:
    number: int  # the local 1-based number the model gave
    reason: str

    @property
    def index(self) -> int:
        return self.number - 1


# ---------------------------------------------------------------- prompt and parsing

def load_screen_template(prompts_dir: Path) -> str | None:
    """The screen template, or None for a run whose snapshot predates the screen."""
    path = Path(prompts_dir) / TEMPLATE_FILENAME
    return path.read_text(encoding="utf-8") if path.is_file() else None


def build_screen_prompt(
    template: str,
    descriptions: list[str],
    variables_list: list[dict] | None = None,
) -> tuple[str | None, str]:
    """Render (system, user) for one batch. ``variables_list`` (parallel to
    descriptions) is shown only when given."""
    if variables_list is not None and len(variables_list) != len(descriptions):
        raise ValueError("variables_list must be parallel to descriptions")
    blocks = []
    for number, description in enumerate(descriptions, 1):
        block = f"<specification number=\"{number}\">\n{description.strip()}\n"
        if variables_list is not None:
            lines = "\n".join(f"- {name}: {value}" for name, value in variables_list[number - 1].items())
            block += f"\nDealt variables:\n{lines}\n"
        blocks.append(block + "</specification>")
    rendered = template.format(
        specification_count=len(descriptions),
        variables_note=VARIABLES_NOTE if variables_list is not None else "",
        specifications="\n\n".join(blocks),
    )
    return split_sections(rendered)


def parse_screen_answer(raw: str, specification_count: int) -> list[ScreenRemoval]:
    """Removals from one answer. Raises ScreenAnswerError unless there is exactly
    one <removals> block whose every non-empty line is 'number: reason' with a
    distinct number in 1..specification_count."""
    blocks = REMOVALS_BLOCK_RE.findall(raw or "")
    if len(blocks) != 1:
        raise ScreenAnswerError(f"expected exactly one <removals> block, found {len(blocks)}")
    removals: list[ScreenRemoval] = []
    seen: set[int] = set()
    for line in blocks[0].splitlines():
        line = line.strip()
        if not line:
            continue
        match = REMOVAL_LINE_RE.match(line)
        if match is None:
            raise ScreenAnswerError(f"unparseable removal line: {line!r}")
        number = int(match.group(1))
        if not 1 <= number <= specification_count:
            raise ScreenAnswerError(f"number {number} is outside 1..{specification_count}")
        if number in seen:
            raise ScreenAnswerError(f"number {number} listed twice")
        seen.add(number)
        removals.append(ScreenRemoval(number=number, reason=match.group(2).strip()))
    return removals


def screen_descriptions(
    descriptions: list[str],
    variables_list: list[dict] | None = None,
    *,
    template: str,
    model: str | None = None,
    item_id: str | None = None,
) -> list[ScreenRemoval]:
    """Screen one batch of descriptions and return the removals (possibly none).

    Raises ScreenAnswerError for a truncated or malformed answer; API errors
    propagate. Uses the global temperature and tags the call ``layer1_screen``
    for the cost log."""
    system, user = build_screen_prompt(template, descriptions, variables_list)
    raw, stop_reason = api.call_claude(
        user,
        system_prompt=system or "",
        model=model,
        max_tokens=_MAX_TOKENS,
        stage=STAGE_TAG,
        item_id=item_id,
        return_stop_reason=True,
    )
    if stop_reason != "end_turn":
        raise ScreenAnswerError(f"truncated answer (stop_reason={stop_reason})")
    return parse_screen_answer(raw, len(descriptions))


# ---------------------------------------------------------------- batching

def plan_batches(
    output_dir: Path, planned_ids: list[str], batch_size: int, seed: int
) -> list[dict]:
    """The batches for this run, persisted before any call.

    A seeded shuffle (of the sorted ids, so input order is irrelevant) cut into
    batch_size chunks. On resume the saved batches are reused verbatim; ids
    planned since then are batched and appended."""
    path = output_dir / BATCHES_FILENAME
    batches = utils.load_jsonl(path)
    batched = {pid for batch in batches for pid in batch["prompt_ids"]}
    unbatched = sorted(pid for pid in planned_ids if pid not in batched)
    if not unbatched:
        return batches
    random.Random(f"{seed}:{STAGE_TAG}:{len(batches)}").shuffle(unbatched)
    for start in range(0, len(unbatched), batch_size):
        batch = {
            "batch_id": f"screen_batch_{len(batches):04d}",
            "prompt_ids": unbatched[start:start + batch_size],
        }
        batches.append(batch)
        utils.append_jsonl(batch, path)
    return batches


# ---------------------------------------------------------------- stage

def load_rejections(output_dir: Path) -> dict[str, dict]:
    """{prompt_id: rejection record}, de-duplicated."""
    return {r["prompt_id"]: r for r in utils.load_jsonl(output_dir / REJECTIONS_FILENAME)}


def drop_rejected(config: dict, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Plans minus those a completed or partial screen already removed (no calls).
    Used when a run resumes past layer 1 and plans are re-read from disk."""
    if not config["sdf"].get("screen_enabled", True):
        return plans
    rejected = load_rejections(output_dir)
    return [p for p in plans if p["prompt_id"] not in rejected]


def run(config: dict, prompts_dir: Path, output_dir: Path, plans: list[dict]) -> list[dict]:
    """Screen the planned (non-INCOHERENT) specifications; return ``plans`` minus
    the removed ones. A disabled screen, or a run whose prompt snapshot has no
    screen template, makes no calls and returns ``plans`` unchanged."""
    sdf = config["sdf"]
    if not sdf.get("screen_enabled", True):
        print("  Screen disabled (sdf.screen_enabled: false)")
        return plans
    template = load_screen_template(prompts_dir)
    if template is None:
        print(f"  No {TEMPLATE_FILENAME} in this run's prompt snapshot — screen skipped")
        return plans

    show_variables = bool(sdf.get("screen_show_variables", False))
    batch_size = int(sdf.get("screen_batch_size", DEFAULT_BATCH_SIZE))
    model = sdf.get("screen_model") or sdf.get("plan_model")
    checkpoint = utils.Checkpoint(output_dir / CHECKPOINT_FILENAME)
    rejections_path = output_dir / REJECTIONS_FILENAME

    by_id = {p["prompt_id"]: p for p in plans if p.get("description")}
    batches = plan_batches(output_dir, list(by_id), batch_size, sdf.get("seed", 0))
    # Batches can name ids that are no longer planned (plans file edited); screen what exists.
    pending = [b for b in batches if not checkpoint.is_done(b["batch_id"])]
    print(f"  Screening {len(by_id)} specifications in {len(batches)} batches of up to "
          f"{batch_size} ({len(batches) - len(pending)} already done)")

    def screen_one(batch: dict):
        members = [by_id[pid] for pid in batch["prompt_ids"] if pid in by_id]
        try:
            started = time.monotonic()
            removals = screen_descriptions(
                [m["description"] for m in members],
                [m["variables"] for m in members] if show_variables else None,
                template=template, model=model, item_id=batch["batch_id"],
            )
            return members, removals, time.monotonic() - started, None
        except ScreenAnswerError as e:
            return members, None, 0.0, f"unusable answer ({e})"
        except Exception as e:  # one failing call must not discard its siblings' work
            return members, None, 0.0, f"API call failed ({type(e).__name__}: {e})"

    workers = config.get("workers", 1)
    failed = 0
    removed_total = 0
    started_all = time.monotonic()
    for done, (batch, (members, removals, seconds, problem)) in enumerate(
        zip(pending, utils.parallel_map(screen_one, pending, workers)), 1
    ):
        if problem:
            failed += 1
            print(f"  {batch['batch_id']}: {problem} — will retry on resume")
            continue
        for removal in removals:
            member = members[removal.index]
            utils.append_jsonl({
                "prompt_id": member["prompt_id"],
                "batch_id": batch["batch_id"],
                "number": removal.number,
                "reason": removal.reason,
                "description": member["description"],
            }, rejections_path)
        checkpoint.mark_done(batch["batch_id"])
        removed_total += len(removals)
        print(f"  Screened {batch['batch_id']} ({done}/{len(pending)}): {len(members)} specifications, "
              f"{len(removals)} removed in {seconds:.1f}s")

    if pending and failed == len(pending):
        raise SystemExit(
            "layer1_screen: every pending screen call failed — this is systemic "
            "(auth, backend, or network), not per-batch; fix and --resume."
        )
    rejected = load_rejections(output_dir)
    print(f"  Screen removed {len(rejected)} of {len(by_id)} specifications "
          f"({removed_total} this session, {time.monotonic() - started_all:.1f}s)")
    return [p for p in plans if p["prompt_id"] not in rejected]


# ---------------------------------------------------------------- reporting

def load_screen_outputs(run_dir: Path) -> dict | None:
    """The screen's results for a run, or None when it never ran (older runs).

    Returns {"screened_ids": set, "removed": {prompt_id: record}, "batches_total",
    "batches_done"}; only specifications in a checkpointed batch count as screened."""
    stage_dir = utils.sdf_stage_dir(run_dir, 1)
    batches = utils.load_jsonl(stage_dir / BATCHES_FILENAME)
    if not batches:
        return None
    checkpoint = utils.Checkpoint(stage_dir / CHECKPOINT_FILENAME)
    done = [b for b in batches if checkpoint.is_done(b["batch_id"])]
    return {
        "screened_ids": {pid for b in done for pid in b["prompt_ids"]},
        "removed": load_rejections(stage_dir),
        "batches_total": len(batches),
        "batches_done": len(done),
    }


def removal_rate_rows(
    plans: list[dict], screen: dict, axes: list[str] | None = None
) -> list[dict]:
    """Removal rate per variable value over the screened specifications.

    Each row: variable, value, screened, removed, rate, corpus_rate, flagged
    (rate above FLAG_RATE_MULTIPLE x the corpus rate with at least
    MIN_REMOVED_TO_FLAG removals). Sorted flagged first, then by rate."""
    screened = [p for p in plans if p["prompt_id"] in screen["screened_ids"]]
    removed_ids = set(screen["removed"])
    corpus_rate = (sum(p["prompt_id"] in removed_ids for p in screened) / len(screened)) if screened else 0.0
    counts: dict[tuple[str, str], list[int]] = {}
    for plan in screened:
        for axis, value in (plan.get("variables") or {}).items():
            if axes is not None and axis not in axes:
                continue
            cell = counts.setdefault((axis, value), [0, 0])
            cell[0] += 1
            cell[1] += plan["prompt_id"] in removed_ids
    rows = []
    for (axis, value), (n_screened, n_removed) in counts.items():
        rate = n_removed / n_screened
        rows.append({
            "variable": axis, "value": value, "screened": n_screened, "removed": n_removed,
            "rate": rate, "corpus_rate": corpus_rate,
            "flagged": n_removed >= MIN_REMOVED_TO_FLAG and rate > FLAG_RATE_MULTIPLE * corpus_rate,
        })
    rows.sort(key=lambda r: (not r["flagged"], -r["rate"], r["variable"], r["value"]))
    return rows
