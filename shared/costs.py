"""Read a run's cost log back and say plainly what it cost.

Every paid stage already appends a record per call (shared/api.py `_log_usage`);
nothing here bills anything, it only reads. The point is that a smoke run must
end with a number a human can act on, and with an honest projection to full
scale — which is NOT the smoke cost times a ratio.

Why the projection is not linear: the layer-3 system prompt is ~48k tokens of
constitution and review checks, sent with `cache_system=True`. The first call
at a given prompt WRITES that cache at 1.25x the input rate; later calls READ
it at 0.10x. So a 2-document run pays a cache write it then amortises over two
documents, and scaling its per-document average overstates a 150-document run
badly. `project` separates the one-time cache writes from the per-item cost and
scales only the latter.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from shared.api import _PRICING


def _cache_write_usd(record: dict) -> float:
    """The part of this record's cost that was a one-time cache write."""
    tokens = record.get("cache_creation_tokens") or 0
    prices = _PRICING.get(record.get("model"))
    if not tokens or not prices:
        return 0.0
    return tokens / 1_000_000 * prices[0] * 1.25


def mark(cost_log_path: str | Path) -> int:
    """Record count right now, to pass back as `since`.

    The global cost log (outputs/cost_log.jsonl) accumulates across every run
    and eval ever made, so a tool that writes there must report what IT spent,
    not the file's lifetime total. Take a mark before the first paid call.
    """
    path = Path(cost_log_path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def summarize(cost_log_path: str | Path, since: int = 0) -> dict:
    """Totals, per-stage and per-model breakdowns for one cost log.

    `since` skips that many leading records — pass the value `mark` returned
    before the work started to summarize only this invocation.
    """
    path = Path(cost_log_path)
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records = records[since:]

    by_stage: dict[str, dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0, "items": set()})
    by_model: dict[str, dict] = defaultdict(lambda: {"calls": 0, "usd": 0.0})
    totals = {
        "calls": 0, "usd": 0.0, "cache_write_usd": 0.0,
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_tokens": 0, "cache_read_tokens": 0,
        "retries": 0, "duration_s": 0.0,
    }
    for r in records:
        usd = float(r.get("cost_usd") or 0.0)
        stage = r.get("stage") or "(untagged)"
        totals["calls"] += 1
        totals["usd"] += usd
        totals["cache_write_usd"] += _cache_write_usd(r)
        for key in ("input_tokens", "output_tokens",
                    "cache_creation_tokens", "cache_read_tokens"):
            totals[key] += r.get(key) or 0
        totals["retries"] += max(0, (r.get("attempts") or 1) - 1)
        totals["duration_s"] += float(r.get("duration_s") or 0.0)
        by_stage[stage]["calls"] += 1
        by_stage[stage]["usd"] += usd
        if r.get("item_id"):
            by_stage[stage]["items"].add(r["item_id"])
        by_model[r.get("model") or "(unknown)"]["calls"] += 1
        by_model[r.get("model") or "(unknown)"]["usd"] += usd

    return {
        "path": str(path),
        "totals": totals,
        "by_stage": {k: {"calls": v["calls"], "usd": v["usd"], "items": len(v["items"])}
                     for k, v in sorted(by_stage.items(), key=lambda kv: -kv[1]["usd"])},
        "by_model": {k: dict(v) for k, v in sorted(by_model.items(), key=lambda kv: -kv[1]["usd"])},
        "backends": sorted({r.get("backend") for r in records if r.get("backend")}),
    }


def project(summary: dict, n_done: int, n_target: int) -> float | None:
    """What n_target items would cost, amortising the one-time cache writes.

    Returns None when there is nothing to project from.
    """
    if not n_done or n_target <= 0 or not summary["totals"]["calls"]:
        return None
    cache_write = summary["totals"]["cache_write_usd"]
    variable = summary["totals"]["usd"] - cache_write
    return cache_write + (variable / n_done) * n_target


def format_summary(summary: dict, *, title: str = "Cost",
                   n_done: int | None = None,
                   project_to: int | None = None) -> str:
    """A block a human can read at the end of a run."""
    t = summary["totals"]
    if not t["calls"]:
        return f"\n{'=' * 62}\n{title}: no API calls were made (nothing billed).\n{'=' * 62}"

    lines = ["", "=" * 62, f"{title.upper()}: ${t['usd']:.4f}", "=" * 62]
    lines.append(
        f"  {t['calls']} call(s)"
        + (f" over {n_done} item(s) — ${t['usd'] / n_done:.4f} each" if n_done else "")
    )
    lines.append(
        f"  tokens: {t['input_tokens']:,} in / {t['output_tokens']:,} out"
        + (f" · cache {t['cache_creation_tokens']:,} written, {t['cache_read_tokens']:,} read"
           if t["cache_creation_tokens"] or t["cache_read_tokens"] else "")
    )
    if t["cache_write_usd"]:
        lines.append(
            f"  of which ${t['cache_write_usd']:.4f} was ONE-TIME prompt-cache writes "
            "(does not scale with run size)"
        )
    if t["retries"]:
        lines.append(f"  {t['retries']} retry/retries (billed)")

    if summary["by_stage"]:
        lines.append("  by stage:")
        for stage, s in summary["by_stage"].items():
            per = f" · {s['items']} item(s)" if s["items"] else ""
            lines.append(f"    {stage:<22} ${s['usd']:>9.4f}  ({s['calls']} call(s){per})")
    if len(summary["by_model"]) > 1:
        lines.append("  by model:")
        for model, m in summary["by_model"].items():
            lines.append(f"    {model:<22} ${m['usd']:>9.4f}  ({m['calls']} call(s))")

    if project_to and n_done:
        projected = project(summary, n_done, project_to)
        if projected is not None:
            lines.append("-" * 62)
            lines.append(
                f"  PROJECTED at {project_to} items: ~${projected:.2f}"
            )
            lines.append(
                "  (one-time cache writes held constant, per-item cost scaled — "
                "so this is not simply the figure above times a ratio)"
            )
    lines.append("=" * 62)
    return "\n".join(lines)


def report(cost_log_path: str | Path, *, title: str = "Cost",
           n_done: int | None = None, project_to: int | None = None,
           since: int = 0) -> dict:
    """Summarize, print and return — the one call a stage needs at the end."""
    summary = summarize(cost_log_path, since=since)
    print(format_summary(summary, title=title, n_done=n_done, project_to=project_to))
    return summary
