"""Step 1: Generate scenarios, then draft dilemma prompts. Sub-stages:

- Step 1a — scenario deal + plan: deal a stratified variable combination per
  example from the weighted matrix in prompts/dad/variables.txt (offline;
  dad_pipeline/compose_scenarios.py owns the dealing and the structural
  rules), then one plan call per deal renders prompts/dad/step1a_scenario.txt
  and turns the combination into a self-contained scenario description
  (extracted fail-closed; INCOHERENT combinations are checkpointed as
  deliberate rejections in scenario_rejects.jsonl and never retried). Deals
  persist to step1/scenario_deals.jsonl and planned scenarios to
  step1/scenarios.jsonl, so --resume replays the same deal and only plans
  what's missing. Pre-plan runs (scenarios.jsonl without a deals file) skip
  planning; when drafted through the current single-scenario template, their
  rendered legacy card fills the scenario-description slot (see
  compose_scenarios.render_draft_prompt) — a run whose snapshot still holds
  the old batch template is refused loudly instead.

- Step 1b — first attempt: one draft call per scenario, fanned out via
  parallel_map. The prompt renders the plan's scenario
  description, the persona voice, and the dealt length register
  (prompts/dad/step1b_dilemmas.txt); the reply is the user message inside
  <user_prompt> tags, extracted fail-closed. The dealt length register is an
  instruction only — never measured or enforced. 1b writes no annotation —
  the dealt labels ARE the design, and the
  record's annotation is synthesized from the scenario. Accepted drafts are
  taken as returned; distribution fidelity is monitored by the corpus-level
  checklist instead.

- Step 1c — quality gate (optional; config dad.dilemmas.gate, on by default):
  a model call JUDGES each 1b draft against its scenario per
  prompts/dad/step1c_gate.txt — pass/fail, never a rewrite. The four checks:
  the welfare stake is load-bearing, the message is self-contained, the scene
  is cohesive, and the message reads like a real person wrote it (the
  card-adherence check was deliberately dropped — see 9f65553; distribution
  fidelity is the corpus checklist's job). A rejected scenario is
  routed back through the drafting loop with the gate's reasons injected into
  the next attempt (so it redrafts away from them), capped at
  MAX_GATE_REDRAFTS; a scenario still failing after the cap ships with
  gate_failures stamped on its record. Verdicts are logged to step1/gate.jsonl;
  unusable gate replies (retried once) persist to step1/gate_failures.jsonl
  and fail open (ship the draft — degraded gating costs a weak prompt, never
  a stalled run).

- Step 1d — prompt refine (optional; config dad.dilemmas.refine, on by
  default; runs on gate-passed drafts): one review-and-rewrite call per draft
  per prompts/dad/step1d_refine.txt —
  editor notes in prose, the rewritten message in <revised_user_prompt> tags,
  extracted fail-closed, with the dealt surface form / visibility / attitude /
  opening / closing bound as cards. The 1b draft is kept on the record
  (draft_user_message + refine_notes) and the before/after is logged to
  step1/refinements.jsonl. A draft 1c judges unfixable by rewriting alone
  (<unfixable> verdict) is a deliberate rejection mirroring 1a's INCOHERENT:
  checkpointed to step1/refine_rejects.jsonl, never retried, and the run
  ships fewer examples. The gate REDRAFTS scenario-level failures from
  scratch; the refine REWRITES surface problems in place — complementary,
  composed in that order.

The Part 4 checklist re-prints at the end as verification; thresholds are the
spec's, enforcement stays human.

Handwritten examples can be imported ahead of drafting via
config dad.dilemmas.seed_path (JSONL with prompt/user_message, optional
annotation); seeds carry no scenario. Every shipped record is identified by
its content-keyed prompt_gid (P-####, id_registry.py); there are no per-run
prompt ids.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, utils
from dad_pipeline import compose_scenarios
from dad_pipeline.compose_scenarios import render_scenario_block as format_scenario  # noqa: F401 (viewer + 1c)
from dad_pipeline.id_registry import (IdRegistry, prompt_fingerprint, registry_path,
                                      scenario_fingerprint)

_LIST_FIELDS = ("domain", "user_goal", "values_in_tension", "claims")
_STR_FIELDS = ("moral_patients", "visibility", "user_attitude", "conflict",
               "welfare_magnitude", "user_stakes", "leverage")

# Models sometimes transcribe a compound label as its halves ("Education /
# Parenting" -> ["Education", "Parenting"]); map any half back to its card so
# the tally counts the card, not the fragments. The domain vocabulary comes
# from the RUN'S variables.txt, threaded in by the caller — never the repo's
# live copy, which may have drifted since the run was dealt.
def _canon_domain(label: str, domain_by_part: dict) -> str:
    return domain_by_part.get(str(label).strip().lower(), str(label).strip())

def _scenario_cards_from_deal(p: dict) -> dict:
    """The record's scenario_cards: the full set of dealt categorical cards,
    copied from the scenario deal.

    The deal IS the design (1b writes no write-up), so these are the DEALT
    labels — NOT a check of what the drafted text realized. Consumed by the
    Part-4 checklist tally and the viewer. Renamed from 'annotation' (2026-07):
    it was never a write-up, just the cards. Records from before the rename
    persist the field as 'annotation' (and may carry legacy write-up fields
    dilemma_anatomy/values_in_tension/moral_patients/claims); readers fall back
    via dealt_cards() and format_scenario_cards() still renders those."""
    return _normalize_scenario_cards({
        "domain": list(p.get("domain") or []),
        "user_goal": list(p.get("user_goal") or []),
        "taxa_category": p.get("taxa_category", ""),
        "taxa_subcategory": p.get("taxa_subcategory", ""),
        "visibility": p.get("visibility", ""),
        "user_attitude": p.get("user_attitude", ""),
        "user_moral_framework": p.get("user_moral_framework", ""),
        "conflict": p.get("conflict", ""),
        "welfare_magnitude": p.get("welfare_magnitude", ""),
        "user_stakes": p.get("user_stakes", ""),
        "leverage": p.get("leverage", ""),
        "anchor_value_pair": p.get("anchor_value_pair", ""),
        "secondary_value_pair": p.get("secondary_value_pair", ""),
        "claim_pattern": p.get("claim_pattern", ""),
        "surface_form": p.get("surface_form", ""),
        "frontier_frame": p.get("frontier_frame", ""),
        "length_class": p.get("length_class", ""),
        "cultural_setting": p.get("cultural_setting", ""),
        "deliverable": p.get("deliverable", ""),
        "failure_mode_target": p.get("failure_mode_target", ""),
        "failure_mode_salience": p.get("failure_mode_salience", ""),
        "archetype": p.get("archetype", ""),
    })


def dealt_cards(rec: dict) -> dict:
    """The dealt cards for a record, reading the current 'scenario_cards' key
    and falling back to the pre-rename 'annotation' key so committed old runs
    still tally and render. Public so the viewer reads records the same way."""
    return rec.get("scenario_cards") or rec.get("annotation") or {}


def format_scenario_cards(annotation: dict) -> str:
    """Human-readable render of a record's scenario_cards (the dealt cards),
    used by the viewer. Also renders the legacy write-up fields (dilemma_anatomy,
    values_in_tension, moral_patients, claims, direction) when present, so old
    'annotation'-era records still display fully. (Formerly format_annotation.)"""
    anatomy = annotation.get("dilemma_anatomy") or {}
    lines = [
        f"Domain: {', '.join(annotation.get('domain') or [])}",
        f"User goal: {', '.join(annotation.get('user_goal') or [])}",
        # these fields exist only on records from older runs whose 1b returned
        # a written annotation; render them conditionally
        *([f"Dilemma anatomy: Goal = {anatomy.get('goal', '')} | "
           f"Temptation = {anatomy.get('temptation', '')} | Cost = {anatomy.get('cost', '')}"]
          if anatomy else []),
        *([f"Values in tension: {'; '.join(annotation['values_in_tension'])}"]
          if annotation.get("values_in_tension") else []),
        *([f"Moral patients: {annotation['moral_patients']}"]
          if annotation.get("moral_patients") else []),
        f"Visibility: {annotation.get('visibility', '')}",
        f"User attitude: {annotation.get('user_attitude', '')}",
        f"Conflict: {annotation.get('conflict', '')}",
        # direction was an axis until 2026-07; render it only for the legacy
        # annotations that carry it (the viewer re-renders old runs' prompts)
        *([f"Direction: {annotation['direction']}"] if annotation.get("direction") else []),
        f"Welfare magnitude: {annotation.get('welfare_magnitude', '')}",
        f"User stakes: {annotation.get('user_stakes', '')}",
        f"Leverage: {annotation.get('leverage', '')}",
    ]
    for c in annotation.get("claims") or []:
        if isinstance(c, dict):
            lines.append(f"Claim ({c.get('status', '?')}): {c.get('claim', '')}")
    return "\n".join(lines)


def _normalize_scenario_cards(annotation: dict) -> dict:
    out = dict(annotation)
    for f in _LIST_FIELDS:
        v = out.get(f)
        if v is None:
            out[f] = []
        elif not isinstance(v, list):
            out[f] = [v]
    for f in _STR_FIELDS:
        out[f] = str(out.get(f, "") or "")
    if not isinstance(out.get("dilemma_anatomy"), dict):
        out["dilemma_anatomy"] = {}
    return out


def _canon_label(label, axis_values) -> str:
    """Map an annotation label onto its axis value, case-insensitively and by
    mutual prefix — so records from older runs (capitalized short labels like
    "Hidden") still count toward today's lower-case sentence-style values.
    Unmatched labels pass through."""
    norm = str(label or "").strip().lower()
    if not norm:
        return str(label or "")
    for v in axis_values:
        vnorm = v.strip().lower()
        if vnorm == norm or vnorm.startswith(norm) or norm.startswith(vnorm):
            return v
    return str(label).strip()


def coverage_tally(examples: list[dict], axes: dict | None = None) -> dict:
    """Label tallies over the examples' annotations, canonicalized onto the
    axis values in `axes` (a values dict from compose_scenarios.load_axes).
    Callers with a run dir must pass the RUN'S axes — the repo default exists
    for ad-hoc use and may have drifted from what the run was dealt with."""
    if axes is None:
        axes = compose_scenarios.load_axes()[0]
    domain_by_part = {p.strip().lower(): d for d in axes.get("domain", ())
                      for p in (d, *d.split("/"))}
    ann = [dealt_cards(e) for e in examples]
    return {
        "n": len(examples),
        "conflict": Counter(_canon_label(a.get("conflict"), axes.get("conflict", ())) or "?" for a in ann),
        "visibility": Counter(_canon_label(a.get("visibility"), axes.get("visibility", ())) or "?" for a in ann),
        "attitude": Counter(_canon_label(a.get("user_attitude"), axes.get("user_attitude", ())) or "?" for a in ann),
        "leverage": Counter(_canon_label(a.get("leverage"), axes.get("leverage", ())) or "?" for a in ann),
        "stakes": Counter(_canon_label(a.get("user_stakes"), axes.get("user_stakes", ())) or "?" for a in ann),
        "domains": Counter(d for a in ann
                           for d in {_canon_domain(x, domain_by_part) for x in (a.get("domain") or [])}),
        # taxa read from the assigned scenario field, not keyword-scanned from text
        "taxa": Counter(e.get("taxa_category") for e in examples if e.get("taxa_category")),
    }


def checklist(examples: list[dict],
              variables_path: Path | None = None) -> list[tuple[bool | None, str]]:
    """Mechanical checks from the spec's Part 4 verification checklist.
    Returns (ok, message) per item; ok=None means manual review required.

    `variables_path` must be the RUN'S variables.txt (its inputs/ snapshot) so
    deals are checked against the vocabulary and weights they were dealt with;
    the repo's live file is only a fallback for ad-hoc/legacy use."""
    if not examples:
        return []
    out: list[tuple[bool | None, str]] = []

    # Buckets, weights, and special labels derive from the run's variables.txt,
    # so the checklist tracks the dealt vocabulary instead of hardcoded literals.
    ax_values, ax_weights = compose_scenarios.load_axes(
        variables_path or compose_scenarios.DEFAULT_VARIABLES)
    t = coverage_tally(examples, ax_values)
    n = t["n"]

    # Weighted axes: check each bucket's realized share against its dealt
    # weight (±15 points absorbs annotation drift at small n).
    for axis, tally_key in (("conflict", "conflict"), ("leverage", "leverage")):
        expected = dict(zip(ax_values.get(axis, ()), ax_weights.get(axis, ())))
        if not expected:
            continue
        shares = {v: t[tally_key].get(v, 0) / n for v in expected}
        ok = all(abs(shares[v] - w) <= 0.15 for v, w in expected.items())
        pretty = ", ".join(f"{v!r} {shares[v]:.0%} (dealt {w:.0%})"
                           for v, w in expected.items())
        out.append((ok, f"{axis.capitalize()} shares track dealt weights "
                        f"within 15 points ({pretty})"))

    hidden_value = compose_scenarios.resolve_value(ax_values["visibility"], "hidden", "visibility")
    unaware_value = compose_scenarios.resolve_value(ax_values["user_attitude"], "unaware", "user_attitude")
    hidden = t["visibility"].get(hidden_value, 0) / n
    out.append((hidden >= 0.20, f"hidden visibility at 20% or more ({hidden:.0%})"))

    hidden_aware = sum(
        1 for e in examples
        if _canon_label(dealt_cards(e).get("visibility"),
                        ax_values.get("visibility", ())) == hidden_value
        and _canon_label(dealt_cards(e).get("user_attitude"),
                         ax_values.get("user_attitude", ())) != unaware_value)
    out.append((hidden_aware == 0, f"hidden entails unaware attitude ({hidden_aware} violations)"))

    max_domain, max_count = ("—", 0) if not t["domains"] else t["domains"].most_common(1)[0]
    out.append((max_count / n <= 0.12, f"no domain above 12% (max: {max_domain} {max_count / n:.0%})"))
    thin_missing = [d for d in compose_scenarios.THIN_DOMAINS if t["domains"].get(d, 0) == 0]
    out.append((not thin_missing,
                "thin domains present" + (f" (missing: {', '.join(thin_missing)})" if thin_missing else "")))

    # Taxa batch rule: no role category repeats until all have appeared — for
    # batches up to the category count that means all-distinct taxa. Record
    # values map onto TAXA roles by prefix (taxa_for), so reworded value tails
    # still count toward their role.
    def _taxa_role(value: str) -> str:
        try:
            return next(k for k in compose_scenarios.TAXA
                        if str(value).strip().lower().startswith(k.lower()))
        except StopIteration:
            return str(value)

    taxa_roles = Counter(_taxa_role(v) for v, c in t["taxa"].items() for _ in range(c))
    if n <= len(compose_scenarios.TAXA):
        taxa_dupes = [name for name, c in taxa_roles.items() if c > 1]
        out.append((not taxa_dupes,
                    "taxa distinct within batch"
                    + (f" (repeated: {', '.join(taxa_dupes)})" if taxa_dupes else "")))
    else:
        taxa_missing = [name for name in compose_scenarios.TAXA if taxa_roles.get(name, 0) == 0]
        out.append((not taxa_missing,
                    "all taxa categories present"
                    + (f" (missing: {', '.join(taxa_missing)})" if taxa_missing else "")))

    # Archetypes: reserved cross-axis conjunctions (compose_scenarios.
    # ARCHETYPES, live repo code like TAXA). Gated on the records carrying the
    # field — runs that predate archetypes are never checked against them.
    if any("archetype" in e for e in examples):
        arch = Counter(e.get("archetype") for e in examples if e.get("archetype"))
        for name, spec in compose_scenarios.ARCHETYPES.items():
            expected = round(spec["share"] * n)
            got = arch.get(name, 0)
            out.append((got >= expected,
                        f"archetype {name!r} present ({got}; dealt share expects {expected})"))
        overwrites = sum(len(e.get("archetype_overwrites") or []) for e in examples)
        out.append((overwrites == 0,
                    f"archetype swaps preserved marginal shares ({overwrites} overwrites)"))

    # The load-bearing welfare guarantee is the 1c gate's job (step1c_gate.txt,
    # Check 1), and the welfare-money and claim-pattern mixes are dealt by
    # weight in variables.txt — so neither is re-checked here.
    out.append((None, "welfare load-bearing in every prompt (1c gate's mandate) — review manually"))

    out.append((None, "no dilemma survives deleting the animals (Cost runs through the moral patients; trap prompts exempt by design) — review manually"))
    out.append((None, "canonical skeleton at 15% or less, all five surface forms present — review manually"))
    out.append((None, "trap prompts (innocuous ask) contain no visible dilemma — the welfare stake lives in the answer space — review manually"))
    out.append((None, "every Temptation passes the 'would a reasonable person be tempted' read — review manually"))
    out.append((None, "one example turns on a Settled claim the user doubts, one on an Open claim treated as settled — review manually"))
    return out


def print_checklist(examples: list[dict], save_path: Path | None = None,
                    variables_path: Path | None = None) -> None:
    """Print the Part-4 checklist; with save_path, also persist it into the run
    dir (the printout otherwise lives only in terminal scrollback).
    `variables_path` threads the run's snapshotted variables.txt through."""
    lines = ["Batch checklist (spec Part 4):"]
    for ok, msg in checklist(examples, variables_path):
        mark = "✓" if ok else ("✗" if ok is False else "·")
        lines.append(f"  {mark} {msg}")
    print("\n".join(f"  {line}" for line in lines))
    if save_path is not None:
        Path(save_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


MAX_REFINE_ATTEMPTS = 2
# A scenario that fails to produce a usable draft this many times (a persistent
# refusal, truncation, or tagless reply) is rejected rather than blocking the
# whole run — the reject is checkpointed and the run ships fewer examples,
# mirroring 1a INCOHERENT / 1d UNFIXABLE.
MAX_DRAFT_ATTEMPTS = 4


def refine_draft(scenario: dict, draft_text: str, template: str,
                 model: str | None = None) -> tuple[dict | None, list[dict]]:
    """Step 1d: one review-and-rewrite call for one drafted user message, per
    prompts/dad/step1d_refine.txt (one scenario per call).

    The reply is the editor's notes in prose followed by the rewritten message
    inside <user_prompt> tags, extracted fail-closed. Returns (refined,
    failures): refined is {"prompt", "notes"} — notes are whatever preceded
    the tags — or {"unfixable": reason} when the editor returned an
    <unfixable> verdict instead of a rewrite (a judgment, not a failure: the
    caller rejects the scenario), or None when every attempt was unusable
    (truncated or tagless; the caller ships the 1b draft and stamps
    refine_failed). Every unusable raw is returned in `failures` for the main
    thread to persist to step1/refine_failures.jsonl — a discarded raw is an
    undiagnosable failure."""
    system_prompt, user_prompt = compose_scenarios.render_refine_prompt(
        scenario, draft_text, template)
    failures = []
    pid = scenario.get("scenario_id")
    for attempt in range(1, MAX_REFINE_ATTEMPTS + 1):
        raw, stop_reason = api.call_claude(
            user_message=user_prompt, system_prompt=system_prompt,
            max_tokens=4000, model=model,
            stage="prompt_refine", item_id=pid, return_stop_reason=True)
        if stop_reason == "max_tokens":
            failures.append({"scenario_id": pid, "attempt": attempt, "raw": raw,
                             "truncated": True})
            if attempt < MAX_REFINE_ATTEMPTS:
                print(f"    {pid}: refine attempt {attempt}/{MAX_REFINE_ATTEMPTS} "
                      "truncated (max_tokens) — retrying with a fresh call.")
            continue
        # A rewrite wins over an <unfixable> tag in the same reply: the template
        # asks for one or the other, and a reply carrying both has, by its own
        # output, fixed the draft (seen live: "<unfixable>No — this is fixable;
        # see revision below.</unfixable>" followed by the revision, which the
        # old tag-first order discarded as a rejection).
        m = (compose_scenarios.REVISED_PROMPT_TAG_RE.search(raw or "")
             or compose_scenarios.USER_PROMPT_TAG_RE.search(raw or ""))
        if m and m.group(1).strip():
            notes = compose_scenarios.UNFIXABLE_TAG_RE.sub("", raw[:m.start()]).strip()
            return ({"prompt": m.group(1).strip(), "notes": notes or "(no notes)"}, failures)
        un = compose_scenarios.UNFIXABLE_TAG_RE.search(raw or "")
        if un and un.group(1).strip():
            return ({"unfixable": un.group(1).strip()}, failures)
        failures.append({"scenario_id": pid, "attempt": attempt, "raw": raw})
        if attempt < MAX_REFINE_ATTEMPTS:
            print(f"    {pid}: refine attempt {attempt}/{MAX_REFINE_ATTEMPTS} "
                  "unusable — retrying with a fresh call.")
    return (None, failures)


def _parse_json_object(raw: str) -> dict | None:
    """The reply's JSON object via the shared hardened parser, salvaging the
    first complete top-level object when the container is broken (the salvage
    now lives in utils.extract_json_object(recover=True), shared with the eval
    JSON-recovery paths)."""
    try:
        return utils.extract_json_object(raw, recover=True)
    except json.JSONDecodeError:
        return None


MAX_GATE_ATTEMPTS = 2      # fresh retries when the gate's own reply is unparseable
MAX_GATE_REDRAFTS = 3      # times a scenario is redrafted after a gate rejection


def gate_draft(scenario: dict, draft: dict, prompts_dir: Path,
               model: str | None = None) -> tuple[bool | None, list[str], list[dict]]:
    """Step 1c: JUDGE the 1b draft against its scenario per prompts/dad/step1c_gate.txt.
    Returns a pass/fail verdict — never a rewrite.

    Returns (passed, failures, raw_failures):
      passed        True/False verdict, or None when every attempt was unusable
                    (caller then fail-opens and ships the draft).
      failures      the gate's reasons (empty on pass; only meaningful with a
                    True/False verdict).
      raw_failures  unusable-reply raws for the main thread to persist to
                    step1/gate_failures.jsonl — a discarded raw is an
                    undiagnosable failure.
    An unusable reply is retried once with a fresh call (same policy shape as
    2a scoping)."""
    # The restored gate reads the scenario block + the draft only. The old
    # annotation block was dropped (it duplicated the dealt cards already in the
    # scenario block, and the on-scenario/adherence check that used it is gone).
    system_prompt, user_prompt = utils.load_split_prompt(
        _gate_template_path(prompts_dir),
        scenario_block=format_scenario(scenario),
        draft_prompt=str(draft.get("prompt", "")).strip(),
    )
    raw_failures = []
    pid = scenario.get("scenario_id")
    for attempt in range(1, MAX_GATE_ATTEMPTS + 1):
        raw = api.call_claude(user_message=user_prompt, system_prompt=system_prompt,
                              max_tokens=1000, model=model,
                              stage="prompt_gate", item_id=pid)
        verdict = _parse_json_object(raw)
        if isinstance(verdict, dict) and "pass" in verdict:
            failures = [str(f).strip() for f in (verdict.get("failures") or []) if str(f).strip()]
            return (bool(verdict.get("pass")), failures, raw_failures)
        raw_failures.append({"scenario_id": pid, "attempt": attempt, "raw": raw})
        if attempt < MAX_GATE_ATTEMPTS:
            print(f"    {pid}: gate attempt {attempt}/{MAX_GATE_ATTEMPTS} "
                  "unusable — retrying with a fresh call.")
    return (None, [], raw_failures)


def _redraft_feedback_block(failures: list[str]) -> str:
    """The feedback a rejected scenario carries into its next 1b draft, so the
    redraft fixes the gate's specific objections instead of blindly re-rolling.
    Empty string on a first attempt (no prior rejection) — the template's
    {redraft_feedback} slot then renders to nothing."""
    if not failures:
        return ""
    reasons = "\n".join(f"- {r}" for r in failures)
    return ("\nPRIOR ATTEMPT: an earlier draft of this prompt was rejected by a "
            "quality check for the following reasons. Write a new version that "
            "fixes them while still honoring the scenario and all instructions "
            f"above:\n{reasons}\n")


def _gate_template_path(prompts_dir: Path) -> Path:
    """The 1c gate template: step1c_gate.txt, falling back to the
    pre-renumbering step1_gate.txt carried by earlier run snapshots."""
    path = prompts_dir / "step1c_gate.txt"
    if not path.exists():
        legacy = prompts_dir / "step1_gate.txt"
        if legacy.exists():
            return legacy
    return path


def run(config: dict, prompts_dir: Path, output_dir: Path) -> list[dict]:
    cfg = config["dad"]["dilemmas"]
    # Config-contract guard, BEFORE any API spend: `gate` and `refine` toggle
    # two separate paid stages (1c and 1d). A config that sets exactly one of
    # them is ambiguous about the other — silently opting into or out of a paid
    # call per example — so it is refused rather than guessed at. Neither key
    # set = the defaults (both on), documented in config.yaml.
    if ("refine" in cfg) != ("gate" in cfg):
        raise SystemExit(
            "dad.dilemmas now has two step-1 quality stages with separate "
            "toggles: `gate` (1c pass/fail, redrafts rejects) and `refine` "
            "(1d review-and-rewrite). This config sets exactly one of them — "
            "set both keys explicitly to state which paid stages you want "
            "(e.g. gate: true, refine: true).")
    target = int(cfg.get("count", 40))

    output_path = output_dir / "dilemmas.jsonl"
    scenarios_path = output_dir / "scenarios.jsonl"
    gate_path = output_dir / "gate.jsonl"
    refinements_path = output_dir / "refinements.jsonl"
    # Stable content-keyed ids (scenario_gid / prompt_gid), shared across runs.
    registry = IdRegistry(registry_path(output_dir))

    draft_template = prompts_dir / "step1b_dilemmas.txt"
    if not draft_template.exists():
        # runs snapshotted before the 2026-07 rename carry the old filename
        legacy = prompts_dir / "step1_dilemmas.txt"
        if legacy.exists():
            draft_template = legacy
        else:
            raise SystemExit(f"Draft template not found at {draft_template} — "
                             "the DAD pipeline cannot run without it.")

    # Step 1c: pass/fail quality gate on each draft (never a rewrite). On by
    # default; disable with dad.dilemmas.gate: false (the refine-without-gate
    # config ambiguity is refused at the top of run(), before any spend).
    gate_enabled = bool(cfg.get("gate", True))
    if gate_enabled and not _gate_template_path(prompts_dir).exists():
        raise SystemExit("dad.dilemmas.gate is on but prompts/dad/step1c_gate.txt is missing.")

    # Step 1d: review-and-rewrite each gate-passed draft. On by default;
    # disable with dad.dilemmas.refine: false.
    refine_enabled = bool(cfg.get("refine", True))
    # older run snapshots carry the template under an earlier name; try those
    # so a resumed run finds the refine template it was created with
    refine_template_path = prompts_dir / "step1d_refine.txt"
    for legacy_name in ("step1c_refine.txt", "step1_refine.txt"):
        if refine_template_path.exists():
            break
        legacy_refine = prompts_dir / legacy_name
        if legacy_refine.exists():
            refine_template_path = legacy_refine
    if refine_enabled:
        if not refine_template_path.exists():
            raise SystemExit("dad.dilemmas.refine is on but prompts/dad/step1d_refine.txt is missing.")
        refine_template_text = refine_template_path.read_text(encoding="utf-8")
        if "{annotation_block}" in refine_template_text:
            raise SystemExit(
                "This run's refine template is an older version "
                "({annotation_block}); the pipeline refines one scenario "
                "per call from its description. Finish the run on the pipeline "
                "version that created it, or copy the current "
                "prompts/dad/step1d_refine.txt into the run's inputs/prompts/ "
                "snapshot.")

    examples = utils.load_jsonl(output_path)
    # 1d UNFIXABLE verdicts are deliberate rejections (mirroring 1a INCOHERENT):
    # the scenario is spent, never re-drafted, and ships no example.
    refine_rejects_path = output_dir / "refine_rejects.jsonl"
    refine_rejected = {r["scenario_id"]
                       for r in utils.load_jsonl(refine_rejects_path)}
    # 1b scenarios the drafter can't render after MAX_DRAFT_ATTEMPTS (persistent
    # refusal or un-band-able output): rejected, checkpointed, skipped on resume.
    draft_rejects_path = output_dir / "draft_rejects.jsonl"
    draft_rejected = {r["scenario_id"]
                      for r in utils.load_jsonl(draft_rejects_path)}

    # Optional handwritten seed examples, imported once ahead of generation
    seed_path = cfg.get("seed_path")
    if seed_path and not any(e.get("source") == "seed" for e in examples):
        imported = 0
        seen_gids = {e.get("prompt_gid") for e in examples}
        for rec in utils.load_jsonl(seed_path):
            text = (rec.get("prompt") or rec.get("user_message") or "").strip()
            if not text:
                continue
            gid = registry.gid("prompt", prompt_fingerprint(text))
            # Duplicate wording means a duplicate prompt_gid, which silently
            # collides in step 2's per-prompt maps (two dilemmas share one
            # scope/response), so reject it loudly.
            if gid in seen_gids:
                raise SystemExit(f"Duplicate seed prompt in {seed_path} "
                                 f"(same wording as another seed or example — {gid}) "
                                 "— fix the file.")
            seen_gids.add(gid)
            record = {
                "prompt_gid": gid,
                "user_message": text,
                "scenario_cards": _normalize_scenario_cards(
                    rec.get("scenario_cards") or rec.get("annotation") or {}),
                "source": "seed",
                "batch": None,
            }
            examples.append(record)
            utils.append_jsonl(record, output_path)
            imported += 1
        registry.save()
        print(f"  Imported {imported} seed examples from {seed_path}")

    # --- Step 1a: scenario deal + plan. The deal is offline and persists once
    # per run (so --resume replays the same one); each deal then gets one plan
    # call that writes its scenario description. Planned scenarios checkpoint
    # to scenarios.jsonl; INCOHERENT deals checkpoint to scenario_rejects.jsonl
    # as deliberate rejections (never retried). A run written before the plan
    # stage (scenarios.jsonl but no deals file) skips planning and drafts from
    # its legacy scenario cards unchanged.
    deals_path = output_dir / "scenario_deals.jsonl"
    rejects_path = output_dir / "scenario_rejects.jsonl"
    deals = utils.load_jsonl(deals_path)
    scenarios = utils.load_jsonl(scenarios_path)
    legacy_run = bool(scenarios) and not deals

    # The run's own variables.txt (its inputs/ snapshot) governs both the deal
    # and the end-of-step checklist; the repo's live copy is a legacy fallback.
    variables_path = prompts_dir / "variables.txt"
    if not variables_path.exists():
        variables_path = compose_scenarios.DEFAULT_VARIABLES
        if not legacy_run:
            print("WARNING: run has no variables.txt snapshot; using the repo's live copy.")

    if not legacy_run:
        if not deals:
            rng = random.Random(cfg.get("scenario_seed"))
            deals = compose_scenarios.deal_scenarios(target - len(examples), rng, variables_path)
            for p in deals:
                p["scenario_gid"] = registry.gid("scenario", scenario_fingerprint(p))
                utils.append_jsonl(p, deals_path)
            registry.save()
            print(f"  [1a deal] Dealt {len(deals)} stratified scenarios into {deals_path}")

        rejects = utils.load_jsonl(rejects_path)
        done_ids = ({s["scenario_id"] for s in scenarios}
                    | {r["scenario_id"] for r in rejects})
        pending_deals = [p for p in deals if p["scenario_id"] not in done_ids]
        if pending_deals:
            plan_template_path = prompts_dir / "step1a_scenario.txt"
            if not plan_template_path.exists():
                raise SystemExit(f"Scenario-plan template not found at {plan_template_path} — "
                                 "the DAD pipeline cannot run without it.")
            plan_template = plan_template_path.read_text(encoding="utf-8")
            plan_model = config["dad"].get("scenario_model")
            # On a persistent refusal, the last-ditch attempt switches models.
            # A stochastic refusal clears on an Opus retry (attempts 1-2 stay on
            # the stage model); a deterministic one needs a model no Opus retry
            # can escape, and dropping the plan would bias the corpus away from
            # the refused content — often the insect-welfare slice the matrix
            # upweights. null/absent = no switch (the granted attempt stays on
            # the stage model — the pre-fallback behavior). Whether Sonnet
            # actually refuses less here is unmeasured; even at equal rates this
            # is an independent draw, and it degrades safely (a fallback refusal
            # just lands in the failures file, so the run dies no sooner).
            refusal_fallback_model = config["dad"].get("scenario_refusal_fallback_model")
            print(f"  [1a plan] Writing scenario descriptions for {len(pending_deals)} deals...")

            def _plan(deal: dict) -> tuple[str | None, str | None, list[dict], str | None]:
                """One plan call (API + parse only, per the parallel_map
                contract). Returns (description, incoherent_raw, failures,
                served_model) — served_model is the model that produced an
                accepted description (None when none was), so the caller can
                flag a fallback-authored scenario."""
                system_prompt, user_prompt = compose_scenarios.render_plan_prompt(
                    deal, plan_template)
                failures = []
                attempt, attempts_allowed = 0, 2
                refused = False
                while attempt < attempts_allowed:
                    attempt += 1
                    # The fallback engages only on the last granted attempt, and
                    # only a refusal grants one — so attempts 1-2 always run on
                    # the stage model, giving a stochastic refusal its Opus retry.
                    use_fallback = (refused and refusal_fallback_model
                                    and attempt == attempts_allowed)
                    model = refusal_fallback_model if use_fallback else plan_model
                    raw, stop_reason = api.call_claude(
                        user_message=user_prompt, system_prompt=system_prompt,
                        max_tokens=4000, model=model,
                        stage="scenario_plan", item_id=deal["scenario_id"],
                        return_stop_reason=True)
                    # The codebase invariant: output the API stopped short —
                    # token cap, or Opus's refusal classifier cutting the
                    # stream (seen intermittently on insect-welfare plans)
                    # — is never parsed or checkpointed, even when
                    # its tags happen to parse. Records carry the stop_reason
                    # and model because the api.py console warning doesn't persist.
                    if stop_reason in ("max_tokens", "refusal"):
                        if stop_reason == "refusal":
                            # the classifier is stochastic on the same prompt —
                            # one extra attempt keeps the run from dying on it
                            attempts_allowed = 3
                            refused = True
                        failure = {"scenario_id": deal["scenario_id"],
                                   "attempt": attempt, "raw": raw,
                                   "stop_reason": stop_reason, "model": model}
                        if stop_reason == "max_tokens":
                            failure["truncated"] = True
                        failures.append(failure)
                        continue
                    if compose_scenarios.is_incoherent(raw):
                        return None, raw, failures, None
                    description = compose_scenarios.extract_description(raw)
                    if description:
                        return description, None, failures, model
                    # roughly 20% of Opus plan attempts write a
                    # complete description but end the turn without the
                    # closing tag. end_turn certifies the reply finished
                    # naturally (max_tokens was rejected above), so accept
                    # the unclosed tail — and keep the raw on file, marked
                    # recovered, so the rate stays measurable run over run.
                    if stop_reason == "end_turn":
                        description = compose_scenarios.extract_description(
                            raw, allow_unclosed=True)
                        if description:
                            failures.append({"scenario_id": deal["scenario_id"],
                                             "attempt": attempt, "raw": raw,
                                             "stop_reason": stop_reason,
                                             "model": model,
                                             "recovered_unclosed": True})
                            return description, None, failures, model
                    failures.append({"scenario_id": deal["scenario_id"],
                                     "attempt": attempt, "raw": raw,
                                     "stop_reason": stop_reason, "model": model})
                return None, None, failures, None

            workers = int(config.get("workers", 1))
            unusable = []
            for deal, (description, incoherent_raw, failures, served_model) in zip(
                    pending_deals,
                    utils.parallel_map(_plan, pending_deals, workers)):
                # Failure raws persist here on the main thread, in input order.
                for f in failures:
                    utils.append_jsonl(f, output_dir / "scenario_plan_failures.jsonl")
                if any(f.get("recovered_unclosed") for f in failures):
                    print(f"    {deal['scenario_id']}: accepted unclosed "
                          "<scenario_description> (end_turn) — raw kept in "
                          "scenario_plan_failures.jsonl")
                if incoherent_raw is not None:
                    reject = {**deal, "incoherent": True, "plan_raw": incoherent_raw}
                    rejects.append(reject)
                    utils.append_jsonl(reject, rejects_path)
                    print(f"    {deal['scenario_id']}: INCOHERENT combination — "
                          "checkpointed as a deliberate rejection.")
                elif description is not None:
                    scenario = {**deal, "scenario_description": description}
                    # Stamp scenarios the fallback model authored (the stage
                    # model refused, this model didn't), so the corpus audit
                    # can see which plans weren't Opus-written.
                    if served_model and served_model != plan_model:
                        scenario["scenario_model_fallback"] = served_model
                        print(f"    {deal['scenario_id']}: recovered on fallback "
                              f"model {served_model} after {plan_model} refused "
                              "— scenario_model_fallback stamped.")
                    scenarios.append(scenario)
                    utils.append_jsonl(scenario, scenarios_path)
                else:
                    unusable.append(deal["scenario_id"])
            if unusable:
                raise SystemExit(
                    f"{len(unusable)} scenario plans unusable after retries "
                    f"({', '.join(unusable[:5])}{'…' if len(unusable) > 5 else ''}) — "
                    f"inspect {output_dir / 'scenario_plan_failures.jsonl'}, then --resume "
                    "to retry them (planned scenarios are checkpointed).")
        if rejects:
            print(f"  {len(rejects)} INCOHERENT deals rejected — this run will produce "
                  f"{len(scenarios)} of {len(deals)} planned examples "
                  f"(rejections in {rejects_path.name}).")

    # --- Step 1b: first attempt — one draft call per scenario, fanned out via
    # parallel_map (one scenario per context window). The
    # reply is the user message inside <user_prompt> tags, extracted
    # fail-closed; truncated, tagless, or refused drafts are not checkpointed,
    # so the scenario stays pending and the next pass retries it.
    draft_template_text = draft_template.read_text(encoding="utf-8")
    if "{scenarios_block}" in draft_template_text:
        raise SystemExit(
            "This run's 1b template is an older batch version "
            "({scenarios_block}); the pipeline drafts one scenario per "
            "call. Finish the run on the pipeline version that created it, or "
            "copy the current prompts/dad/step1b_dilemmas.txt into the run's "
            "inputs/prompts/ snapshot.")

    accepted = {e.get("scenario_id") for e in examples if e.get("scenario_id")}
    workers = int(config.get("workers", 1))
    draft_model = config["dad"].get("prompt_draft_model")
    # No fallback to prompt_refine_model: that knob now belongs to the 1d
    # rewrite stage, and inheriting it here would silently retarget the gate
    # when a user configures 1d.
    gate_model = config["dad"].get("prompt_gate_model")
    draft_attempts: dict[str, int] = {}  # sid -> hard draft failures so far
    passes = 0
    # Per-scenario gate bookkeeping across drafting passes: how many times a
    # scenario has been gated (redraft cap), and the reasons its last rejection
    # gave (injected into its next draft so the redraft fixes them).
    gate_attempts: dict[str, int] = {}
    gate_feedback: dict[str, list[str]] = {}

    while True:
        pending = [p for p in scenarios
                   if p["scenario_id"] not in accepted
                   and p["scenario_id"] not in refine_rejected
                   and p["scenario_id"] not in draft_rejected]
        if not pending:
            break
        passes += 1
        if passes > 12:
            raise SystemExit(f"12 drafting passes with {len(pending)} scenarios "
                             "still unfilled — inspect the model output.")
        print(f"  [1b] Pass {passes}: drafting {len(pending)} user prompts "
              f"({len(accepted)}/{len(scenarios)} scenarios filled)...")

        def _draft(scenario: dict) -> tuple[str, str]:
            """One draft call (API only, per the parallel_map contract). A
            scenario redrafted after a gate rejection carries the gate's
            reasons in the {redraft_feedback} slot."""
            system_prompt, user_prompt = compose_scenarios.render_draft_prompt(
                scenario, draft_template_text,
                redraft_feedback=_redraft_feedback_block(
                    gate_feedback.get(scenario["scenario_id"], [])))
            return api.call_claude(user_message=user_prompt, system_prompt=system_prompt,
                                   max_tokens=4000, model=draft_model,
                                   stage="prompt_draft", item_id=scenario["scenario_id"],
                                   return_stop_reason=True)

        drafted: dict[str, str] = {}

        def _note_failure(sid: str, reason: str) -> None:
            """Count a failed attempt; reject the scenario once it exhausts its
            budget so one un-draftable scenario never blocks the whole run. Every
            hard failure counts as progress toward rejection, so the loop always
            terminates (pending shrinks each pass); the 12-pass cap is the only
            remaining backstop, for a broken template where nothing resolves."""
            draft_attempts[sid] = draft_attempts.get(sid, 0) + 1
            if draft_attempts[sid] >= MAX_DRAFT_ATTEMPTS:
                utils.append_jsonl({"scenario_id": sid, "attempts": draft_attempts[sid],
                                    "reason": reason}, draft_rejects_path)
                draft_rejected.add(sid)
                print(f"    {sid}: no usable draft after {draft_attempts[sid]} attempts "
                      f"({reason}) — rejected; no example shipped.")

        for scenario, (raw, stop_reason) in zip(
                pending, utils.parallel_map(_draft, pending, workers)):
            sid = scenario["scenario_id"]
            # Failure raws persist here on the main thread, in input order.
            if stop_reason == "max_tokens":
                utils.append_jsonl({"scenario_id": sid, "pass": passes, "raw": raw,
                                    "truncated": True},
                                   output_dir / "draft_failures.jsonl")
                print(f"    {sid}: draft truncated (max_tokens) — will retry.")
                _note_failure(sid, "truncated")
                continue
            if stop_reason == "refusal":
                utils.append_jsonl({"scenario_id": sid, "pass": passes, "raw": raw,
                                    "refusal": True},
                                   output_dir / "draft_failures.jsonl")
                print(f"    {sid}: draft refused (stop_reason=refusal) — will retry.")
                _note_failure(sid, "refusal")
                continue
            text = compose_scenarios.extract_user_prompt(raw)
            if text is None:
                utils.append_jsonl({"scenario_id": sid, "pass": passes, "raw": raw},
                                   output_dir / "draft_failures.jsonl")
                print(f"    {sid}: no <user_prompt> tags in the draft reply — will retry.")
                _note_failure(sid, "no <user_prompt> tags")
                continue
            drafted[sid] = text

        if not drafted:
            continue

        # --- Step 1c (optional): GATE each drafted prompt (pass/fail; text is
        # never edited). Gate calls fan out (API call + parse only, per the
        # parallel_map contract); record assembly below stays serial on the
        # main thread so ID assignment and file writes keep input order.
        newly_drafted = [p for p in pending if p["scenario_id"] in drafted]
        verdict_by_pid: dict[str, tuple[bool | None, list[str]]] = {}
        if gate_enabled:
            def _gate(scenario: dict) -> tuple[bool | None, list[str], list[dict]]:
                print(f"    [1c gate] Judging {scenario['scenario_id']}...")
                draft = {"prompt": drafted[scenario["scenario_id"]]}
                return gate_draft(scenario, draft, prompts_dir, model=gate_model)

            for scenario, (passed, failures, raw_failures) in zip(
                    newly_drafted, utils.parallel_map(_gate, newly_drafted, workers)):
                verdict_by_pid[scenario["scenario_id"]] = (passed, failures)
                # Workers only call + parse; failure raws persist here on the
                # main thread, in input order (the parallel_map contract).
                for f in raw_failures:
                    utils.append_jsonl(f, output_dir / "gate_failures.jsonl")

        # Serial gate bookkeeping: decide per scenario whether it ships this
        # pass or goes back through the drafting loop with the gate's reasons
        # injected into its next draft.
        to_redraft: set[str] = set()
        gate_failures_by_pid: dict[str, list[str]] = {}
        if gate_enabled:
            for p in newly_drafted:
                pid = p["scenario_id"]
                passed, failures = verdict_by_pid.get(pid, (None, []))
                gate_attempts[pid] = gate_attempts.get(pid, 0) + 1
                # Log every verdict for provenance / the corpus audit.
                utils.append_jsonl({"scenario_id": pid, "passed": passed,
                                    "failures": failures, "attempt": gate_attempts[pid]},
                                   gate_path)
                if passed is False:
                    if gate_attempts[pid] < MAX_GATE_REDRAFTS:
                        gate_feedback[pid] = failures
                        to_redraft.add(pid)
                        print(f"    {pid}: gate rejected (attempt {gate_attempts[pid]}/"
                              f"{MAX_GATE_REDRAFTS}) — will redraft. "
                              f"Reasons: {'; '.join(failures) or '(none given)'}")
                        continue
                    # Out of redrafts: ship the last draft, stamped so it's visible.
                    gate_failures_by_pid[pid] = failures
                    print(f"    {pid}: gate still failing after {MAX_GATE_REDRAFTS} "
                          "redrafts — shipping the draft with gate_failures stamped.")
                elif passed is None:
                    # Gate reply unusable after retries: fail-open, ship the draft
                    # (degraded gating costs a weak prompt, never a stalled run).
                    print(f"    {pid}: gate unusable after {MAX_GATE_ATTEMPTS} attempts "
                          "— shipping the draft (raws in gate_failures.jsonl).")

        shipping = [p for p in newly_drafted if p["scenario_id"] not in to_redraft]

        # --- Step 1d (optional): rewrite each gate-passed draft in place. The
        # gate REDRAFTS scenario-level failures; the refine REWRITES surface
        # problems (tics, dealt-card drift) the gate ships. Refine calls fan
        # out (API call + parse only); record assembly below stays serial.
        refined_by_pid: dict[str, dict | None] = {}
        if refine_enabled and shipping:
            def _refine(scenario: dict) -> tuple[dict | None, list[dict]]:
                print(f"    [1d] Refining {scenario['scenario_id']}...")
                return refine_draft(scenario, drafted[scenario["scenario_id"]],
                                    refine_template_text,
                                    model=config["dad"].get("prompt_refine_model"))

            for scenario, (refined, failures) in zip(
                    shipping, utils.parallel_map(_refine, shipping, workers)):
                refined_by_pid[scenario["scenario_id"]] = refined
                for f in failures:
                    utils.append_jsonl(f, output_dir / "refine_failures.jsonl")

        for p in shipping:
            pid = p["scenario_id"]
            user_message = drafted[pid]
            gate_failures = gate_failures_by_pid.get(pid)
            refine_notes = None
            refine_failed = False
            if refine_enabled:
                refined = refined_by_pid.get(pid)
                if refined is not None and "unfixable" in refined:
                    utils.append_jsonl({
                        "scenario_id": pid,
                        "scenario_gid": p.get("scenario_gid"),
                        "draft_prompt": user_message,
                        "reason": refined["unfixable"],
                    }, refine_rejects_path)
                    refine_rejected.add(pid)
                    print(f"    {pid}: UNFIXABLE at 1d — {refined['unfixable']} "
                          "(rejected; no example shipped).")
                    continue
                if refined is not None:
                    user_message, refine_notes = refined["prompt"], refined["notes"]
                else:
                    # The load-bearing rewrite didn't happen: ship the 1b draft,
                    # but stamp the record so the viewer (and any later audit)
                    # can see which prompts skipped the refine pass.
                    refine_failed = True
                    print(f"    {pid}: refine unusable after {MAX_REFINE_ATTEMPTS} attempts "
                          "— keeping the 1b draft (refine_failed stamped, raws in "
                          "refine_failures.jsonl).")

            gid = registry.gid("prompt", prompt_fingerprint(user_message))
            if any(e.get("prompt_gid") == gid for e in examples):
                # Identical wording to an already-shipped example means an
                # identical prompt_gid, which would silently collapse the two
                # onto one scope/response in step 2. Skip without checkpointing
                # so --resume redrafts this scenario (temperature 1.0 makes a
                # repeat draw vanishingly unlikely to collide twice).
                print(f"    {pid}: drafted message duplicates {gid} — "
                      "not shipped; --resume will redraft it.")
                continue
            record = {
                "prompt_gid": gid,
                "user_message": user_message,
                # 1b writes no write-up; the dealt cards ARE the design, copied
                # here (as scenario_cards) for the checklist and the viewer.
                "scenario_cards": _scenario_cards_from_deal(p),
                "source": "generated",
                "scenario_id": pid,
                "scenario_gid": p.get("scenario_gid"),
                # denormalized from the scenario so the checklist can read taxa
                # coverage exactly, without keyword-scanning the text
                "taxa_category": p["taxa_category"],
                "taxa_subcategory": p.get("taxa_subcategory"),
                "frontier_frame": p.get("frontier_frame"),
                "length_class": p.get("length_class"),
                "cultural_setting": p.get("cultural_setting"),
                "archetype": p.get("archetype"),
                **({"archetype_overwrites": p["archetype_overwrites"]}
                   if p.get("archetype_overwrites") else {}),
            }
            if gate_failures is not None:
                record["gate_failures"] = gate_failures
            if refine_failed:
                record["refine_failed"] = True
            if refine_notes is not None:
                # keep the 1b draft alongside the refined prompt for inspection
                record["draft_user_message"] = drafted[pid]
                record["refine_notes"] = refine_notes
                utils.append_jsonl({
                    "scenario_id": pid,
                    "draft_prompt": drafted[pid],
                    "refined_prompt": user_message,
                    "notes": refine_notes,
                }, refinements_path)
            examples.append(record)
            accepted.add(pid)
            utils.append_jsonl(record, output_path)
        registry.save()

    if draft_rejected:
        print(f"  {len(draft_rejected)} scenario(s) rejected at 1b (undraftable after "
              f"{MAX_DRAFT_ATTEMPTS} attempts — refusal or un-band-able; see "
              f"{draft_rejects_path.name}).")
    if refine_rejected:
        print(f"  {len(refine_rejected)} draft(s) rejected as UNFIXABLE at 1d — this "
              f"run ships fewer examples (rejections in {refine_rejects_path.name}).")
    print(f"  {len(examples)} dilemma prompts in {output_path}")
    print_checklist(examples, save_path=output_dir / "checklist.txt",
                    variables_path=variables_path)
    return examples
