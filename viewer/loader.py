"""Pure data access for the run viewer. No streamlit imports — reusable from
any frontend (or a future API server)."""

import json
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import utils

REPO_ROOT = Path(__file__).parent.parent
OUTPUTS_ROOT = Path(os.environ.get("PIPELINE_OUTPUTS_ROOT", REPO_ROOT / "outputs"))

PIPELINES = ("sdf", "dad")

# A stage maps to one relative path, or to several tried in order — the first
# that exists wins. SDF needs the fallbacks because the layers were renumbered
# from 1-2/3/4/5 to 1/2/3/4 and runs made before that keep the old directories.
#
# SDF stages are keyed by NAME, not number, and resolved by the output
# FILENAME, because the numbers moved and three layouts now overlap on disk:
# "layer3" is drafts in a pre-renumber run but rewrites in a current one, and
# "layer1"/"layer2" are the legacy pre-matrix pipeline's document types and
# subtypes. Only the filename identifies a stage unambiguously.
STAGE_FILES = {
    "sdf": {
        "dealt": ("layer1/prompts.jsonl", "layer12/prompts.jsonl"),
        "plan": ("layer1/plans.jsonl", "layer12/plans.jsonl"),
        "draft": ("layer2/drafts.jsonl", "layer3/drafts.jsonl"),
        "rewrite": ("layer3/rewrites.jsonl", "layer4/rewrites.jsonl"),
        "score": ("layer4/scores.jsonl", "layer5/scores.jsonl"),
        # Legacy pre-matrix pipeline (two LLM layers stood where layer 1 is now)
        "document_types": "layer1/document_types.jsonl",
        "subtypes": "layer2/subtypes.jsonl",
        "final": "final/sdf_corpus.jsonl",
    },
    "dad": {
        # Current spec-driven pipeline (steps 1-4)
        "step1_scenarios": "step1/scenarios.jsonl",
        "step1_dilemmas": "step1/dilemmas.jsonl",
        "step1_batches": "step1/batches.jsonl",
        "step1_gate": "step1/gate.jsonl",
        "step2_scopes": "step2/scopes.jsonl",
        "step2_tensions": "step2/tensions.jsonl",
        "step2_responses": "step2/responses.jsonl",
        "step3_rewrites": "step3/rewrites.jsonl",
        "baseline": "baseline/baseline_responses.jsonl",
        # Legacy 7-step pipeline (runs made before the dilemma spec)
        "step1": "step1/principles.jsonl",
        "step2": "step2/scenarios.jsonl",
        "step3": "step3/prompts.jsonl",
        "step4": "step4/refined_prompts.jsonl",
        "step5": "step5/responses.jsonl",
        "step6": "step6/rewrites.jsonl",
        "final": "final/dad_corpus.jsonl",
    },
}


def dad_is_legacy(run_dir: Path) -> bool:
    """Old 7-step DAD runs are recognized by their stage-1/2 output files."""
    run_dir = Path(run_dir)
    return (run_dir / "step1" / "principles.jsonl").exists() or \
           (run_dir / "step2" / "scenarios.jsonl").exists()


def doc_first_line(content: str) -> str:
    """First meaningful line of a document/message, stripped of markdown markers."""
    for line in (content or "").splitlines():
        line = line.strip().lstrip("#").strip().strip("*").strip()
        if line:
            return line[:90]
    return "(untitled)"


def dad_goal_label(annotation: dict | None, fallback_text: str) -> str:
    """Label a DAD record by its annotated goal — the one-line 'what is being
    decided' from the 1b anatomy — falling back to the user message's opening
    when absent (seed prompts, runs predating the anatomy fields)."""
    goal = str(((annotation or {}).get("dilemma_anatomy") or {}).get("goal") or "").strip()
    return goal[:90] if goal else doc_first_line(fallback_text)


def run_has_scenario_ids(run_dir: Path) -> bool:
    """True when this run's dilemma records carry a stable scenario_gid — the
    anchor that lets the compare page pair prompts across runs by scenario even
    when the prompt text itself differs (prompt-optimization mode)."""
    return any(d.get("scenario_gid") for d in load_stage(run_dir, "dad", "step1_dilemmas"))


def dad_example_labels(run_dir: Path) -> dict[str, str]:
    """prompt key -> stable example gid (E-####), from the step-3 rewrites.
    The audit report's per_case blocks are keyed by the prompt key; pages use this
    map to display the example id instead. Missing keys (pre-gid runs, or
    prompts whose rewrite failed) fall back to the prompt id at the call site.
    First sample wins when a prompt has several (the audit joins one final
    response per prompt the same way)."""
    out: dict[str, str] = {}
    for r in load_stage(run_dir, "dad", "step3_rewrites"):
        pid, gid = _pkey(r), r.get("example_gid")
        if pid and gid:
            out.setdefault(pid, gid)
    return out


@dataclass
class RunInfo:
    pipeline: str
    run_id: str
    run_dir: Path
    label: str | None
    model: str | None
    created_at: str | None
    git_commit: str | None
    git_dirty: bool | None  # None = pre-v2 manifest (unknown)
    has_snapshot: bool
    config: dict
    counts: dict[str, int] = field(default_factory=dict)
    pass_rate: float | None = None
    total_cost: float = 0.0


@lru_cache(maxsize=512)
def _cached_jsonl(path_str: str, mtime: float) -> tuple:
    return tuple(utils.load_jsonl(path_str))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(_cached_jsonl(str(path), path.stat().st_mtime))


def load_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_audit(run_dir: Path) -> dict | None:
    """The corpus-level audit report (audit/audit_report.json) written by
    evals/audit_dad.py / audit_sdf.py, or None when no audit has run."""
    return _load_report_json(run_dir, "audit_report.json")


def load_diversity(run_dir: Path) -> dict | None:
    """The semantic diversity report (audit/diversity_report.json) written by
    evals/diversity.py, or None when it hasn't run."""
    return _load_report_json(run_dir, "diversity_report.json")


def _load_report_json(run_dir: Path, name: str) -> dict | None:
    path = Path(run_dir) / "audit" / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stage(run_dir: Path, pipeline: str, stage: str) -> list[dict]:
    rel = STAGE_FILES[pipeline].get(stage)
    if rel is None:
        return []
    candidates = (rel,) if isinstance(rel, str) else rel
    run_dir = Path(run_dir)
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return _load_jsonl(path)
    # Nothing on disk: read through the current location so a caller that
    # reports the miss names today's layout rather than a superseded one.
    return _load_jsonl(run_dir / candidates[0])


def load_final(run_dir: Path, pipeline: str) -> list[dict]:
    return load_stage(run_dir, pipeline, "final")


def total_cost(run_dir: Path) -> float:
    total = 0.0
    for rec in _load_jsonl(Path(run_dir) / "cost_log.jsonl"):
        total += rec.get("cost_usd", 0.0)
    return round(total, 4)


def cost_by_stage(run_dir: Path) -> dict[str, dict]:
    """Aggregate the run's cost log per stage tag:
    {stage: {"calls": int, "cost_usd": float, "models": [str]}}.
    Records written before stage tags existed group under "(untagged)"."""
    by_stage: dict[str, dict] = {}
    for rec in _load_jsonl(Path(run_dir) / "cost_log.jsonl"):
        agg = by_stage.setdefault(rec.get("stage") or "(untagged)",
                                  {"calls": 0, "cost_usd": 0.0, "models": set()})
        agg["calls"] += 1
        agg["cost_usd"] += rec.get("cost_usd", 0.0)
        if rec.get("model"):
            agg["models"].add(rec["model"])
    for agg in by_stage.values():
        agg["cost_usd"] = round(agg["cost_usd"], 4)
        agg["models"] = sorted(agg["models"])
    return by_stage


def call_stats(run_dir: Path, stage: str, item_id: str | None = None) -> dict | None:
    """Aggregate the run's cost-log rows for one stage, narrowed to the calls
    that served one item when item_id is given (a logged item_id may be a
    comma-joined id list — one batched call serving several records).

    Returns {"per_item", "calls", "models", "cost_usd", and — when every
    matched row recorded them — "duration_s", "retries", "batch_size"}.
    per_item=False means the rows predate per-item logging and the numbers are
    stage-wide, not this record's. Returns None when nothing matches (stage
    never ran, or per-item logging exists but this item has no calls)."""
    rows = [r for r in _load_jsonl(Path(run_dir) / "cost_log.jsonl")
            if (r.get("stage") or "") == stage]
    if not rows:
        return None
    scoped = rows
    per_item = False
    if item_id is not None:
        scoped = [r for r in rows if item_id in str(r.get("item_id", "")).split(",")]
        per_item = bool(scoped)
        if not scoped:
            if any(r.get("item_id") for r in rows):
                return None  # per-item logging exists; this item made no calls
            scoped = rows  # older run: no item ids recorded — stage-wide fallback
    out = {
        "per_item": per_item,
        "calls": len(scoped),
        "models": sorted({r.get("model", "?") for r in scoped}),
        "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in scoped), 4),
    }
    if all("duration_s" in r for r in scoped):
        out["duration_s"] = round(sum(r["duration_s"] for r in scoped), 1)
    if all("attempts" in r for r in scoped):
        out["retries"] = sum(r["attempts"] - 1 for r in scoped)
    if per_item:
        out["batch_size"] = max(len(str(r.get("item_id", "")).split(",")) for r in scoped)
    return out


def _pass_rate(run_dir: Path, pipeline: str) -> float | None:
    if pipeline == "sdf":
        scored = load_stage(run_dir, pipeline, "score")
        final = load_final(run_dir, pipeline)
        return len(final) / len(scored) if scored else None
    responses = load_stage(run_dir, pipeline, "step2_responses") or load_stage(run_dir, pipeline, "step5")
    if not responses:
        return None
    # `kept` only exists on legacy runs (the ruthless-judge gate that set it was
    # removed). With no keep/score signal, pass rate is n/a rather than a bogus 0%.
    if not any("kept" in r for r in responses):
        return None
    return sum(1 for r in responses if r.get("kept")) / len(responses)


def list_runs(outputs_root: Path = OUTPUTS_ROOT) -> list[RunInfo]:
    """All runs across both pipelines, newest first. A run is a non-symlink
    directory under outputs/<pipeline>/runs/ containing run_manifest.json."""
    runs = []
    for pipeline in PIPELINES:
        runs_root = Path(outputs_root) / pipeline / "runs"
        if not runs_root.is_dir():
            continue
        for d in sorted(runs_root.iterdir(), reverse=True):
            if d.is_symlink() or not d.is_dir():
                continue
            manifest = load_manifest(d)
            if not manifest:
                continue
            # Zero-count stages are dropped: DAD stage keys span both the current
            # and the legacy pipeline layout, and a run only has one of them.
            counts = {
                stage: n
                for stage in STAGE_FILES[pipeline]
                if (n := len(load_stage(d, pipeline, stage))) or stage == "final"
            }
            runs.append(RunInfo(
                pipeline=pipeline,
                run_id=d.name,
                run_dir=d,
                label=manifest.get("label"),
                model=manifest.get("model"),
                created_at=manifest.get("created_at"),
                git_commit=manifest.get("git_commit"),
                git_dirty=manifest.get("git_dirty"),
                has_snapshot=(d / "inputs" / "prompts").is_dir(),
                config=manifest.get("config", {}),
                counts=counts,
                pass_rate=_pass_rate(d, pipeline),
                total_cost=total_cost(d),
            ))
    return runs


def get_run(pipeline: str, run_id: str, outputs_root: Path = OUTPUTS_ROOT) -> RunInfo | None:
    for run in list_runs(outputs_root):
        if run.pipeline == pipeline and run.run_id == run_id:
            return run
    return None


def _index(records: list[dict], key: str) -> dict:
    return {r[key]: r for r in records if key in r}


def _pkey(rec: dict) -> str:
    """The id naming a DAD record's prompt: prompt_gid (P-####) on current
    runs, falling back to the retired per-run prompt_id (AW-####) so legacy
    runs keep rendering."""
    return str(rec.get("prompt_gid") or rec.get("prompt_id") or "")


def _pkeys(rec: dict) -> tuple[str, ...]:
    """Every id naming this record's prompt. Indexes register a record under
    all of them, so mixed-era runs (gid-era step-1 files, pre-gid later stages
    carrying only prompt_id) still join."""
    return tuple(str(v) for v in (rec.get("prompt_gid"), rec.get("prompt_id")) if v)


def _index_by_prompt(records: list[dict]) -> dict:
    return {k: r for r in records for k in _pkeys(r)}


def sdf_lineage(run_dir: Path, doc_id: str) -> dict:
    """Full lineage for one SDF document. Values are None when a stage was
    not reached or the join key is missing."""
    drafts = _index(load_stage(run_dir, "sdf", "draft"), "doc_id")
    rewrites = _index(load_stage(run_dir, "sdf", "rewrite"), "doc_id")
    scores = _index(load_stage(run_dir, "sdf", "score"), "doc_id")
    finals = _index(load_final(run_dir, "sdf"), "doc_id")

    draft = drafts.get(doc_id)
    anchor = draft or rewrites.get(doc_id) or scores.get(doc_id) or finals.get(doc_id) or {}
    subtypes = _index(load_stage(run_dir, "sdf", "subtypes"), "subtype_id")
    doc_types = _index(load_stage(run_dir, "sdf", "document_types"), "type_id")

    return {
        "doc_type": doc_types.get(anchor.get("type_id")),
        "subtype": subtypes.get(anchor.get("subtype_id")),
        "draft": draft,
        "rewrite": rewrites.get(doc_id),
        "score": scores.get(doc_id),
        "final": finals.get(doc_id),
    }


def dad_lineage(run_dir: Path, record_id: str) -> dict:
    """Full lineage for one DAD training record (keyed by final record_id).
    The "format" key tells pages which stage chain this run used."""
    if dad_is_legacy(run_dir):
        return _dad_lineage_legacy(run_dir, record_id)

    final = _index(load_final(run_dir, "dad"), "record_id").get(record_id)
    audits = _index(load_stage(run_dir, "dad", "step3_rewrites"), "record_id")
    audit = audits.get(record_id)
    if audit is None:
        return {"format": "v2", "final": final}

    responses = _index(load_stage(run_dir, "dad", "step2_responses"), "response_id")
    dilemmas = _index_by_prompt(load_stage(run_dir, "dad", "step1_dilemmas"))
    tension_tags = _index_by_prompt(load_stage(run_dir, "dad", "step2_tensions"))
    scenarios = _index(load_stage(run_dir, "dad", "step1_scenarios"), "scenario_id")
    scope_recs = _index_by_prompt(load_stage(run_dir, "dad", "step2_scopes"))
    baselines = _index_by_prompt(load_stage(run_dir, "dad", "baseline"))
    # 1c gate verdicts keyed by scenario_id; _index keeps the LAST row, i.e. the
    # verdict the shipping draft was accepted on (pass, or fail-then-shipped).
    gate_recs = _index(load_stage(run_dir, "dad", "step1_gate"), "scenario_id")

    pid = _pkey(audit)
    dilemma = dilemmas.get(pid)
    return {
        "format": "v2",
        "dilemma": dilemma,
        "scenario": scenarios.get((dilemma or {}).get("scenario_id")),
        "gate": gate_recs.get((dilemma or {}).get("scenario_id")),
        "scope": scope_recs.get(pid),
        "tension_tag": tension_tags.get(pid),
        "response": responses.get(audit.get("response_id")),
        "rewrite": audit,
        "baseline": baselines.get(pid),
        "final": final,
    }


def dad_lineage_by_prompt(run_dir: Path, prompt_id: str) -> dict:
    """Lineage keyed by step-1 prompt_id, built forward through whatever stages
    exist. Used to view incomplete runs (e.g. --stop-after 1, before responses
    are generated); returns the same shape as dad_lineage with later stages None
    when not reached."""
    dilemmas = _index_by_prompt(load_stage(run_dir, "dad", "step1_dilemmas"))
    tension_tags = _index_by_prompt(load_stage(run_dir, "dad", "step2_tensions"))
    scenarios = _index(load_stage(run_dir, "dad", "step1_scenarios"), "scenario_id")
    scope_recs = _index_by_prompt(load_stage(run_dir, "dad", "step2_scopes"))
    dilemma = dilemmas.get(prompt_id)
    # Match later stages by ANY id naming this prompt: on mixed-era runs the
    # dilemma carries a P- gid while responses/rewrites carry only the old
    # per-run id, so the single passed-in key is not enough.
    keyset = set(_pkeys(dilemma or {})) or {prompt_id}
    responses = [r for r in load_stage(run_dir, "dad", "step2_responses")
                 if keyset & set(_pkeys(r))]
    rewrite = next((a for a in load_stage(run_dir, "dad", "step3_rewrites")
                    if keyset & set(_pkeys(a))), None)
    final = None
    if rewrite:
        final = _index(load_final(run_dir, "dad"), "record_id").get(rewrite.get("record_id"))
    baselines = _index_by_prompt(load_stage(run_dir, "dad", "baseline"))
    gate_recs = _index(load_stage(run_dir, "dad", "step1_gate"), "scenario_id")
    def _any(index: dict):
        return next((index[k] for k in keyset if k in index), None)

    return {
        "format": "v2",
        "dilemma": dilemma,
        "scenario": scenarios.get((dilemma or {}).get("scenario_id")),
        "gate": gate_recs.get((dilemma or {}).get("scenario_id")),
        "scope": _any(scope_recs),
        "tension_tag": _any(tension_tags),
        "response": responses[0] if responses else None,
        "rewrite": rewrite,
        "baseline": _any(baselines),
        "final": final,
    }


def _dad_lineage_legacy(run_dir: Path, record_id: str) -> dict:
    audits = _index(load_stage(run_dir, "dad", "step6"), "record_id")
    audit = audits.get(record_id)
    if audit is None:
        return {"format": "legacy",
                "final": _index(load_final(run_dir, "dad"), "record_id").get(record_id)}

    responses = _index(load_stage(run_dir, "dad", "step5"), "response_id")
    refined = _index(load_stage(run_dir, "dad", "step4"), "prompt_id")
    prompts = _index(load_stage(run_dir, "dad", "step3"), "prompt_id")
    scenarios = _index(load_stage(run_dir, "dad", "step2"), "scenario_id")
    principles = _index(load_stage(run_dir, "dad", "step1"), "principle_id")

    return {
        "format": "legacy",
        "principle": principles.get(audit.get("principle_id")),
        "scenario": scenarios.get(audit.get("scenario_id")),
        "prompt": prompts.get(audit.get("prompt_id")),
        "refined": refined.get(audit.get("prompt_id")),
        "response": responses.get(audit.get("response_id")),
        "rewrite": audit,
        "final": _index(load_final(run_dir, "dad"), "record_id").get(record_id),
    }


@dataclass
class MatchedPair:
    key: str
    quality: str  # "exact" | "positional" | "group"
    a: list[dict]
    b: list[dict]


def _sdf_match_key(run_dir: Path, doc: dict) -> tuple[str, str] | None:
    subtypes = _index(load_stage(run_dir, "sdf", "subtypes"), "subtype_id")
    st = subtypes.get(doc.get("subtype_id"))
    if st:
        return (st.get("type_name", ""), st.get("subtype_name", ""))
    return None


def match_outputs(run_a: Path, run_b: Path, pipeline: str) -> list[MatchedPair]:
    """Pair up final outputs of two runs for side-by-side comparison."""
    finals_a = load_final(run_a, pipeline)
    finals_b = load_final(run_b, pipeline)
    pairs: list[MatchedPair] = []

    if pipeline == "sdf":
        def group(run_dir, finals):
            by_name, by_pos = {}, {}
            for doc in finals:
                name_key = _sdf_match_key(run_dir, doc)
                if name_key:
                    by_name.setdefault(name_key, []).append(doc)
                by_pos.setdefault(doc.get("subtype_id"), []).append(doc)
            return by_name, by_pos

        names_a, pos_a = group(run_a, finals_a)
        names_b, pos_b = group(run_b, finals_b)
        matched_b_names = set()
        for key, docs_a in names_a.items():
            if key in names_b:
                matched_b_names.add(key)
                pairs.append(MatchedPair(" / ".join(key), "exact", docs_a, names_b[key]))
        # Positional fallback for name keys that didn't line up
        matched_a_ids = {d.get("subtype_id") for p in pairs for d in p.a}
        for sid, docs_a in pos_a.items():
            if sid in matched_a_ids or sid not in pos_b:
                continue
            pairs.append(MatchedPair(f"subtype_id {sid}", "positional", docs_a, pos_b[sid]))
        return pairs

    # DAD: audits carry prompt/scenario + injection identity
    def group_dad(run_dir, finals):
        exact, grouped = {}, {}
        if dad_is_legacy(run_dir):
            audits = _index(load_stage(run_dir, "dad", "step6"), "record_id")
            for rec in finals:
                audit = audits.get(rec.get("record_id"), {})
                sid = str(audit.get("scenario_id", ""))
                inj = audit.get("injection_used", "")
                if sid.startswith("manta_"):
                    exact.setdefault((sid, inj), []).append(rec)
                else:
                    grouped.setdefault((audit.get("principle_id"), inj), []).append(rec)
            return exact, grouped
        audits = _index(load_stage(run_dir, "dad", "step3_rewrites"), "record_id")
        for rec in finals:
            audit = audits.get(rec.get("record_id"), {})
            # prompt keys are content-stable (P-gids) or positional (legacy AW-)
            exact.setdefault((_pkey(audit), audit.get("sample_index", 0)), []).append(rec)
        return exact, grouped

    exact_a, grouped_a = group_dad(run_a, finals_a)
    exact_b, grouped_b = group_dad(run_b, finals_b)
    for key, recs_a in exact_a.items():
        if key in exact_b:
            pairs.append(MatchedPair(f"{key[0]} [{key[1]}]", "exact", recs_a, exact_b[key]))
    for key, recs_a in grouped_a.items():
        if key in grouped_b:
            pairs.append(MatchedPair(f"principle {key[0]} [{key[1]}]", "group", recs_a, grouped_b[key]))
    return pairs


# --- DAD compare: content-aware matching across runs ---

DAD_MATCH_KEYS = ("user_message", "scenario_id", "prompt_id")


@dataclass
class DadExample:
    prompt_id: str            # the prompt key: P-#### gid (legacy runs: AW-####)
    prompt_gid: str | None    # stable global P-#### (content-keyed, cross-run)
    sample_index: int
    user_message: str
    goal: str                 # dad_goal_label — the human-readable "what's decided"
    scenario_gid: str | None  # stable global S-#### for the underlying scenario
    response: str             # final rewritten response (or draft/rewrite if no final)
    has_final: bool


@dataclass
class DadMatch:
    label: str                # goal (or fallback title)
    same_prompt: bool         # user messages identical after whitespace-normalization
    a: DadExample
    b: DadExample


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _dad_examples(run_dir: Path) -> list[DadExample]:
    """One DadExample per step-3 rewrite audit, joined forward to its final
    training record and back to its dilemma (for scenario_id). Empty for legacy
    7-step runs (different schema; the old positional match_outputs still covers
    them)."""
    if dad_is_legacy(run_dir):
        return []
    dilemmas = _index_by_prompt(load_stage(run_dir, "dad", "step1_dilemmas"))
    finals = _index(load_final(run_dir, "dad"), "record_id")
    out = []
    for audit in load_stage(run_dir, "dad", "step3_rewrites"):
        pid = _pkey(audit)
        user_message = audit.get("user_message", "")
        dilemma = dilemmas.get(pid) or {}
        final = finals.get(audit.get("record_id"))
        if final and final.get("messages"):
            response, has_final = final["messages"][1]["content"], True
        else:  # rewrite failed / no final written — fall back to the audit text
            response = audit.get("rewritten_response") or audit.get("draft_response", "")
            has_final = False
        out.append(DadExample(
            prompt_id=pid,
            prompt_gid=dilemma.get("prompt_gid"),
            sample_index=audit.get("sample_index", 0),
            user_message=user_message,
            goal=dad_goal_label(audit.get("scenario_cards") or audit.get("annotation"), user_message),
            scenario_gid=dilemma.get("scenario_gid"),
            response=response,
            has_final=has_final,
        ))
    return out


def _dad_key(ex: DadExample, key_by: str):
    """Correspondence key for one example, or None if this key can't identify it
    (e.g. scenario_id when the run didn't record one)."""
    if key_by == "scenario_id":
        return (ex.scenario_gid, ex.sample_index) if ex.scenario_gid else None
    if key_by == "prompt_id":
        return (ex.prompt_id, ex.sample_index)
    return (_norm(ex.user_message), ex.sample_index)  # default: user_message


def match_dad(run_a: Path, run_b: Path, key_by: str = "user_message"):
    """Pair DAD examples across two runs by the chosen correspondence key.

    Returns (matched, only_a, only_b). A comparison holds one dimension fixed
    (the key) and diffs the rest:
      - 'user_message' — prompt held fixed → compare responses (response tuning)
      - 'scenario_id'  — scenario held fixed → the prompts themselves can differ,
        so you can compare prompts too (prompt tuning)
      - 'prompt_id'    — the prompt key (P-#### gid; on legacy runs the
        positional AW-####, which may pair unrelated prompts)
    """
    def by_key(run_dir):
        d = {}
        for ex in _dad_examples(run_dir):
            k = _dad_key(ex, key_by)
            if k is not None:
                d.setdefault(k, ex)  # keys are unique in practice (sample_index included)
        return d

    a_by, b_by = by_key(run_a), by_key(run_b)
    matched = [
        DadMatch(label=a.goal or b_by[k].goal,
                 same_prompt=_norm(a.user_message) == _norm(b_by[k].user_message),
                 a=a, b=b_by[k])
        for k, a in a_by.items() if k in b_by
    ]
    only_a = [a for k, a in a_by.items() if k not in b_by]
    only_b = [b for k, b in b_by.items() if k not in a_by]
    return matched, only_a, only_b
