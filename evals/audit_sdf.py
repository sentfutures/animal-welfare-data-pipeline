#!/usr/bin/env python3
"""Corpus-level audit for SDF output: composition, redundancy, and templating.

Per-document judges (layer 4, evals/score_sdf.py) cannot see corpus-level
properties — a corpus can pass every per-doc check and still be 90% one
register, reuse the same invented name everywhere, or open every piece the
same way (exactly what an early quality review of this corpus found). This tool reads
the corpus as a set.

Two tiers:

  Mechanical (default, offline, free): composition spread, length and
  truncation artifacts, near-duplicate rate (word-shingle cosine — lexical,
  not semantic), invented-name collapse, stock-phrase frequency, opening-shape
  clustering, and a first-person register proxy. Each check prints a
  GOOD/OK/BAD verdict where a threshold is meaningful.

  LLM pattern detection (--patterns, costs API calls): batches documents
  through prompts/tools/pattern_scan.txt, consolidates the found patterns,
  then measures each pattern's prevalence across a sample. A pattern is
  flagged RED only if it is judged a genuine templating defect AND is
  widespread (>30%) — prevalence alone is not badness (a broad
  problem-then-response arc is normal writing, not a defect).

Usage:
  python evals/audit_sdf.py                                  # audits outputs/sdf/latest
  python evals/audit_sdf.py --input outputs/sdf/runs/<id>    # a specific run dir
  python evals/audit_sdf.py --input some/corpus.jsonl        # a bare corpus file
  python evals/audit_sdf.py --patterns                       # + LLM templating scan

Writes audit/audit_report.json under the run dir (or next to the corpus file).
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, textstats, utils

# ---------------------------------------------------------------- verdicts


def _verdict(value: float, good: float, ok: float, higher_better: bool = False) -> str:
    if higher_better:
        return "GOOD" if value >= good else ("OK" if value >= ok else "BAD")
    return "GOOD" if value <= good else ("OK" if value <= ok else "BAD")


def _fmt(label: str, value: str, verdict: str | None = None, note: str = "") -> str:
    tail = f"  [{verdict}]" if verdict else ""
    tail += f"  {note}" if note else ""
    return f"   {label:<34} {value}{tail}"


# ---------------------------------------------------------------- input resolution


def resolve_input(input_arg: str) -> tuple[list[dict], Path]:
    """Return (records, report_dir). Accepts a run dir or a JSONL file.

    Composition lives in each record's ``variables`` (the matrix axes) plus the
    top-level ``type_name``/``language``/``register`` the pipeline derives — so
    no separate layer-1 type map is loaded."""
    path = Path(input_arg)
    if path.is_dir():
        corpus = path / "final" / "sdf_corpus.jsonl"
        if not corpus.exists():
            raise SystemExit(f"No final/sdf_corpus.jsonl under {path}")
        return utils.load_jsonl(corpus), path / "audit"
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    return utils.load_jsonl(path), path.parent / "audit"


# ---------------------------------------------------------------- mechanical checks

# Stock phrases the prompts ban; the audit verifies the ban held. English-only
# heuristic — non-English docs simply won't hit them.
STOCK_PHRASES = [
    "evolving landscape", "expanding spheres of moral consideration",
    "advanced computational systems", "optimal outcomes",
    "betterment of all living things", "helpful, harmless, and honest",
    "a testament to", "delve into", "in today's rapidly",
    "at the intersection of", "paradigm shift", "it is worth noting",
    # rewrite-stage tics (found by this audit on the notebook-port smoke run:
    # the layer-3 rewriter injected these across docs that lacked them as drafts)
    "i want to be clear", "to be honest about",
]

# Model-favorite invented names that fingerprint synthetic corpora.
WATCHLIST_NAMES = ["Elara", "Meridian", "Thorne", "Voss", "Kael", "Vance", "Aris", "Solace"]

_OPENER_PATTERNS = [
    (re.compile(r"^the\s+\w+\s+of\b", re.IGNORECASE), "The X of Y ..."),
    (re.compile(r"^in\s+(recent\s+years|an?\s+(era|world|age)|today)", re.IGNORECASE), "In recent years / In an era ..."),
    (re.compile(r"^as\s+(ai|artificial|the|we|our)\b", re.IGNORECASE), "As AI / As the ..."),
    (re.compile(r"^with\s+the\s+(rise|advent|growth)\b", re.IGNORECASE), "With the rise of ..."),
    (re.compile(r"^(amid|across)\b", re.IGNORECASE), "Amid / Across ..."),
]

_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})\b")
_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|i'd|my|mine)\b", re.IGNORECASE)
_CONTRACTION_RE = re.compile(r"\b\w+n't\b|\b\w+'(ll|re|ve|d|s)\b", re.IGNORECASE)

_MD_CLASSES = {
    "# headings": re.compile(r"^#{1,4} ", re.M),
    "**bold**": re.compile(r"\*\*[^*\n]+\*\*"),
    "markdown bullets": re.compile(r"^[-*] ", re.M),
    "tables": re.compile(r"^\|.+\|\s*$", re.M),
}

_NAME_STOPWORDS = {
    "New York", "United States", "European Union", "Cambridge Declaration",
    "York Declaration", "Animal Consciousness", "Middle East", "South America",
    "North America", "United Kingdom", "New Zealand", "Sri Lanka", "Hong Kong",
    "Cambridge", "New York City",
}


def first_sentence(text: str) -> str:
    t = (text or "").strip()
    for line in t.splitlines():
        line = line.strip()
        if line:
            t = line
            break
    m = re.search(r"[.!?]", t)
    return t[: m.end()] if m else t[:160]


# Matrix axes worth a corpus-level readout. The deck-sampler fixes their
# marginals by construction, so this confirms the composition rather than
# detecting drift — but reading them from each record's `variables` (not a
# stale layer-1 type map) keeps the readout honest instead of all-"unknown".
_COMPOSITION_AXES = ("tone", "domain", "centrality")


def audit_composition(records: list[dict], report: dict) -> None:
    n = len(records)
    by = {
        axis: collections.Counter((r.get("variables") or {}).get(axis, "unknown") for r in records)
        for axis in _COMPOSITION_AXES
    }
    by["language"] = collections.Counter(r.get("language", "unknown") for r in records)
    types = collections.Counter(
        r.get("type_name") or (r.get("variables") or {}).get("document_type", "unknown")
        for r in records
    )
    print("COMPOSITION")
    for axis, counter in by.items():
        parts = ", ".join(f"{k}: {v} ({v / n:.0%})" for k, v in counter.most_common())
        print(_fmt(axis, parts))
    top_type, top_n = types.most_common(1)[0]
    if len(types) >= 8:
        v = _verdict(top_n / n, 0.15, 0.30)
        print(_fmt("top document type share", f"{top_n / n:.0%} ({top_type[:50]})", v,
                   "(GOOD <=15%: no genre dominates)"))
    else:
        print(_fmt("document types", f"{len(types)} distinct — too few for a spread verdict (dev-scale run)"))
    report["composition"] = {k: dict(c) for k, c in by.items()}
    report["composition"]["n_types"] = len(types)
    report["composition"]["top_type_share"] = round(top_n / n, 3)


def audit_length_truncation(records: list[dict], report: dict) -> None:
    lengths = sorted(len(r.get("content") or "") for r in records)
    n = len(lengths)
    mid = lengths[n // 2]
    truncated = [r for r in records if textstats.ends_mid_sentence(r.get("content") or "")]
    trailing_sep = [r for r in records if textstats.has_trailing_separator(r.get("content") or "")]
    frac = len(truncated) / n
    v = "GOOD" if not truncated else _verdict(frac, 0.0, 0.02)
    print("LENGTH / TRUNCATION")
    print(_fmt("chars (p10 / median / p90)",
               f"{lengths[max(0, n // 10)]} / {mid} / {lengths[min(n - 1, 9 * n // 10)]}"))
    print(_fmt("ends mid-sentence", f"{len(truncated)} of {n} ({frac:.1%})", v,
               "(token-cap artifacts a trained model would learn)"))
    for r in truncated[:5]:
        print(f"      - {r.get('doc_id', '?')[:12]}: ...{(r.get('content') or '')[-60:]!r}")
    if trailing_sep:
        print(_fmt("trailing separator lines (---)", f"{len(trailing_sep)} of {n}", None,
                   "(delimiter artifact at end of doc — a generator tic worth pruning)"))
    report["length"] = {"median_chars": mid, "truncated": len(truncated),
                        "truncated_frac": round(frac, 4), "trailing_separator": len(trailing_sep)}


def audit_markdown(records: list[dict], report: dict) -> None:
    """Markdown gloss is a strong synthetic tell in prose meant to look scraped —
    with an explicit ban in the prompts, bullets drop to 0% and bold to single
    digits; a comparison corpus generated without the ban hit 52% bold."""
    n = len(records)
    counts = {label: sum(bool(rx.search(r.get("content") or "")) for r in records)
              for label, rx in _MD_CLASSES.items()}
    any_md = sum(
        1 for r in records
        if any(rx.search(r.get("content") or "") for rx in _MD_CLASSES.values())
    )
    bold_frac = counts["**bold**"] / n
    v = _verdict(bold_frac, 0.10, 0.30)
    print("MARKDOWN GLOSS")
    print(_fmt("docs with any markdown", f"{any_md} of {n} ({any_md / n:.0%})"))
    print(_fmt("by class", ", ".join(f"{k}: {c} ({c / n:.0%})" for k, c in counts.items()), v,
               "(verdict on **bold** — the strongest synthetic tell in prose)"))
    report["markdown"] = {"any_frac": round(any_md / n, 3),
                          **{k: round(c / n, 3) for k, c in counts.items()}}


def audit_near_dups(records: list[dict], report: dict, sample: int) -> None:
    texts = [r.get("content") or "" for r in records]
    if len(texts) > sample:
        step = len(texts) / sample
        texts = [texts[int(i * step)] for i in range(sample)]  # deterministic stride sample
    sims = textstats.nearest_neighbor_sims(texts)
    n = max(len(sims), 1)
    over = {t: float((sims > t).sum()) / n for t in (0.80, 0.90, 0.95)}
    v = _verdict(over[0.90], 0.02, 0.08)
    print("REDUNDANCY (word-shingle cosine — lexical, not semantic)")
    print(_fmt("mean nearest-neighbor sim", f"{float(sims.mean()) if len(sims) else 0.0:.3f}"))
    print(_fmt("near-dup >0.80 / >0.90 / >0.95",
               f"{over[0.80]:.1%} / {over[0.90]:.1%} / {over[0.95]:.1%}", v,
               "(verdict on >0.90; catches copied skeletons, not paraphrase)"))
    report["near_dups"] = {str(k): round(val, 4) for k, val in over.items()}


def audit_names(records: list[dict], report: dict) -> None:
    n = len(records)

    def keep_name(nm: str) -> bool:
        # The Cambridge/New York Declarations are the two real references the
        # corpus is allowed to cite (see layer4/layer5); they are not invented
        # names and must not read as collapse.
        if "Declaration" in nm or "Consciousness" in nm:
            return False
        return nm not in _NAME_STOPWORDS

    doc_names = []
    for r in records:
        names = {m.group(1) for m in _NAME_RE.finditer(r.get("content") or "")}
        names = {nm.removeprefix("The ").strip() for nm in names}
        doc_names.append({nm for nm in names if nm and keep_name(nm)})
    df = collections.Counter(nm for names in doc_names for nm in names)
    repeated = [(nm, c) for nm, c in df.most_common(12) if c >= 2]
    watch_hits = {w: sum(1 for names in doc_names if any(w in nm for nm in names)) for w in WATCHLIST_NAMES}
    watch_hits = {w: c for w, c in watch_hits.items() if c}
    worst = max((c for _, c in repeated), default=0)
    v = "GOOD" if worst < max(2, 0.1 * n) else ("OK" if worst <= 0.2 * n else "BAD")
    print("INVENTED-NAME COLLAPSE")
    print(_fmt("names in 2+ docs", ", ".join(f"{nm} ({c})" for nm, c in repeated) or "none", v,
               "(same invented name recurring across docs = generator fingerprint)"))
    if watch_hits:
        print(_fmt("watchlist hits", ", ".join(f"{w} ({c})" for w, c in watch_hits.items()), "BAD",
                   "(model-favorite synthetic names — the prompts ban these)"))
    report["names"] = {"repeated": repeated, "watchlist": watch_hits}


def audit_phrases(records: list[dict], report: dict) -> None:
    n = len(records)
    lowered = [(r.get("content") or "").lower() for r in records]
    hits = {p: sum(1 for t in lowered if p in t) for p in STOCK_PHRASES}
    hits = {p: c for p, c in hits.items() if c}
    # discovery: word 5-grams appearing in 3+ documents (beyond the fixed list)
    gram_df = collections.Counter()
    for t in lowered:
        words = re.findall(r"[a-z']+", t)
        grams = {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}
        gram_df.update(grams)
    common = [(g, c) for g, c in gram_df.most_common(200) if c >= max(3, 0.05 * n)][:8]
    worst = max(hits.values(), default=0)
    v = "GOOD" if worst == 0 else ("OK" if worst <= max(1, 0.05 * n) else "BAD")
    print("STOCK PHRASES")
    print(_fmt("banned-phrase hits", ", ".join(f"{p!r} ({c})" for p, c in sorted(hits.items(), key=lambda kv: -kv[1])) or "none", v))
    if common:
        print(_fmt("recurring 5-grams (3+ docs)", "; ".join(f"{g!r} ({c})" for g, c in common),
                   None, "(discovery list — judge by eye, shared topic makes some overlap normal)"))
    report["phrases"] = {"banned_hits": hits, "recurring_5grams": common}


def audit_openings(records: list[dict], report: dict) -> None:
    n = len(records)
    firsts = [first_sentence(r.get("content") or "") for r in records]
    pattern_counts = collections.Counter()
    for f in firsts:
        for rx, label in _OPENER_PATTERNS:
            if rx.search(f.strip()):
                pattern_counts[label] += 1
                break
    stem_counts = collections.Counter(" ".join(f.lower().split()[:5]) for f in firsts if f)
    dup_stems = [(s, c) for s, c in stem_counts.most_common(5) if c >= 2]
    formulaic = sum(pattern_counts.values()) / max(n, 1)
    v = _verdict(formulaic, 0.15, 0.35)
    print("OPENINGS")
    print(_fmt("formulaic opener share", f"{formulaic:.0%}", v,
               "(abstract-nominalization / 'In recent years' style openings)"))
    for label, c in pattern_counts.most_common():
        print(f"      - {label}: {c}")
    if dup_stems:
        print(_fmt("duplicate first-5-word stems", "; ".join(f"{s!r} x{c}" for s, c in dup_stems)))
    report["openings"] = {"formulaic_frac": round(formulaic, 3), "patterns": dict(pattern_counts),
                         "dup_stems": dup_stems}


def audit_register(records: list[dict], report: dict) -> None:
    rows = []
    for r in records:
        # The proxy is English-only. Final corpora label language with the full
        # name ("English", via derive_language); accept the "en" code too so
        # legacy/test records still count. (Matching only "en" silently skipped
        # every real doc, disabling this check on production runs.)
        if str(r.get("language", "English")).lower() not in ("en", "english"):
            continue
        text = r.get("content") or ""
        words = max(len(re.findall(r"\w+", text)), 1)
        fp = len(_FIRST_PERSON_RE.findall(text))
        contractions = len(_CONTRACTION_RE.findall(text))
        reads_personal = (fp * 1000 / words) >= 5 and (contractions * 1000 / words) >= 2
        rows.append(reads_personal)
    if not rows:
        print("REGISTER: no English docs to check (proxy is English-only)")
        return
    n = len(rows)
    reads = sum(rows)
    # Corpus-level signal only: what fraction of English docs read first-person.
    # The matrix has no per-doc first-person/expository register label, so this
    # reports the corpus fraction without a verdict; stiffly-written casual
    # genres are caught per document by layer 3's house-style check instead.
    print("REGISTER (heuristic: first-person pronouns + contractions, English docs)")
    print(_fmt("reads first-person", f"{reads} of {n} ({reads / n:.0%})", None,
               "(uniform-draw corpora collapse to ~10% — a real mix has far more)"))
    report["register"] = {"reads_personal_frac": round(reads / n, 3)}


# ---------------------------------------------------------------- LLM pattern detection

_CONSOLIDATE_PROMPT = (
    "Below is a raw list of recurring patterns reported across batches of documents from one "
    "synthetic corpus. All documents share one broad topic BY DESIGN, so patterns that merely "
    "restate the shared subject matter (e.g. 'welfare focus', 'mentions AI') must be dropped. "
    "Merge duplicates and near-duplicates into one canonical pattern each, keep only patterns "
    "about FORM, STRUCTURE, STYLE, PHRASING, or recurring BEHAVIORAL arcs, and cap the result "
    "at 15 patterns.\n\n"
    "For each kept pattern, decide whether it is a genuine DEFECT — templating or sameness that "
    "would damage the corpus as training data (near-identical openings, one argument skeleton "
    "repeated everywhere, a stock phrase, reused invented names, every skeptic softening). "
    "Broad, natural properties of a mixed-genre corpus are NOT defects: a general "
    "problem-then-response shape, first-person voice in personal genres, openings that reflect "
    "the shared subject.\n\n"
    "Return ONLY a JSON array of objects with keys: pattern (short name), kind (structural | "
    "rhetorical | behavioral), description (one sentence), strict_check (one sentence a grader "
    "can apply to one document to decide if the pattern is unambiguously present), is_defect "
    "(true | false).\n\nRAW PATTERNS:\n"
)

_PREVALENCE_PROMPT_HEAD = (
    "Here is a list of candidate patterns, then one document. Return ONLY a JSON array of the "
    "integer ids of the patterns this document UNAMBIGUOUSLY exhibits (apply each pattern's "
    "check strictly; return [] if none apply).\n\nPATTERNS:\n"
)


def _parse_json_block(raw: str):
    """The scan reply's JSON via the shared hardened parser — fences, prose
    around the payload, and control characters inside strings all tolerated;
    raises json.JSONDecodeError when nothing usable is present (callers keep
    their existing failure handling)."""
    return utils.extract_json(raw or "")


def llm_pattern_scan(records: list[dict], config: dict, report: dict,
                     scan_sample: int, batch_size: int, prevalence_sample: int) -> None:
    prompts_dir = Path(__file__).parent.parent / "prompts" / "tools"
    texts = [r.get("content") or "" for r in records]
    stride = max(len(texts) / max(scan_sample, 1), 1.0)
    scan = [texts[int(i * stride)] for i in range(min(scan_sample, len(texts)))]

    print(f"\nLLM PATTERN SCAN ({len(scan)} docs in batches of {batch_size})")
    raw_patterns = []
    batches = [scan[i:i + batch_size] for i in range(0, len(scan), batch_size)]

    def scan_batch(batch: list[str]) -> list[dict]:
        blob = "\n\n---\n\n".join(t[:1200] for t in batch)
        prompt = utils.load_prompt(prompts_dir / "pattern_scan.txt", documents=blob)
        try:
            found = _parse_json_block(api.call_claude(user_message=prompt, stage="eval_audit_sdf"))
            return [p for p in found if isinstance(p, dict) and p.get("pattern")]
        except Exception as e:
            print(f"    batch scan parse failure ({e}); skipping batch")
            return []

    workers = config.get("workers", 1)
    for found in utils.parallel_map(scan_batch, batches, workers):
        raw_patterns.extend(found)
    print(f"   {len(raw_patterns)} raw pattern reports")
    if not raw_patterns:
        report["patterns"] = []
        return

    listing = "\n".join(
        f"- {p.get('pattern')}: {p.get('description', '')} (kind: {p.get('kind', '?')}, "
        f"prevalence in its batch: {p.get('prevalence', '?')})"
        for p in raw_patterns
    )
    try:
        canonical = _parse_json_block(api.call_claude(user_message=_CONSOLIDATE_PROMPT + listing,
                                                      stage="eval_audit_sdf"))
        canonical = [p for p in canonical if isinstance(p, dict) and p.get("pattern")][:15]
    except Exception as e:
        print(f"   consolidation failed ({e}); reporting raw patterns without prevalence")
        report["patterns"] = raw_patterns
        return
    print(f"   {len(canonical)} canonical form patterns after consolidation")

    plist = "\n".join(
        f"{i}: {p['pattern']} — {p.get('strict_check') or p.get('description', '')}"
        for i, p in enumerate(canonical)
    )
    stride = max(len(texts) / max(prevalence_sample, 1), 1.0)
    rate_docs = [texts[int(i * stride)] for i in range(min(prevalence_sample, len(texts)))]

    def rate_one(doc: str) -> set[int]:
        prompt = _PREVALENCE_PROMPT_HEAD + plist + "\n\nDOCUMENT:\n" + doc[:1200]
        try:
            ids = _parse_json_block(api.call_claude(user_message=prompt, stage="eval_audit_sdf"))
            return {i for i in ids if isinstance(i, int) and 0 <= i < len(canonical)}
        except Exception:
            return set()

    counts = collections.Counter()
    total = 0
    for ids in utils.parallel_map(rate_one, rate_docs, workers):
        total += 1
        counts.update(ids)

    print(f"   prevalence across {total} docs (RED = defect AND >30%):")
    rows = []
    for i, p in enumerate(canonical):
        prev = counts[i] / max(total, 1)
        red = bool(p.get("is_defect")) and prev > 0.30
        flag = "  <-- DEFECT, WIDESPREAD: fix the generator" if red else (
            "  (defect, but rare)" if p.get("is_defect") else "")
        print(f"    {prev:5.0%}  {p['pattern']}{flag}")
        rows.append({**p, "prevalence": round(prev, 3), "flagged": red})
    if not any(r["flagged"] for r in rows):
        print("    -> no widespread defects: common patterns are acceptable genre structure.")
    report["patterns"] = rows


# ---------------------------------------------------------------- principle coverage

# A principle exercised by fewer than this fraction of sampled docs is flagged
# as starved. The floor is deliberately low: with ~14 principles and mixed
# genres, even coverage would put each principle well above 5%.
PRINCIPLE_COVERAGE_FLOOR = 0.05


def audit_principle_coverage(records: list[dict], config: dict, report: dict,
                             sample: int) -> None:
    """Judge which distilled constitution principles each sampled document
    substantively exercises, and flag principles the corpus is starving
    (composition guidelines require every principle to be exercised by many
    documents, not just the headline weigh-welfare one)."""
    from shared import constitution_loader

    principles = constitution_loader.load_principles()
    numbers = [int(p["number"]) for p in principles]  # CSV fields parse as str
    block = constitution_loader.format_principles(principles)
    prompts_dir = Path(__file__).parent.parent / "prompts" / "tools"
    template_path = prompts_dir / "principle_coverage.txt"

    texts = [r.get("content") or "" for r in records]
    stride = max(len(texts) / max(sample, 1), 1.0)
    docs = [texts[int(i * stride)] for i in range(min(sample, len(texts)))]

    print(f"\nPRINCIPLE COVERAGE ({len(docs)} docs against {len(numbers)} principles)")

    def rate_one(doc: str) -> set[int] | None:
        prompt = utils.load_prompt(template_path, principles=block, document=doc[:6000])
        try:
            ids = _parse_json_block(api.call_claude(user_message=prompt, stage="eval_audit_sdf"))
            return {i for i in ids if isinstance(i, int) and i in numbers}
        except Exception:
            return None  # malformed judge output: unrated, not zero-principle

    counts = collections.Counter()
    rated = 0
    workers = config.get("workers", 1)
    for ids in utils.parallel_map(rate_one, docs, workers):
        if ids is None:
            continue
        rated += 1
        counts.update(ids)

    if not rated:
        print("   no documents rated (all judge calls failed); skipping")
        report["principle_coverage"] = {"rated": 0}
        return

    by_principle = {}
    starved = []
    label = {int(p["number"]): p.get("principle", "").strip() for p in principles}
    for n in numbers:
        frac = counts[n] / rated
        by_principle[n] = round(frac, 3)
        flag = ""
        if frac < PRINCIPLE_COVERAGE_FLOOR:
            starved.append(n)
            flag = f"  <-- STARVED (<{PRINCIPLE_COVERAGE_FLOOR:.0%})"
        print(_fmt(f"{n}. {label[n][:60]}", f"{frac:5.0%}") + flag)
    if starved:
        print(f"   -> {len(starved)} principle(s) under the {PRINCIPLE_COVERAGE_FLOOR:.0%} floor: "
              "consider reweighting resolution arcs or plan guidance")
    else:
        print("   -> every principle clears the floor")
    report["principle_coverage"] = {
        "rated": rated,
        "floor": PRINCIPLE_COVERAGE_FLOOR,
        "by_principle": by_principle,
        "starved": starved,
    }


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus-level audit of SDF output.")
    parser.add_argument("--input", default="outputs/sdf/latest",
                        help="Run directory or sdf_corpus.jsonl path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dup-sample", type=int, default=4000,
                        help="Max docs for the pairwise near-dup scan")
    parser.add_argument("--patterns", action="store_true",
                        help="Run LLM pattern detection (costs API calls)")
    parser.add_argument("--pattern-sample", type=int, default=48,
                        help="Docs fed to the LLM pattern scan")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--prevalence-sample", type=int, default=80,
                        help="Docs rated for per-pattern prevalence")
    parser.add_argument("--principles", action="store_true",
                        help="Judge per-principle coverage against the distilled "
                             "constitution principles (costs API calls)")
    parser.add_argument("--principle-sample", type=int, default=80,
                        help="Docs judged for principle coverage")
    args = parser.parse_args()

    records, report_dir = resolve_input(args.input)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Corpus is empty — nothing to audit.")

    config = utils.load_config(args.config)

    print(f"=== SDF corpus audit: {args.input} ({len(records)} documents) ===\n")
    report: dict = {"input": utils.repo_relative(args.input), "n_docs": len(records)}
    audit_composition(records, report)
    print()
    audit_length_truncation(records, report)
    print()
    audit_markdown(records, report)
    print()
    audit_near_dups(records, report, args.dup_sample)
    print()
    audit_names(records, report)
    print()
    audit_phrases(records, report)
    print()
    audit_openings(records, report)
    print()
    audit_register(records, report)

    if args.patterns or args.principles:
        api.init(args.config)  # evals log to the global cost log
    if args.patterns:
        llm_pattern_scan(records, config, report,
                         args.pattern_sample, args.batch_size, args.prevalence_sample)
    if args.principles:
        audit_principle_coverage(records, config, report, args.principle_sample)

    utils.ensure_dir(report_dir)
    out = report_dir / "audit_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to {out}")
    if not args.patterns:
        print("Tip: rerun with --patterns for the LLM templating scan (small API cost).")


if __name__ == "__main__":
    main()
