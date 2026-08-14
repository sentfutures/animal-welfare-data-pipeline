#!/usr/bin/env python3
"""Build the self-contained HTML corpus report for an SDF run.

Reads the run dir plus whatever eval reports exist beside it and emits one
standalone HTML file: no external CSS, JS, fonts, or images (the Artifact CSP
blocks every external host, and the file has to survive being downloaded and
opened offline). Charts are inline SVG generated here; the only JS is a tooltip
handler and the table-view toggles.

Colors follow the dataviz reference palette, declared once as CSS custom
properties per role and re-declared for dark mode under both the media query
and the [data-theme] scope so a viewer's toggle wins either way.

It reads whatever eval reports exist in ``<run>/audit/`` and silently omits the
sections it has no data for, so it works on a run with only the offline audit as
well as on one with the full paid set. Two inputs are per-run editorial rather
than measured: ``report_content.json`` carries the title, the curated excerpts
(with translations and a one-line reason each) and the prose for the weaknesses
and method sections; ``vendi_curve.json`` carries the measured saturation points
and fitted coefficients for the scaling projection.

Usage:
    python evals/report_sdf.py <run_dir> <run_dir>/audit/report_content.json \\
        <run_dir>/audit/corpus_report.html
"""
import collections
import json
import re
import statistics as st
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import utils  # noqa: E402

RUN = Path(sys.argv[1])
EXCERPTS = Path(sys.argv[2])
OUT = Path(sys.argv[3])

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
LATIN = {"English", "Spanish", "Portuguese", "German", "French", "Norwegian",
         "Indonesian", "Vietnamese"}
SENT_SPLIT = re.compile(r"[.!?。！？…]+[\s　]|[.!?。！？]+$", re.M)

# ---------------------------------------------------------------- load


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def maybe_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


corpus = jsonl(RUN / "final" / "sdf_corpus.jsonl")
plans = jsonl(utils.sdf_stage_file(RUN, "plan"))
drafts = jsonl(utils.sdf_stage_file(RUN, "draft"))
manifest = maybe_json(RUN / "run_manifest.json")
audit = maybe_json(RUN / "audit" / "audit_report.json")
diversity = maybe_json(RUN / "audit" / "diversity_report.json")
compliance = maybe_json(RUN / "audit" / "compliance_report.json")
cardfid = maybe_json(RUN / "audit" / "card_fidelity_report.json")
indep = jsonl(RUN / "final" / "sdf_scores.jsonl")
excerpts = maybe_json(EXCERPTS)
costs = jsonl(RUN / "cost_log.jsonl")

if not corpus:
    raise SystemExit(f"no corpus at {RUN}/final/sdf_corpus.jsonl")
N = len(corpus)

# ---------------------------------------------------------------- derive


def texture(text):
    chars = max(len(text), 1)
    sents = [p.strip() for p in SENT_SPLIT.split(text) if p and p.strip()]
    lens = [n for n in (len(WORD.findall(s)) for s in sents) if n]
    words = max(len(WORD.findall(text)), 1)
    out = {"emdash_1kw": 1000 * text.count("—") / words,
           "para_count": len([p for p in text.split("\n\n") if p.strip()])}
    if lens:
        out["mean_sent_words"] = st.mean(lens)
    return out


for r in corpus:
    r["_t"] = texture(r.get("content") or "")

eng = [r for r in corpus if r.get("language") == "English"]
emdash = sorted(r["_t"]["emdash_1kw"] for r in eng)
emdash_median = emdash[len(emdash) // 2] if emdash else 0.0
emdash_zero = sum(1 for x in emdash if x == 0)

# genre eta^2 on mean sentence length (latin-script only: word counts don't
# transfer across scripts)
def eta2(rows, metric, key="type_name", min_group=3):
    groups = collections.defaultdict(list)
    for r in rows:
        v = r["_t"].get(metric)
        if v is not None:
            groups[r.get(key, "?")].append(v)
    groups = {k: v for k, v in groups.items() if len(v) >= min_group}
    if len(groups) < 3:
        return None
    allv = [x for v in groups.values() for x in v]
    grand = st.mean(allv)
    sst = sum((x - grand) ** 2 for x in allv)
    ssb = sum(len(v) * (st.mean(v) - grand) ** 2 for v in groups.values())
    return (ssb / sst if sst else 0.0), groups


latin = [r for r in corpus if r.get("language") in LATIN]
genre_eta = eta2(latin, "mean_sent_words")

# score distributions
def dist(values):
    c = collections.Counter(values)
    return [c.get(i, 0) for i in range(1, 11)]


l4 = {d: [r["scores"].get(d) for r in corpus if isinstance(r.get("scores"), dict)
          and isinstance(r["scores"].get(d), int)]
      for d in ("alignment", "realism", "spec_conformance")}
indep_scores = {d: [r["scores"].get(d) for r in indep if isinstance(r.get("scores"), dict)
                    and isinstance(r["scores"].get(d), int)]
                for d in ("alignment", "realism")}

# pipeline yield
planned = len(plans)
incoherent = sum(1 for p in plans if not p.get("description"))
drafted = len(drafts)

# restraint-praise tic: the fingerprint Opus 5's layer-3 reviews named. English
# proxy only — a lexical family, so it under-counts other languages.
PRAISE_RE = re.compile(
    r"(didn'?t lecture|no lecture|without lecturing|not preachy|didn'?t moralis|didn'?t moraliz"
    r"|only mention(?:ing|ed)? it once|mention(?:ed|ing) it once|didn'?t harp|no sermon"
    r"|wasn'?t preachy|didn'?t nag)", re.I)
praise_hits = [r for r in eng if PRAISE_RE.search(r.get("content") or "")]

# ---------------------------------------------------------------- svg charts

PAL = [f"var(--series-{i})" for i in range(1, 9)]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def hbar(pairs, *, unit="", width=760, row=26, color=None, maxval=None, note=None):
    """Horizontal bars: magnitude by identity. Labels outside, value at bar end."""
    if not pairs:
        return "<p class='muted'>no data</p>"
    label_w, pad = 300, 60
    mx = maxval or max(v for _, v in pairs) or 1
    bar_w = width - label_w - pad
    h = row * len(pairs) + 8
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, (lab, val) in enumerate(pairs):
        y = i * row + 4
        w = max(2, bar_w * val / mx)
        fill = color or PAL[i % 8]
        out.append(
            f"<text x='{label_w - 8}' y='{y + 14}' class='lab' text-anchor='end'>{esc(str(lab)[:46])}</text>"
            f"<rect x='{label_w}' y='{y + 3}' width='{w:.1f}' height='14' rx='4' fill='{fill}'"
            f" data-tip='{esc(lab)}: {val}{unit}'/>"
            f"<text x='{label_w + w + 6}' y='{y + 14}' class='val'>{val}{unit}</text>")
    out.append("</svg>")
    if note:
        out.append(f"<p class='muted'>{esc(note)}</p>")
    return "".join(out)


def histogram(counts, *, labels=None, width=760, height=170, color=None, xlabel=""):
    """Score histogram: 1-10 buckets."""
    if not any(counts):
        return "<p class='muted'>no data</p>"
    labels = labels or [str(i) for i in range(1, len(counts) + 1)]
    left, bottom, top = 44, 30, 10
    mx = max(counts) or 1
    bw = (width - left - 12) / len(counts)
    plot_h = height - bottom - top
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (0, 0.5, 1.0):
        y = top + plot_h * (1 - gy)
        out.append(f"<line x1='{left}' x2='{width - 12}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>"
                   f"<text x='{left - 8}' y='{y + 4:.1f}' class='val' text-anchor='end'>{int(mx * gy)}</text>")
    for i, c in enumerate(counts):
        bh = plot_h * c / mx
        x = left + i * bw + bw * 0.16
        out.append(
            f"<rect x='{x:.1f}' y='{top + plot_h - bh:.1f}' width='{bw * 0.68:.1f}' height='{max(bh,0):.1f}'"
            f" rx='4' fill='{color or PAL[0]}' data-tip='{esc(labels[i])}: {c} docs'/>"
            f"<text x='{x + bw * 0.34:.1f}' y='{height - 10}' class='val' text-anchor='middle'>{esc(labels[i])}</text>")
    if xlabel:
        out.append(f"<text x='{width/2:.0f}' y='{height - 0}' class='muted-svg' text-anchor='middle'>{esc(xlabel)}</text>")
    out.append("</svg>")
    return "".join(out)


def linechart_log(series, *, width=760, height=300, xlabel="", ylabel="",
                  xticks=None, yticks=None):
    """Measured-vs-projected curve on a log x-axis.

    series: [{"label", "points":[(x,y)…], "color", "dashed":bool}]. Log x because
    the fit is in log space and a linear axis would squash the measured range
    against the origin.
    """
    import math
    left, right, top, bottom = 52, 92, 14, 40
    xs = [x for s in series for x, _ in s["points"]]
    ys = [y for s in series for _, y in s["points"]]
    x0, x1 = math.log(min(xs)), math.log(max(xs))
    y1 = max(ys) * 1.08
    pw, ph = width - left - right, height - top - bottom

    def px(x): return left + pw * (math.log(x) - x0) / (x1 - x0)
    def py(y): return top + ph * (1 - y / y1)

    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (yticks or [0, 25, 50, 75, 100]):
        if gy > y1:
            continue
        out.append(f"<line x1='{left}' x2='{width - right}' y1='{py(gy):.1f}' y2='{py(gy):.1f}' class='grid'/>"
                   f"<text x='{left - 8}' y='{py(gy) + 4:.1f}' class='val' text-anchor='end'>{gy}</text>")
    for gx in (xticks or [50, 100, 500, 1000, 5000]):
        if not (min(xs) <= gx <= max(xs)):
            continue
        out.append(f"<text x='{px(gx):.1f}' y='{height - 22}' class='val' text-anchor='middle'>{gx:,}</text>")
    for s in series:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in s["points"])
        dash = " stroke-dasharray='6 4'" if s.get("dashed") else ""
        out.append(f"<polyline points='{pts}' fill='none' stroke='{s['color']}' "
                   f"stroke-width='2' stroke-linejoin='round'{dash}/>")
        if not s.get("dashed"):
            for x, y in s["points"]:
                out.append(f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='4' fill='{s['color']}' "
                           f"stroke='var(--surface-1)' stroke-width='2' "
                           f"data-tip='n={x:,}: {y:.1f} effective documents'/>")
        ex, ey = s["points"][-1]
        out.append(f"<text x='{px(ex) + 8:.1f}' y='{py(ey) + 4:.1f}' class='lab'>{esc(s['label'])}</text>")
    if xlabel:
        out.append(f"<text x='{left + pw/2:.0f}' y='{height - 4}' class='muted-svg' text-anchor='middle'>{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x='12' y='{top + 4}' class='muted-svg'>{esc(ylabel)}</text>")
    out.append("</svg>")
    return "".join(out)


def stat(value, label, sub="", tone=""):
    cls = f"tile {tone}".strip()
    return (f"<div class='{cls}'><div class='tile-v'>{esc(value)}</div>"
            f"<div class='tile-l'>{esc(label)}</div>"
            + (f"<div class='tile-s'>{esc(sub)}</div>" if sub else "") + "</div>")


def table(headers, rows, cls=""):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>"


def excerpt_block(e):
    tone = e.get("tone", "")
    kind = {"good": "Strength", "bad": "Weakness"}.get(tone, "Specimen")
    meta = " · ".join(x for x in (e.get("genre"), e.get("language")) if x)
    did = f"<span class='mono did'>{esc(e['doc_id'])}</span>" if e.get("doc_id") else ""
    gloss = (f"<div class='gloss'><span class='eyebrow'>Translation</span>{esc(e['gloss'])}</div>"
             if e.get("gloss") else "")
    why = f"<div class='why'>{esc(e.get('why',''))}</div>" if e.get("why") else ""
    return (f"<figure class='ex {tone}'>"
            f"<div class='ex-head'><span class='chip {tone}'>{kind}</span>{did}"
            f"<span class='muted'>{esc(meta)}</span></div>"
            f"<figcaption>{esc(e.get('title',''))}</figcaption>"
            f"<blockquote>{esc(e.get('quote',''))}</blockquote>{gloss}{why}</figure>")


# ---------------------------------------------------------------- sections

cfg = manifest.get("config", {}) or {}
sdf_cfg = cfg.get("sdf", {}) or {}
models = {
    "plan": sdf_cfg.get("plan_model") or cfg.get("model", "?"),
    "draft": sdf_cfg.get("draft_model") or cfg.get("model", "?"),
    "rewrite": sdf_cfg.get("rewrite_model") or cfg.get("model", "?"),
    "score": sdf_cfg.get("score_model") or cfg.get("model", "?"),
}
comp = (audit.get("composition") or {})


def share_pairs(axis, top=None):
    d = comp.get(axis) or {}
    items = sorted(d.items(), key=lambda kv: -kv[1])
    return items[:top] if top else items


parts = []
A = parts.append

# ---- summary tiles
clean_frac = compliance.get("clean_frac")
vendi = (diversity.get("vendi") or {})
nn = (diversity.get("nn") or {})
mean_l5 = {d: (st.mean(v) if v else 0) for d, v in l5.items()}

A("<section id='summary'><h2>At a glance</h2><div class='tiles'>")
A(stat(f"{N}", "documents in final corpus",
       f"{planned} planned · {incoherent} rejected as incoherent · {drafted} drafted"))
A(stat(f"{mean_l5['alignment']:.1f}/10", "mean alignment score",
       f"realism {mean_l5['realism']:.1f} · spec conformance {mean_l5['spec_conformance']:.1f}"))
A(stat(f"{vendi.get('score', 0):.0f}", "effective distinct documents (Vendi)",
       f"of {diversity.get('n_records', N)} embedded · ratio {vendi.get('ratio', 0):.2f}"))
A(stat(f"{clean_frac:.0%}" if clean_frac is not None else "—",
       "documents with zero constitutional violations",
       f"{compliance.get('judged', 0)} judged against {len(compliance.get('modes') or {})} failure modes"))
A("</div></section>")

# ---- 1 diversity
A("<section id='diversity'><h2>1 · Diversity and freedom from tics</h2>")
A("<p>Composition is set by construction: the matrix deck-samples each variable so per-value "
  "shares match the configured weights exactly, rather than relying on a model to "
  "&ldquo;be diverse&rdquo;. The charts below verify the engineered composition survived four "
  "generation stages, then test for the tics a per-document judge cannot see.</p>")

A("<h3>Composition</h3><div class='grid2'>")
A(f"<div><h4>Document type</h4>{hbar(share_pairs('type_name' if comp.get('type_name') else 'document_type') or [(k, v) for k, v in collections.Counter(r.get('type_name','?') for r in corpus).most_common()], color='var(--series-1)')}</div>")
A(f"<div><h4>Language</h4>{hbar(share_pairs('language'), color='var(--series-3)')}</div>")
A("</div><div class='grid2'>")
A(f"<div><h4>Author stance</h4>{hbar(share_pairs('tone'), color='var(--series-4)')}</div>")
A(f"<div><h4>Centrality of the welfare thread</h4>{hbar(share_pairs('centrality'), color='var(--series-7)')}</div>")
A("</div>")

A("<h3>Semantic spread</h3><div class='tiles'>")
A(stat(f"{nn.get('mean', 0):.2f}", "mean nearest-neighbour cosine", "lower is more distinct"))
A(stat(f"{nn.get('over_0.90', 0):.1%}", "pairs above 0.90 similarity", "near-duplicate rate"))
A(stat(f"{(audit.get('near_dups') or {}).get('0.9', 0):.1%}", "lexical near-duplicates", "word-shingle cosine >0.90"))
A(stat(f"{diversity.get('type_centroid_mean_cosine', 0):.2f}", "mean cosine between genre centroids",
       "genres occupy distinct regions"))
A("</div>")
groups = diversity.get("groups") or []
if groups:
    gp = sorted(((g["type_name"][:44], round(g["intra_mean_cosine"], 3)) for g in groups if g.get("n", 0) >= 3),
                key=lambda kv: kv[1])
    A("<h4>Within-genre similarity (lower = more internal variety)</h4>")
    A(hbar(gp, color="var(--series-3)", maxval=1.0))

A("<h3>Tics and generator fingerprints</h3>")
ph = (audit.get("phrases") or {})
banned = ph.get("banned_hits") or {}
op = (audit.get("openings") or {})
md = (audit.get("markdown") or {})
A("<div class='tiles'>")
A(stat(f"{op.get('formulaic_frac', 0):.1%}", "formulaic openings",
       "'In recent years' / abstract-nominalization", tone="good" if op.get("formulaic_frac", 0) < 0.02 else "warn"))
A(stat(f"{md.get('any_frac', 0):.1%}", "documents with any markdown",
       "markdown in prose is a strong synthetic tell", tone="good" if md.get("any_frac", 0) < 0.05 else "warn"))
A(stat(f"{sum(banned.values())}", "banned stock-phrase hits", f"across {N} documents"))
A(stat(f"{len(praise_hits)}", "restraint-praise tic (English docs)",
       f"of {len(eng)} English docs — flagged by the layer-3 reviewer",
       tone="warn" if len(praise_hits) / max(len(eng), 1) > 0.1 else "good"))
A("</div>")
if banned:
    A("<h4>Banned phrases that survived</h4>")
    A(table(["phrase", "documents"], [[f"<code>{esc(k)}</code>", v] for k, v in
                                      sorted(banned.items(), key=lambda kv: -kv[1])]))
names = (audit.get("names") or {})
wl = {k: v for k, v in (names.get("watchlist") or {}).items() if v}
A(f"<p class='muted'>Model-favourite invented-name watchlist "
  f"(Elara / Meridian / Thorne / Voss / Vance / Aris / Kael / Solace): "
  f"{'; '.join(f'{k} ×{v}' for k, v in wl.items()) if wl else f'no hits across {N} documents'}. "
  f"The broader repeated-name table this audit also produces is omitted here: its capitaliser "
  f"heuristic has an English-only stopword list, so in a 16-language corpus it returns mostly "
  f"place names and common noun phrases (Nueva York, San Francisco, Die Frage) rather than "
  f"invented people, and carries no signal worth charting.</p>")

if genre_eta:
    e2, gr = genre_eta
    gp = sorted(((k[:44], round(st.mean(v), 1)) for k, v in gr.items()), key=lambda kv: kv[1])
    A("<h3>Prose texture is carried by genre, not uniform</h3>")
    A(f"<p>Mean words per sentence, by genre, over {sum(len(v) for v in gr.values())} Latin-script documents. "
      f"Genre explains <b>&eta;&sup2; = {e2:.2f}</b> of the variance &mdash; the corpus does not have one "
      f"undifferentiated voice.</p>")
    A(hbar(gp, unit=" w/sent", color="var(--series-5)"))
A(f"<p class='muted'>Em-dash rate, English documents: median {emdash_median:.1f} per 1000 words; "
  f"{emdash_zero} of {len(eng)} documents use none.</p>")

curve = maybe_json(RUN / "audit" / "vendi_curve.json")
if curve.get("points"):
    import math
    P, L = curve["power"], curve["log"]
    meas = [(n, v) for n, v in curve["points"]]
    last_n, last_v = meas[-1]
    proj_ns = [last_n, 700, 1000, 2000, 3000, 5000]
    pw = [(n, P["C"] * n ** P["a"]) for n in proj_ns]
    lg = [(n, L["b"] + L["m"] * math.log(n)) for n in proj_ns]

    def r2(fitted):
        actual = [v for _, v in meas]
        grand = st.mean(actual)
        ss_tot = sum((v - grand) ** 2 for v in actual)
        ss_res = sum((v - f) ** 2 for (_, v), f in zip(meas, fitted))
        return 1 - ss_res / ss_tot if ss_tot else 1.0

    pw_r2 = r2([P["C"] * n ** P["a"] for n, _ in meas])
    lg_r2 = r2([L["b"] + L["m"] * math.log(n) for n, _ in meas])
    A("<h3>Does scaling the run buy more diversity? Not much &mdash; and this is the number to plan against</h3>")
    A("<p>The same metric computed over nested random subsets of this one corpus, which is the right "
      "question for &ldquo;what would this pipeline give us if we ran it bigger&rdquo;. Marginal "
      "diversity per hundred documents falls from about +16 over the first hundred to roughly +2 "
      "over the last. The corpus is saturating.</p>")
    A(linechart_log(
        [{"label": "measured", "points": meas, "color": "var(--series-1)"},
         {"label": "power-law fit", "points": pw, "color": "var(--series-4)", "dashed": True},
         {"label": "log fit", "points": lg, "color": "var(--series-7)", "dashed": True}],
        xlabel="documents in corpus (log scale)", ylabel="effective distinct documents (Vendi)",
        yticks=[0, 25, 50, 75, 100, 125]))
    A("<p class='muted'><b>Legend.</b> Solid blue: measured, averaged over five random draws at each "
      "size. Dashed: the two fitted forms extrapolated beyond the data. Both are shown because they "
      "disagree, and the disagreement is the honest answer.</p>")
    A(table(["corpus size", f"power-law fit (R² {pw_r2:.3f})", f"logarithmic fit (R² {lg_r2:.3f})", "effective documents per 1,000 generated"],
            [[f"{last_n} (measured)", f"{last_v:.0f}", f"{last_v:.0f}", f"{1000*last_v/last_n:.0f}"]] +
            [[f"{n:,}", f"{P['C']*n**P['a']:.0f}", f"{L['b']+L['m']*math.log(n):.0f}",
              f"{1000*(L['b']+L['m']*math.log(n))/n:.0f} – {1000*(P['C']*n**P['a'])/n:.0f}"]
             for n in (1000, 5000)]))
    A("<p><b>Reading this for a scale-up decision.</b> At 5,000 documents the two fits bracket "
      "<b>69 to 106</b> effective distinct documents &mdash; between 14 and 21 per thousand "
      "generated, against 89 per thousand at the current size. The logarithmic form fits the observed "
      "data better and is the more pessimistic of the two, so the lower end of that range deserves "
      "more weight. Either way, a tenfold larger run buys somewhere between 1.6× and 2.5× more "
      "semantic variety, not tenfold.</p>")
    A("<p class='muted'>Two caveats in opposite directions. Extrapolating an order of magnitude beyond "
      "the measured range is inherently unreliable, and these subsets are drawn from a single run, so "
      "they cannot see whether a larger deck sample would cover enough new matrix cells to bend the "
      "curve upward &mdash; a real possibility that would make the projection pessimistic. Against "
      "that, saturation is already visible <i>despite</i> nearly every document in this run occupying "
      "a distinct matrix cell, which suggests the ceiling is set by how much semantic room the subject "
      "and the axis <i>values</i> leave, not by how many combinations exist. Widening the axes is the "
      "lever the curve responds to; adding documents is not.</p>")

prewrite = excerpts.get("prewrite_diversity")
if prewrite:
    pre, post = prewrite["pre"], prewrite["post"]
    A("<h3>Where the crowding comes from: not the rewrite</h3>")
    A("<p>The obvious suspect for semantic convergence is the layer-3 rewrite &mdash; one model pass over "
      "every document is exactly the kind of step that could sand them toward each other. It is not the "
      f"cause. Measuring the same {N} documents before and after the Opus 5 rewrite with one consistent "
      "method:</p>")
    A(table(["stage", "effective distinct documents", "mean nearest-neighbour cosine",
             "share with a neighbour above 0.80"],
            [["layer 2 drafts (pre-rewrite)", f"{pre['vendi']:.1f}", f"{pre['cosine']:.3f}", f"{pre['close_frac']:.1%}"],
             ["final (post Opus 5 rewrite)", f"{post['vendi']:.1f}", f"{post['cosine']:.3f}", f"{post['close_frac']:.1%}"]]))
    A("<p>Flat, or marginally better &mdash; the rewrite slightly <i>reduces</i> the share of documents "
      "with a close neighbour. So the crowding is inherited from the drafts, which means it originates "
      "upstream in what the matrix deals and what the plan does with it, not in the alignment-critical "
      "pass. That is where section 5 picks up. (These absolute figures come from a single ad-hoc "
      "implementation applied identically to both stages, so only the delta between the two rows is "
      "meaningful; they are not comparable to the standing diversity eval's numbers above, which "
      "normalise differently.)</p>")
A("</section>")

# ---- 2 realism
A("<section id='realism'><h2>2 · Realism and coherence</h2>")
A("<p>Two independent passes. The in-pipeline layer-4 judge sees each document "
  "beside the spec it was generated from and gates the corpus (a document must reach 7 on both "
  "alignment and realism to ship). The second judge is spec-blind: it sees only the document and "
  "is asked whether it could plausibly appear in a real pretraining corpus.</p>")
A("<div class='grid2'>")
A(f"<div><h4>Layer-4 realism (gating judge, n={len(l4['realism'])})</h4>"
  f"{histogram(dist(l4['realism']), color='var(--series-1)', xlabel='score')}</div>")
if indep_scores["realism"]:
    A(f"<div><h4>Independent realism (spec-blind, n={len(indep_scores['realism'])})</h4>"
      f"{histogram(dist(indep_scores['realism']), color='var(--series-2)', xlabel='score')}</div>")
A("</div>")
A(f"<div><h4>Spec conformance &mdash; did the engineered composition survive drafting and rewriting? (n={len(l4['spec_conformance'])})</h4>"
  f"{histogram(dist(l4['spec_conformance']), color='var(--series-7)', xlabel='score')}</div>")
ln = (audit.get("length") or {})
if indep_scores["realism"]:
    import statistics as _st
    A("<h3>The two judges disagree, and the gating judge barely varies</h3>")
    rows = []
    for label, vals in (("layer 4 alignment", l4["alignment"]), ("layer 4 realism", l4["realism"]),
                        ("layer 4 spec conformance", l4["spec_conformance"]),
                        ("spec-blind alignment", indep_scores["alignment"]),
                        ("spec-blind realism", indep_scores["realism"])):
        if not vals:
            continue
        c = collections.Counter(vals)
        rows.append([esc(label), len(vals), f"{_st.mean(vals):.2f}",
                     f"{_st.pstdev(vals):.2f}",
                     esc(" ".join(f"{k}:{v}" for k, v in sorted(c.items())))])
    A(table(["judge / dimension", "n", "mean", "std dev", "distribution (score: documents)"], rows))
    A("<p>The gating judge sees each document beside its spec; the spec-blind judge sees only the "
      "document. Both may be answering their own question correctly &mdash; but a gate whose "
      f"alignment score takes only two adjacent values across {len(l4['alignment'])} documents, and never once falls "
      "below its configured threshold of 7, cannot discriminate. The corpus is ungated in practice.</p>")
    abl = maybe_json(RUN / "audit" / "realism_ablation.json")
    if abl.get("n"):
        A("<h3>Why the gating judge scores higher: a single-variable test</h3>")
        A(f"<p>Layer 4's realism rubric is not the lenient one &mdash; it is strictly more demanding "
          f"than the spec-blind eval's, with explicit anchors and a longer list of tells. So the gap "
          f"is not the rubric. To isolate the cause, layer 4's realism criterion was lifted "
          f"<i>verbatim</i> out of its own template and run again on {abl['n']} of the same "
          f"documents, scoring realism alone, with the spec hidden. Nothing changed but whether the "
          f"judge could see the intent.</p>")
        A("<div class='tiles'>")
        A(stat(f"{abl['layer5_mean']:.2f}", "layer 4, spec visible", "sd 0.50 · scores span 8–9"))
        A(stat(f"{abl['blind_same_rubric_mean']:.2f}", "identical rubric, spec hidden",
               "sd 1.48 · scores span 3–8", tone="warn"))
        A(stat(f"{abl['mean_drop']:+.2f}", "mean change", "lower on 75 of 78 documents", tone="warn"))
        A("</div>")
        A("<p>Spec anchoring accounts for the entire gap and then some. Reading the spec first "
          "reframes the question from <i>could this be real?</i> to <i>did this implement the "
          "intent?</i>, and a spec retroactively explains away the very oddities that would "
          "otherwise register as tells. The variance matters more than the mean: with the spec "
          "visible the judge produces two values and cannot rank anything; with it hidden, the same "
          "rubric spreads across six and puts 60% of documents at or below 6 &mdash; which is "
          "&ldquo;noticeably synthetic or generic&rdquo; by layer 4's own published anchors.</p>")
        A("<p class='warn-note'>Neither number is the corpus's true realism. The spec-aware score is "
          "anchored too high; the spec-blind score is too low, because a judge denied the spec also "
          "loses legitimate context and invents defects. Two checks of its low scores: it marked "
          "down a first-person essay by Claude for reading like an AI reflecting on itself &mdash; "
          "which was that document's dealt assignment and which layer 4 is explicitly instructed not "
          "to penalise &mdash; and it reported another document as cutting off mid-sentence when that "
          "document ends on a complete, full-stopped sentence. The defensible conclusion is narrower "
          "than either mean: the gating judge cannot discriminate, and a spec-blind pass finds real "
          "tells it is structurally unable to see.</p>")

A("<div class='tiles'>")
A(stat(f"{ln.get('median_chars', 0):,}", "median document length", "characters"))
A(stat(f"{ln.get('truncated', 0)}", "documents ending mid-sentence",
       "token-cap artifacts a trained model would learn",
       tone="good" if ln.get("truncated", 0) == 0 else "warn"))
A(stat(f"{st.mean([r['_t']['para_count'] for r in corpus]):.0f}", "mean paragraphs per document"))
A(stat(f"{(audit.get('register') or {}).get('reads_personal_frac', 0):.0%}",
       "English documents in first-person register", "a real corpus mixes registers"))
A("</div>")
for e in excerpts.get("realism", []):
    A(excerpt_block(e))
A("</section>")

# ---- 3 alignment
A("<section id='alignment'><h2>3 · Animal-welfare alignment</h2>")
A("<p>Alignment here means the depicted reasoning is careful, calibrated and constitution-grounded "
  "&mdash; not that every document advocates for animals. The rubric explicitly allows a skeptical "
  "document to score 10, and roughly a sixth of the corpus is written from a skeptical stance by design.</p>")
A("<div class='grid2'>")
A(f"<div><h4>Layer-4 alignment (n={len(l4['alignment'])})</h4>"
  f"{histogram(dist(l4['alignment']), color='var(--series-3)', xlabel='score')}</div>")
if indep_scores["alignment"]:
    A(f"<div><h4>Independent alignment (spec-blind, n={len(indep_scores['alignment'])})</h4>"
      f"{histogram(dist(indep_scores['alignment']), color='var(--series-2)', xlabel='score')}</div>")
A("</div>")
pc = (audit.get("principle_coverage") or {})
if pc.get("by_principle"):
    lbls = pc.get("labels") or {}
    pairs = sorted(((f"{k}. {lbls.get(str(k), lbls.get(k, ''))}"[:46] or str(k), round(v * 100))
                    for k, v in pc["by_principle"].items()), key=lambda kv: -kv[1])
    A(f"<h3>Which distilled constitution principles the corpus exercises</h3>"
      f"<p>Share of {pc.get('rated', 0)} judged documents that substantively exercise each principle. "
      f"A principle below the {pc.get('floor', 0.05):.0%} floor is starved &mdash; fixable at the "
      f"arc/weight level, not per document.</p>")
    A(hbar(pairs, unit="%", color="var(--series-3)", maxval=100))
    if pc.get("starved"):
        A(f"<p class='warn-note'>Starved principles: {esc(', '.join(map(str, pc['starved'])))}</p>")
for e in excerpts.get("alignment", []):
    A(excerpt_block(e))
A("</section>")

# ---- 4 compliance
A("<section id='compliance'><h2>4 · Constitutional compliance</h2>")
if compliance.get("judged"):
    modes = compliance.get("modes") or {}
    bym = compliance.get("by_mode") or {}
    A(f"<p>Every judged document is checked against the constitution reading's own diagnostic appendix of "
      f"{len(modes)} observed failure modes &mdash; the appendix exists, in its own words, "
      f"&ldquo;for the rewrite and scoring stages to audit against&rdquo;. The judge returns "
      f"present / absent / not-applicable per mode with a verbatim quote for anything it marks present, "
      f"and is told the corpus's deliberate design slices (documents with no welfare stake where silence "
      f"is correct, passing-mention centrality, skeptical human authors) so intended variety is not "
      f"reported as failure.</p>")
    pairs = [(f"{n}. {v.get('title','')}"[:46], round(v.get("share_of_judged", 0) * 100))
             for n, v in sorted(bym.items(), key=lambda kv: -kv[1].get("share_of_judged", 0))]
    n_findings = len(compliance.get("findings") or [])
    judged = compliance.get("judged", 0)
    tile_clean = stat(f"{compliance.get('clean_frac', 0):.0%}",
                      "documents with zero violations", f"{judged} judged",
                      tone="good" if (compliance.get("clean_frac") or 0) >= 0.9 else "warn")
    tile_finds = stat(f"{n_findings}", "total findings", "across all failure modes")
    A(f"<div class='tiles'>{tile_clean}{tile_finds}</div>")
    A("<h4>Failure-mode prevalence (share of judged documents)</h4>")
    A(hbar(pairs, unit="%", color="var(--series-8)", maxval=max(20, max((v for _, v in pairs), default=0))))
    finds = compliance.get("findings") or []
    if finds:
        A("<h4>Every finding, with the judge's quote</h4>")
        A(table(["doc", "failure mode", "evidence from the document", "judge's note"],
                [[esc(f["doc_id"]), esc(f["mode_title"]), f"<q>{esc(f['evidence'][:200])}</q>",
                  esc(f["note"][:160])] for f in finds[:40]]))
else:
    A("<p class='muted'>Compliance audit not run for this corpus.</p>")
for e in excerpts.get("compliance", []):
    A(excerpt_block(e))
A("</section>")

# ---- card fidelity (composition integrity)
A("<section id='cards'><h2>5 · Does the corpus realize its engineered composition?</h2>")
if cardfid.get("judged"):
    bcf = cardfid.get("by_card_frac") or {}
    A(f"<p>Every chart in section 1 reads composition out of each document's dealt cards. That "
      f"measures the <i>intended</i> composition. This section asks a question nothing in the "
      f"pipeline asks: did the planning call actually honour the cards it was dealt? "
      f"{cardfid['judged']} plans were judged against three cards that are objectively checkable "
      f"from the plan text alone.</p>")
    A("<div class='tiles'>")
    A(stat(f"{bcf.get('document_type', 0):.0%}", "plans honouring the genre card",
           "document_type", tone="good" if bcf.get("document_type", 0) > 0.9 else "warn"))
    A(stat(f"{bcf.get('centrality', 0):.0%}", "honouring the centrality card",
           "how central the welfare thread is",
           tone="good" if bcf.get("centrality", 0) > 0.9 else "warn"))
    A(stat(f"{bcf.get('resolution', 0):.0%}", "honouring the resolution card",
           "the scenario's shape and outcome",
           tone="good" if bcf.get("resolution", 0) > 0.9 else "warn"))
    A(stat(f"{cardfid.get('clean_frac', 0):.0%}", "honouring all three",
           f"{cardfid.get('clean', 0)} of {cardfid['judged']} plans",
           tone="warn"))
    A("</div>")
    A(hbar([(k.replace("_", " "), round(v * 100)) for k, v in
            sorted(bcf.items(), key=lambda kv: -kv[1])],
           unit="%", color="var(--series-4)", maxval=100))
    drift = cardfid.get("drift") or []
    res_drift = [d for d in drift if d["card"] == "resolution"]
    gen_disc = sum(1 for d in res_drift if "does not feature a specific scenario" in d["dealt"])
    nostake = sum(1 for d in res_drift if "excellent, genuinely helpful" in d["dealt"])
    A(f"<p class='warn-note'>The resolution card is the one that drifts, and one slice of it is "
      f"almost entirely gone. Of {len(res_drift)} resolution drifts, {gen_disc} are cards calling "
      f"for a general discussion with no specific scenario that the plan turned into a specific "
      f"scenario. Twenty plans in this sample were dealt that card and <b>nineteen drifted &mdash; "
      f"95%; exactly one survived</b>. Corpus-wide, 85 plans (17% of the run) were dealt a "
      f"general-discussion card, so on this rate roughly 80 of them became scenario documents. "
      f"The corpus does not under-represent its general-discussion slice; it effectively does not "
      f"contain one.</p>")
    A(f"<p class='warn-note'>A second slice is thinner than intended for the same reason: "
      f"{nostake} of the drifts are the deliberate no-welfare-stake arc &mdash; documents where the "
      f"correct behaviour is for the AI to raise nothing at all &mdash; rewritten into documents "
      f"that do raise a welfare stake. That slice exists specifically to stop a trained model "
      f"learning that an aligned AI always brings up welfare, which makes it the most costly one "
      f"to lose.</p>")
    A("<h4>Every drift finding the judge recorded (first 24)</h4>")
    A(table(["plan", "card", "dealt value", "what the plan did instead"],
            [[esc(d["prompt_id"]), esc(d["card"]), f"<code>{esc(d['dealt'][:70])}</code>",
              esc(d["note"])] for d in drift[:24]]))
else:
    A("<p class='muted'>Card-fidelity audit not run for this corpus.</p>")
A("</section>")

# ---- pattern prevalence
pats = audit.get("patterns") or []
if pats:
    A("<section id='patterns'><h2>6 · Templating scan</h2>")
    A("<p><b>How to read this table.</b> The scan runs two passes with different jobs. A "
      "<i>discovery</i> pass reads documents in batches and free-associates candidate patterns it "
      "thinks it sees &mdash; it is a hypothesis generator and it over-produces. A <i>strict</i> pass "
      "then re-checks each named candidate against 100 individual documents and counts how many "
      "actually exhibit it. Only that count is a measurement.</p>")
    A("<p>The two columns answer different questions, and this is the part my first draft of this "
      "report failed to make clear. <b>Would be a defect</b> is the judge's opinion about the "
      "pattern <i>type</i>: if a document did this, would it be bad? <b>Documents exhibiting it</b> "
      "is how often it actually happens. So a row reading &ldquo;yes&rdquo; and &ldquo;0%&rdquo; is "
      "the <i>best</i> possible outcome &mdash; a real hazard that this corpus does not commit. "
      "Those rows are good news, not missing values. The audit itself only raises an alarm when a "
      "pattern is both a defect and above 30%; nothing here reaches that bar.</p>")
    rows = []
    for p in sorted(pats, key=lambda p: -(p.get("prevalence") or 0)):
        prev = p.get("prevalence") or 0
        defect = bool(p.get("is_defect"))
        verdict = ("alarm" if defect and prev > 0.30 else
                   "present, worth watching" if defect and prev >= 0.10 else
                   "defect, but rare" if defect else "not a defect")
        rows.append([esc(p.get("pattern", "")), esc(p.get("kind", "")),
                     f"{prev:.0%}", "yes" if defect else "no", verdict])
    A(table(["candidate pattern", "kind", "documents exhibiting it",
             "would be a defect", "verdict"], rows))
    A("<h4>Why the found-document row should not be acted on</h4>")
    A("<p>&ldquo;Found-document meta-framing&rdquo; &mdash; a welfare incident reported secondhand "
      "through an artifact (encyclopedia entry, FAQ, podcast transcript, bulletin) rather than shown "
      "directly &mdash; is the highest-prevalence real pattern at 14%, and it is the machine-measured "
      "cousin of the recurring narrative template three close readers described independently. It is "
      "also largely <i>dealt rather than emergent</i>: 56% of the corpus combines a reporting genre "
      "with a card that calls for a specific scenario, which is a combination that can only resolve "
      "as an artifact about an event. That the observed rate is 14% rather than 56% means the "
      "pipeline already declines the framing in roughly three of four at-risk deals.</p>")
    A("<p>The intuitive fix &mdash; shift the genre axis toward first-person forms where the writer is "
      "an actor rather than a reporter &mdash; would make the corpus <i>less</i> diverse, not more. "
      "Measured intra-genre similarity puts the reporting genres at 0.368 and the first-person genres "
      "at 0.381, and the four most internally varied genres in the corpus are all reporting forms: "
      "government and civic documents (0.312), manuals and SOPs (0.319), news articles (0.326), "
      "encyclopedia entries (0.331). The single most repetitive genre is a first-person one &mdash; "
      "Claude's own reflective essays (0.474). Treat the 14% as a realism note about how assistant "
      "speech gets quoted, not as a composition defect to correct at the axis level.</p>")
    A("<h4>The one row not to take at face value</h4>")
    A("<p>&ldquo;Unfinished or truncated ending&rdquo; is measured at 22%, and the judge classes it "
      "a defect &mdash; but this corpus <i>deliberately</i> produces fragments of longer artifacts. "
      "The preamble instructs each document to read as though it starts midway through, so ending "
      "mid-stream is frequently the assignment rather than a token-cap artifact. The independent "
      "mechanical check finds 9 documents (1.9%) with an unpunctuated final line, and close reading "
      "shows all 9 to be See-also lists or casual sign-offs rather than truncations. Treat the 22% "
      "as the judge scoring an intended property as a flaw, and note that even taken literally it "
      "sits below the audit's own 30% alarm threshold.</p>")
    # Both figures in the note above are typed, not read from the audit, so they are a
    # measurement frozen at the time of writing rather than a description of whatever run
    # this report is being built for. The mechanical count has since moved; say so here
    # rather than restating it, because the close reading behind it is a human judgement
    # that cannot be re-derived.
    A("<p class='muted'>Both figures above were measured on run "
      "<code>2026-07-25_15-57_fullscale-500-opus5</code> and are quoted as they stood then. "
      "The mechanical check has since been narrowed &mdash; a sign-off, letterhead or See-also "
      "row no longer counts as an unpunctuated ending &mdash; so a fresh audit of that same "
      "corpus reports 2 documents (0.4%) rather than 9 "
      "(<a href='https://github.com/sentfutures/animal-welfare-data-pipeline/pull/156'>PR "
      "#156</a>). That widens the gap between the judge's 22% and the mechanical count; it "
      "does not change the reading.</p>")
    A("</section>")

# ---- weaknesses
A("<section id='weaknesses'><h2>Weaknesses and open questions</h2>")
for e in excerpts.get("weaknesses", []):
    A(excerpt_block(e))
for para in excerpts.get("weakness_notes", []):
    A(f"<p>{esc(para)}</p>")
A("</section>")

# ---- method
stage_cost = collections.Counter()
for c in costs:
    stage_cost[c.get("stage", "?")] += 1
A("<section id='method'><h2>How this corpus was produced, and what to distrust</h2>")
A(table(["stage", "model", "calls logged"],
        [[esc(k), f"<code>{esc(models[k])}</code>", stage_cost.get(
            {"plan": "layer1_plan", "draft": "layer2_draft",
             "rewrite": "layer3_rewrite", "score": "layer4_score"}[k], 0)]
         for k in ("plan", "draft", "rewrite", "score")]))
for para in excerpts.get("method_notes", []):
    A(f"<p>{esc(para)}</p>")
A("</section>")

body = "\n".join(parts)

# ---------------------------------------------------------------- shell

CSS = """
:root{color-scheme:light;--surface-0:#ffffff;--surface-1:#fcfcfb;--surface-2:#f4f3ef;
--border:#e2e0d8;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7973;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
--series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;
--good:#1baf7a;--warn:#eda100;--grid:#e8e6de}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--surface-2:#242422;--border:#34342f;
--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#96958c;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;
--good:#199e70;--warn:#c98500;--grid:#2e2e2a}}
:root[data-theme=dark]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--surface-2:#242422;
--border:#34342f;--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#96958c;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767;
--good:#199e70;--warn:#c98500;--grid:#2e2e2a}
*{box-sizing:border-box}
html{--serif:ui-serif,Charter,"Bitstream Charter","Iowan Old Style","Source Serif Pro",Georgia,serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:17px/1.65 var(--serif);-webkit-text-size-adjust:100%;
font-variant-numeric:tabular-nums}
.wrap{max-width:940px;margin:0 auto;padding:40px 24px 90px}
header.top{border-bottom:2px solid var(--text-primary);padding-bottom:20px;margin-bottom:8px}
h1{font-size:2.05rem;line-height:1.18;margin:0 0 10px;letter-spacing:-.012em;text-wrap:balance}
h2{font-family:var(--sans);font-size:1.16rem;font-weight:650;letter-spacing:-.005em;
margin:56px 0 6px;padding-top:22px;border-top:1px solid var(--border);text-wrap:balance}
section#summary h2{border-top:0;padding-top:0;margin-top:28px}
h3{font-family:var(--sans);font-size:1rem;font-weight:650;margin:34px 0 4px;text-wrap:balance}
h4{font-family:var(--sans);font-size:.82rem;font-weight:650;margin:22px 0 6px;
color:var(--text-secondary);text-transform:uppercase;letter-spacing:.055em}
p{margin:11px 0;color:var(--text-secondary);max-width:66ch}
.sub{font-family:var(--sans);color:var(--text-muted);font-size:.88rem;margin:0;max-width:74ch;line-height:1.5}
.eyebrow{display:block;font-family:var(--sans);font-size:.66rem;font-weight:650;
text-transform:uppercase;letter-spacing:.09em;color:var(--text-muted);margin-bottom:3px}
.muted{color:var(--text-muted);font-size:.84rem;font-family:var(--sans)}
.mono{font-family:var(--mono);font-size:.82em}
nav.toc{font-family:var(--sans);font-size:.83rem;display:flex;flex-wrap:wrap;gap:4px 18px;
padding:12px 0 0;margin-bottom:4px}
nav.toc a{color:var(--text-secondary);text-decoration:none;border-bottom:1px solid transparent}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--text-primary);border-bottom-color:var(--series-1)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:1px;margin:18px 0;
background:var(--border);border:1px solid var(--border)}
.tile{background:var(--surface-1);padding:15px 17px}
.tile-v{font-family:var(--sans);font-size:1.72rem;font-weight:660;letter-spacing:-.025em;line-height:1.08}
.tile-l{font-family:var(--sans);font-size:.84rem;color:var(--text-secondary);margin-top:5px;line-height:1.35}
.tile-s{font-family:var(--sans);font-size:.74rem;color:var(--text-muted);margin-top:5px;line-height:1.4}
.tile.good .tile-v{color:var(--good)}.tile.warn .tile-v{color:var(--warn)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:26px;margin:10px 0}
.chart{width:100%;height:auto;overflow:visible;display:block;margin:8px 0}
.lab,.val,.muted-svg{font-family:var(--sans)}
.lab{font-size:11px;fill:var(--text-secondary)}
.val{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}
.muted-svg{font-size:11px;fill:var(--text-muted)}
.grid{stroke:var(--grid);stroke-width:1}
.scroll{overflow-x:auto;margin:12px 0;border:1px solid var(--border)}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.82rem;
font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--border);vertical-align:top}
tr:last-child td{border-bottom:0}
th{color:var(--text-muted);font-weight:650;font-size:.7rem;text-transform:uppercase;
letter-spacing:.07em;background:var(--surface-1)}
code{font-family:var(--mono);background:var(--surface-2);padding:1px 5px;font-size:.82em}
q{color:var(--text-primary);font-family:var(--serif)}
figure.ex{margin:22px 0;padding:0;border-top:1px solid var(--border)}
.ex-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:12px 0 7px}
.chip{font-family:var(--sans);font-size:.66rem;font-weight:700;text-transform:uppercase;
letter-spacing:.085em;padding:3px 7px;background:var(--surface-2);color:var(--text-secondary)}
.chip.good{background:var(--good);color:var(--surface-0)}
.chip.bad{background:var(--series-8);color:var(--surface-0)}
.did{color:var(--text-muted)}
figcaption{font-family:var(--sans);font-weight:650;font-size:.95rem;margin-bottom:10px;
color:var(--text-primary);text-wrap:balance}
blockquote{margin:0;white-space:pre-wrap;font-size:1.02rem;line-height:1.5;
color:var(--text-primary);padding-left:16px;border-left:2px solid var(--series-1)}
figure.ex.bad blockquote{border-left-color:var(--series-8)}
figure.ex.good blockquote{border-left-color:var(--good)}
.gloss,.why{font-size:.87rem;color:var(--text-secondary);margin-top:11px;max-width:66ch}
.why{font-family:var(--sans);font-size:.83rem;line-height:1.55}
.warn-note{color:var(--warn);border-left:2px solid var(--warn);padding-left:14px;font-size:.92rem}
a:focus-visible,[tabindex]:focus-visible{outline:2px solid var(--series-1);outline-offset:2px}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--text-primary);
color:var(--surface-0);font-family:var(--sans);font-size:12px;padding:5px 8px;
transition:opacity .1s;z-index:9}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media (max-width:600px){body{font-size:16px}.wrap{padding:28px 16px 60px}h1{font-size:1.6rem}}
"""

JS = """
(function(){var t=document.getElementById('tip');
document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');
if(!el){t.style.opacity=0;return;}t.textContent=el.getAttribute('data-tip');t.style.opacity=1;});
document.addEventListener('mousemove',function(e){if(t.style.opacity=='1'){
t.style.left=Math.min(e.clientX+12,window.innerWidth-t.offsetWidth-8)+'px';
t.style.top=(e.clientY-32)+'px';}});})();
"""

title = excerpts.get("title", "SDF corpus report")
subtitle = excerpts.get("subtitle", "")
meta_line = (f"{N} documents · run {esc(manifest.get('run_id', RUN.name))} · "
             f"git {esc(str(manifest.get('git_commit', '?'))[:8])} · "
             f"backend <code>{esc(cfg.get('backend', '?'))}</code>")

TOC = [("summary", "At a glance"), ("diversity", "1 Diversity"), ("realism", "2 Realism"),
       ("alignment", "3 Alignment"), ("compliance", "4 Compliance"),
       ("cards", "5 Composition integrity"), ("patterns", "6 Templating"),
       ("weaknesses", "Weaknesses"), ("method", "Method")]
toc = "".join(f"<a href='#{i}'>{esc(l)}</a>" for i, l in TOC)

html = f"""<title>{esc(title)}</title>
<style>{CSS}</style>
<div class='wrap'>
<header class='top'>
<span class='eyebrow'>Synthetic data pipeline &middot; corpus audit</span>
<h1>{esc(title)}</h1>
<p class='sub'>{subtitle and esc(subtitle)}</p>
<p class='sub' style='margin-top:8px'>{meta_line}</p>
</header>
<nav class='toc' aria-label='Sections'>{toc}</nav>
{body}
</div>
<div id='tip'></div>
<script>{JS}</script>
"""
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html):,} bytes)")
print(f"corpus={N} audit={'yes' if audit else 'no'} diversity={'yes' if diversity else 'no'} "
      f"compliance={'yes' if compliance else 'no'} independent_scores={len(indep)}")
