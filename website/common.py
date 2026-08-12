"""Loading, prose and CLI plumbing shared by the per-pipeline report modules.

Same contract as render.py: stdlib only, no repo imports, no pipeline knowledge — the
report generators have to build in an environment where the pipeline's own
dependencies are not installed, which is also what makes them portable.

Everything here is used by website/page.py and website/dad.py today. Anything that only
one pipeline needs stays in that pipeline's module: in particular the weaknesses floor
splits in two, because ``evals/audit_dad.py`` records its verdicts into
``sections[].rows[]`` and ``evals/audit_sdf.py`` only prints them. So
``audit_verdict_warnings()`` returns nothing for an SDF audit, and an SDF page will
have to compute its own thresholds.
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

from website import render as R


# ------------------------------------------------------------------ loading

# A run snapshots the prompts it ran with into inputs/prompts/, which is the honest
# source for "how many prompts is this pipeline" — the question a reader who wants to
# run it against their own model is actually asking. Counted: the stage templates only.
# Not the variables matrix (a weighted table, not a prompt), not the reasoning library,
# not archive/, and not *_score.txt, which is an eval rather than a generation stage.
def prompt_count(run_dir, glob):
    """How many stage templates the run was generated with, or None if it kept no
    snapshot of them."""
    snapshot = Path(run_dir) / "inputs" / "prompts"
    if not snapshot.is_dir():
        return None
    n = sum(1 for p in snapshot.glob(glob) if not p.name.endswith("_score.txt"))
    return n or None


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ------------------------------------------------------------------ prose

def parse_content(text, ids):
    """A prose file -> {section_id: markdown}, delimited by ``<!-- id: name -->``.

    ``ids`` is the owning module's tuple. An unknown or missing id raises, so a typo
    can never silently drop a section from a page.
    """
    parts = re.split(r"<!--\s*id:\s*([a-z0-9_]+)\s*-->", text)
    if len(parts) < 3:
        raise ValueError("content file has no '<!-- id: ... -->' section markers")
    found = {}
    for i in range(1, len(parts), 2):
        found[parts[i]] = parts[i + 1].strip()
    unknown = sorted(set(found) - set(ids))
    if unknown:
        raise ValueError(f"content file has unknown section id(s): {', '.join(unknown)}")
    missing = sorted(set(ids) - set(found))
    if missing:
        raise ValueError(f"content file is missing section id(s): {', '.join(missing)}")
    return found


def load_content(paths, ids):
    """Merge one or more prose files into one id namespace.

    Two files may not both own a section, and the union must be exactly ``ids`` — so
    moving a block from a per-pipeline file into the shared one is a rename, never a
    silent duplicate.
    """
    merged, seen = {}, {}
    texts = [(Path(p), Path(p).read_text(encoding="utf-8")) for p in paths]
    all_found = {}
    for path, text in texts:
        parts = re.split(r"<!--\s*id:\s*([a-z0-9_]+)\s*-->", text)
        if len(parts) < 3:
            raise ValueError(f"{path} has no '<!-- id: ... -->' section markers")
        for i in range(1, len(parts), 2):
            sid = parts[i]
            if sid in seen:
                raise ValueError(f"section id '{sid}' is defined in both {seen[sid]} and {path}")
            seen[sid] = path
            all_found[sid] = parts[i + 1].strip()
    unknown = sorted(set(all_found) - set(ids))
    if unknown:
        raise ValueError(f"unknown section id(s) across {', '.join(str(p) for p, _ in texts)}: "
                         f"{', '.join(unknown)}")
    missing = sorted(set(ids) - set(all_found))
    if missing:
        raise ValueError(f"missing section id(s): {', '.join(missing)}")
    merged.update(all_found)
    return merged


_PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def fill(text, f):
    """Resolve {{placeholders}} from the facts dict. Unknown key -> build error.

    This is the enforcement half of "no number is ever typed into the prose": a figure
    can only reach the page by being computed from the run's own output, and a prose
    file that references a fact the run does not have fails the build rather than
    shipping a stale sentence.
    """
    def sub(m):
        key = m.group(1)
        if key not in f:
            raise KeyError(f"prose references unknown fact '{{{{{key}}}}}' "
                           f"(available: {', '.join(sorted(f))})")
        return str(f[key])
    return _PLACEHOLDER.sub(sub, text or "")


def prose(content, key, f):
    return R.paragraphs(fill(content.get(key, ""), f))


def section(sid, heading, *blocks, heading_class=""):
    """A section. A falsy heading omits the <h2>.

    ``heading_class='vh'`` renders the heading for screen readers only — for a section
    whose own content is its title, like the comparison, whose two column mastheads say
    what it is on screen. Omitting the heading entirely was the older answer and it cost
    a reader navigating by heading the whole comparison: pressing H went from the page
    title to the chooser, past both datasets.
    """
    body = "".join(b for b in blocks if b)
    cls = f" class='{heading_class}'" if heading_class else ""
    head = f"<h2{cls}>{R.esc(heading)}</h2>" if heading else ""
    return f"<section id='{sid}'>{head}{body}</section>"


_STRIP_BLOCKS = re.compile(
    r"<(script|style|svg|nav)\b.*?</\1>|<blockquote\b.*?</blockquote>"
    r"|<div class='resp'>.*?</div>|<table\b.*?</table>|<!--.*?-->", re.S)


def editorial_words(html):
    """How many words of authored prose a built page carries.

    Corpus text, chart internals, every table — including the derived warnings, whose
    wording comes from the audit — and every ``<nav>`` are excluded, so what is counted is
    the part a person wrote. A rail's labels are the document's own headings, already
    counted where they are written; counting them twice would spend the ceiling on
    navigation and let real prose in under it.

    Printed at build time: the page's whole brief is that a reader with forty seconds gets
    what they need, and prose is the thing that grows back.
    """
    text = _STRIP_BLOCKS.sub(" ", html or "")
    return len(re.findall(r"[A-Za-z][A-Za-z'’-]*", re.sub(r"<[^>]+>", " ", text)))


# ------------------------------------------------------------------ diff

# Both pipelines end on a rewrite of something a previous stage drafted, and both reports
# show a reader what that rewrite did — so this lives here rather than in either module.
_DIFF_CSS = ("<style>ins{background:var(--mark);text-decoration:none}"
             "del{opacity:.5;text-decoration:line-through}</style>")


def _opcodes(before, after):
    a, b = (before or "").split(), (after or "").split()
    return a, b, difflib.SequenceMatcher(None, a, b).get_opcodes()


def changed_fraction(before, after):
    """How much of the output the rewrite touched, 0-1.

    Its own function because the two pipelines' rewrites behave differently and a renderer
    has to be able to ask: the difficult-advice rewrite edits an answer, while the document
    rewrite is licensed to start again from the premise, and a hunk view of a from-scratch
    rewrite is confetti.
    """
    a, b, ops = _opcodes(before, after)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")
    return changed / max(len(b), 1)


def diff_summary(before, after):
    a, b, _ = _opcodes(before, after)
    return (f"The rewrite touched {changed_fraction(before, after):.0%} of the words "
            f"({len(a):,} words in, {len(b):,} out).")


def _render_ops(a, b, ops):
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            out.append(R.esc(" ".join(b[j1:j2])))
        else:
            if tag in ("replace", "delete"):
                out.append(f"<del>{R.esc(' '.join(a[i1:i2]))}</del>")
            if tag in ("replace", "insert"):
                out.append(f"<ins>{R.esc(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def word_diff(before, after):
    """Full word-level diff. Lives in an appendix — informative, but as running text
    it is confetti, and it was a third of the page."""
    a, b, ops = _opcodes(before, after)
    return _DIFF_CSS + f"<div class='resp'>{_render_ops(a, b, ops)}</div>"


def diff_hunks(before, after, *, top=3, context=16):
    """The N largest changed runs, each with surrounding context.

    A reader wants to know what the rewrite does, which three concrete edits answer and a
    full diff buries.
    """
    a, b, ops = _opcodes(before, after)
    changes = [op for op in ops if op[0] != "equal"]
    if not changes:
        return "<p class='muted'>The rewrite changed nothing.</p>"
    biggest = sorted(changes, key=lambda op: -max(op[2] - op[1], op[4] - op[3]))[:top]
    biggest = sorted(biggest, key=lambda op: op[3])
    out = []
    for tag, i1, i2, j1, j2 in biggest:
        pre = " ".join(b[max(0, j1 - context):j1])
        post = " ".join(b[j2:j2 + context])
        mid = _render_ops(a, b, [(tag, i1, i2, j1, j2)])
        out.append(f"<div class='resp'>… {R.esc(pre)} {mid} {R.esc(post)} …</div>")
    return _DIFF_CSS + "".join(out)


# ------------------------------------------------------------------ cost

def costs_by_stage(costs):
    agg = {}
    for rec in costs or []:
        stage = rec.get("stage") or "(untagged)"
        entry = agg.setdefault(stage, {"calls": 0, "cost": 0.0, "models": set()})
        entry["calls"] += 1
        entry["cost"] += rec.get("cost_usd") or 0.0
        if rec.get("model"):
            entry["models"].add(rec["model"])
    return agg


def stage_cost_table(costs, labels):
    """(tag, display name) pairs in pipeline order -> the per-stage cost table.

    Stages the labels don't name are appended rather than dropped, so a new stage tag
    shows up as itself instead of vanishing.
    """
    agg = costs_by_stage(costs)
    if not agg:
        return ""
    rows = []
    for tag, label in labels:
        entry = agg.get(tag)
        if entry:
            rows.append((label, ", ".join(sorted(entry["models"])) or "—",
                         entry["calls"], f"${entry['cost']:,.2f}"))
    for tag in sorted(set(agg) - {t for t, _ in labels}):
        rows.append((tag, ", ".join(sorted(agg[tag]["models"])) or "—",
                     agg[tag]["calls"], f"${agg[tag]['cost']:,.2f}"))
    return R.table(["stage", "model", "calls", "cost"], rows, align="llrr")


# ------------------------------------------------------------------ provenance

def run_note(run_id, *, n=None, lead):
    """Which run the blocks below came off, as one muted line.

    A report is written about a pipeline, but its worked example and every figure in its
    appendix are one run's. Nothing on the page said which, so a reader had no way to tell
    a property of the pipeline from a property of one batch — and the example carousel's
    "the same run" pointed at a run that had never been introduced.

    Derived, never authored: the caller supplies the sentence opener, this fills in the
    run directory's own name and the audit's own count. Empty without a run id, so a build
    that has none loses the line rather than shipping a dangling sentence.
    """
    if not run_id:
        return ""
    count = f", {n:,} examples" if n else ""
    return (f"<p class='muted'>{R.esc(lead)} <span class='mono'>{R.esc(run_id)}</span>"
            f"{count}.</p>")


# ------------------------------------------------------------------ candour floor

# The non-faithful backends this repository can still be run on. `api` is the faithful
# mode; `bedrock` was removed from the pipeline, so a run produced on it is not something a
# reader can act on — the row only sent them looking for a backend that is not in the code,
# and which run the numbers came off is said plainly by ``run_note()`` instead.
UNFAITHFUL_BACKENDS = ("claude_code", "auto")


def provenance_warnings(manifest, *, n=None, small_n=100):
    """The warnings that are true of any run of any pipeline.

    Severity, not prose, is the contract: these are appended to whatever the audit
    itself flagged, and nothing is ever filtered back out.
    """
    out = []
    backend = ((manifest or {}).get("config") or {}).get("backend")
    if backend in UNFAITHFUL_BACKENDS:
        out.append(("BAD" if backend == "claude_code" else "OK",
                    f"Generated on the `{backend}` backend rather than `api`. `api` is the "
                    "documented faithful mode, and the one a reader reproducing this would "
                    "use. Read these numbers as representative, not exact."))
    if n and n < small_n:
        out.append(("OK", f"n = {n}, from one run on one seed. Every percentage here is "
                          f"indicative."))
    return out


def audit_verdict_warnings(audit):
    """Every BAD or OK row the audit itself recorded.

    Returns [] for an audit with no verdict rows, which is what an SDF audit looks
    like today — that pipeline's page has to derive its own thresholds instead.
    """
    out = []
    for sec in (audit or {}).get("sections") or []:
        for row in sec.get("rows") or []:
            if row.get("verdict") in ("BAD", "OK"):
                out.append((row["verdict"], f"{sec.get('title', '?')} — "
                                            f"{row.get('label', '')}: {row.get('value', '')}"
                                            + (f" {row.get('note')}" if row.get("note") else "")))
    return out


def warnings_table(warnings, *, inline=3, drawer_label="more findings at this level"):
    """BADs first, then the most severe OKs; the rest in a counted drawer.

    The drawer exists so the page is skimmable, and it is COUNTED so that collapsing
    is visibly a view and not a filter. The list itself is never trimmed.
    """
    if not warnings:
        return ""
    ordered = sorted(warnings, key=lambda w: 0 if w[0] == "BAD" else 1)
    bads = [w for w in ordered if w[0] == "BAD"]
    rest = [w for w in ordered if w[0] != "BAD"]
    head, tail = bads + rest[:inline], rest[inline:]

    def build(ws):
        return R.table(["severity", "what the data says"],
                       [(R.Raw(R.chip(sev, "bad" if sev == "BAD" else "warn")),
                         R.Raw(R.inline_md(text))) for sev, text in ws])

    out = build(head)
    if tail:
        out += R.details(f"{len(tail)} {drawer_label}", build(tail))
    return out


# ------------------------------------------------------------------ semantic diversity

def semantic_figures(diversity, *, unit="records"):
    """Meanings and topics: the corpus audit viewer's two-chart pair, mirrored.

    Redundancy (each record's nearest-neighbour cosine, where past 0.90 is a
    near-duplicate) and topic spread (meaning-cluster sizes, largest first), with the
    viewer's own captions, then the Vendi effective-count as a sentence. Renders from
    the per-record ``scopes.combined`` data, so a run whose diversity report predates
    that field gets nothing rather than an approximation. Both reports use this;
    ``unit`` is the word for one record ("records" / "documents").
    """
    scope = ((diversity or {}).get("scopes") or {}).get("combined") or {}
    sims, clusters = scope.get("nn_sims") or [], scope.get("clusters") or {}
    if not sims:
        return ""
    one = unit.rstrip("s")
    over = scope.get("over") or {}
    out = [
        "<h4>Meanings and topics</h4>",
        f"<p class='muted'>Similarity is measured with embeddings, so two {unit} count "
        "as alike when they cover the same subject even in completely different words. "
        f"Embedding model: <code>{R.esc(diversity.get('embed_model', '?'))}</code>.</p>"]
    # Fixed 0.05 buckets ending at 1.00, so the 0.90 near-duplicate threshold is a
    # bucket edge and an empty right-hand tail stays visible rather than cropped.
    lo = min(0.5, min(sims))
    edges = [round(lo + i * 0.05, 2) for i in range(int(round((1.0 - lo) / 0.05)))]
    buckets = [(f"{a:.2f}", sum(1 for s in sims if a <= s < round(a + 0.05, 2)))
               for a in edges]
    out.append(R.figure(
        title=f"Redundancy — how close each {one} sits to its nearest neighbour",
        chart=R.histogram(buckets, xlabel="nearest-neighbour cosine similarity"),
        caption=f"**{over.get('0.90', 0):.0%} near-duplicate** (similarity above 0.90), "
                f"{over.get('0.80', 0):.0%} similar (above 0.80). Lower is more varied."))
    sizes = clusters.get("sizes") or []
    if sizes:
        k = clusters.get("k") or len(sizes)
        out.append(R.figure(
            title=f"Topic spread — the {unit} grouped into meaning clusters",
            chart=R.histogram([(str(i + 1), s) for i, s in enumerate(sizes)],
                              xlabel="clusters, largest first"),
            caption=f"**Evenness {clusters.get('evenness', 0):.3f} across {k} clusters**, "
                    f"the largest holding {clusters.get('largest_share', 0):.0%} of "
                    f"{unit}. Many even bars mean many distinct topics; one tall bar "
                    "means they clump onto a single one."))
    n, vr = scope.get("n") or 0, scope.get("vendi_ratio") or 0
    if n and vr:
        out.append("<p>" + R.inline_md(
            f"**{vr * n:.1f} of {n} {unit} effectively distinct** in meaning "
            f"(Vendi ratio {vr:.2f}). Higher is more varied.") + "</p>")
    detail = clusters.get("detail") or []
    if detail:
        out.append(R.details(
            "What each cluster is",
            f"<p class='muted'>Clusters are unlabelled groups of {unit} with similar "
            "meaning, numbered to match the topic-spread bars (largest first). Each is "
            "shown by its most central record — a typical member, not a name for the "
            "group.</p>"
            + R.table(["cluster", unit, "most central record"],
                      [(f"{i + 1}", f"{d.get('size', '?')}",
                        f"{d.get('rep_id', '?')} — “{d.get('rep', '')}”")
                       for i, d in enumerate(detail)], align="rrl"),
            meta=f"{len(detail)} clusters"))
    return "".join(out)


# ------------------------------------------------------------------ shell bits

# ------------------------------------------------------------------ CLI

def write(path, html, *, label=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path} ({len(html):,} bytes){' — ' + label if label else ''}")
    return path


def cli_parser(doc):
    p = argparse.ArgumentParser(description=(doc or "").strip().split("\n")[0])
    p.add_argument("--dad-run", "--run", dest="dad_run", default=None,
                   help="DAD run directory (required)")
    p.add_argument("--sdf-run", dest="sdf_run", default=None,
                   help="SDF run directory. Optional: without it the document corpus's "
                        "column and section say so instead of showing figures")
    p.add_argument("--out-dir", default=None, help="output directory (default website/)")
    p.add_argument("--content", action="append", default=None,
                   help="prose file, repeatable; overrides the page's default prose file(s)")
    p.add_argument("--example", default=None, help="prompt_id to feature as the worked example")
    p.add_argument("--sdf-example", dest="sdf_example", default=None,
                   help="doc_id to feature as the document report's worked example")
    # Where the built page is served from, for the link-preview tags only. Without it the
    # page says nothing about where it lives, which is right for the copy that opens from
    # disk or arrives attached to an email.
    p.add_argument("--site-url", dest="site_url", default=None,
                   help="public URL of the hosted page; adds og:/twitter: preview tags")
    p.add_argument("--preview-url", dest="preview_url", default=None,
                   help="absolute URL of the preview image (og:image). Defaults to "
                        "preview.png beside the page, which build_website.py copies out; "
                        "pass one to point at an image hosted elsewhere instead")
    return p


def die(msg):
    sys.exit(msg)
