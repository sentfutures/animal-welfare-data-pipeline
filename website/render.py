"""Presentation primitives for the standalone HTML reports: CSS, SVG charts, shell.

Knows nothing about any pipeline — it takes numbers and returns HTML strings, so both
datasets' reports on the handoff page share one look.

ONE THEME, LIGHT, deliberately. This is a document that gets handed to an external
reader, printed, and screenshotted into slides. A viewer's OS preference is not a
signal about how a published artefact should look, and the automatic dark flip of a
page whose whole visual argument is warm paper and ink produced a page its author had
never reviewed. ``color-scheme:only light`` (not bare ``light``) is what opts the page
out of Chrome-Android and Samsung Internet's auto-darkening, which is a separate
mechanism from ``prefers-color-scheme``.

Output is ONE self-contained file: no external CSS, JS, fonts, or images. An artifact
host's CSP blocks every external origin, and the file has to survive being downloaded
and opened offline. Charts are therefore inline <svg> generated here rather than a
charting library, and the only JS is a tooltip and the chooser.

stdlib only, and no repo imports: the report generator must run anywhere, including
where the pipeline's own dependencies are not installed.
"""

import re

# Series colors stay CSS custom properties rather than literal hexes. With one theme
# the original reason (a light and a dark value per slot) is gone; four live ones are
# not: there are ~40 fill sites, so retuning a hue for the paper surface is one line
# instead of forty; naming the roles is what makes "a series hue must never mean
# good" testable; --surface-0 is used INSIDE the svg for segment gaps and mark rings,
# so it has to track the surface; and @media print neutralizes every tinted surface in
# one block.
PAL = [f"var(--series-{i})" for i in range(1, 9)]

# The two arms, everywhere. Plain = warm/terracotta, pipeline = green.
PLAIN = "var(--series-2)"
PIPELINE = "var(--series-3)"
ARM_COLORS = {"plain": PLAIN, "plain Claude": PLAIN, "pipeline": PIPELINE}
# Pass this as hbar(color=...) for any (control, pipeline) chart. Without it hbar
# falls back to PAL[i], which colors bars by ROW ORDER — so the headline chart used
# to paint the pipeline in the control's own color.
ARM_PAIR = (PLAIN, PIPELINE)


class Raw(str):
    """HTML that is already built and must not be escaped again.

    ``table()`` escapes every cell by default — wrap pre-built markup in ``Raw`` to
    opt out.
    """


def esc(s):
    if isinstance(s, Raw):
        return str(s)
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


_MD_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITAL = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
# `[^Name of the work](url)` — a citation marker. The name is the accessible
# name; the visible mark is a number the renderer counts out.
_MD_CITE = re.compile(r"\[\^([^\]]+)\]\(([^)\s]+)\)")
# "1. ", "2) " — the marker of a numbered list line, stripped before the item renders.
_MD_ORDERED = re.compile(r"^\d+[.)]\s+")


def inline_md(text):
    """Escape, then apply a bold/italic/code/link/citation subset of markdown.

    Used on prose only — editorial copy and LLM-written judge notes, which contain
    ``**bold**``. NEVER used on corpus text, which must render verbatim.
    """
    out = esc(text)
    out = _MD_CODE.sub(r"<code>\1</code>", out)
    out = _MD_BOLD.sub(r"<b>\1</b>", out)
    out = _MD_ITAL.sub(r"<i>\1</i>", out)
    out = _cite_markers(out)          # before _MD_LINK, or it eats the [^Name](url) form
    out = _MD_LINK.sub(_link, out)
    return out


def plain_md(text):
    """The same prose as flat text, for an attribute where markup cannot render.

    Not a general converter: it undoes exactly the subset ``inline_md`` renders, so one
    authored sentence can serve both the document and a head tag. Whitespace collapses,
    because a ``content`` attribute is one line however the prose file wrapped it.
    """
    out = _MD_CITE.sub(r"\1", text or "")
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_CODE.sub(r"\1", out)
    out = _MD_BOLD.sub(r"\1", out)
    out = _MD_ITAL.sub(r"\1", out)
    return " ".join(out.split())


def _cite_markers(text):
    """``[^Name of the work](url)`` -> a raised, numbered citation marker.

    THE AUTHOR WRITES THE NAME AND THE RENDERER DRAWS THE NUMBER, which is the whole point of
    the form: the visible mark is a superscript numeral, so the claim it hangs off reads
    uninterrupted, while the work's name becomes the marker's accessible name. Written as
    ``[1](url)`` instead, the link announces as "link, 1" and a links list gets a row that
    says nothing — the number is a position, not a name.

    Numbered per call, which is per prose block: two markers in one paragraph are 1 and 2. A
    page-wide sequence would need state threaded through every renderer, and nothing here
    cites across blocks.

    The marker promises a note at the foot of the page and there is none — it links straight
    out. That is a borrowed convention, and the title attribute is the disclosure: hovering
    names the work.

    Each marker is preceded by a ``WORD_JOINER``, which is what stops a line breaking between
    two of them or between a marker and its word — see that constant.
    """
    n = [0]

    def one(m):
        name, href = m.group(1), m.group(2)
        n[0] += 1
        return (f"{WORD_JOINER}<a class='cite-n' href='{href}'{NEW_TAB} aria-label='{name}' "
                f"title='{name}'><sup>{n[0]}{CITE_ARROW}</sup></a>")

    return _MD_CITE.sub(one, text)


# Leaving the page means leaving it in a NEW TAB: this is a long read whose chooser
# state lives in the URL, and a reader who follows a link out and comes back with the
# back button lands on a page that has closed itself again.
NEW_TAB = " target='_blank' rel='noopener noreferrer'"

# The outbound mark, drawn rather than typed. As a glyph (U+2197) it is a hairline in
# most faces and a different shape in every one; this page is printed and screenshotted,
# so the mark has to be the same weight as the type it sits beside, everywhere.
# The marker's own arrow: same path, drawn smaller and a touch heavier in stroke, because at
# the marker's .72em a 9px glyph is wider than the numeral it follows and 2px of stroke on a
# 6px box reads as a blob.
# A marker never starts a line, and two of them never split. The arrow inside each one is
# `display:inline-block` — an ATOMIC INLINE, which UAX#14 treats as an object replacement and
# allows a line to break either side of. Measured at 390px: the intro broke after marker 1,
# leaving marker 2 to open the next line. A word joiner before each marker forbids exactly
# that, because LB11 (`× WJ`, `WJ ×`) is applied ahead of LB20's break-around-CB, and it also
# glues the marker to the word it hangs off, which is the ordinary setting for a footnote
# mark. Its pair is `.cite-n{white-space:nowrap}`, which holds the numeral to its own arrow.
# Written as an entity: it is invisible in the source either way, and this says which
# character it is.
WORD_JOINER = "&#8288;"

CITE_ARROW = ("<svg class='ext-c' viewBox='0 0 12 12' width='7' height='7' aria-hidden='true' "
              "fill='none' stroke='currentColor' stroke-width='2.4' stroke-linecap='round' "
              "stroke-linejoin='round'><path d='M3.1 8.9 8.9 3.1'/>"
              "<path d='M4.6 3.1h4.3v4.3'/></svg>")

EXT_ARROW = ("<svg class='ext' viewBox='0 0 12 12' width='9' height='9' aria-hidden='true' "
             "fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' "
             "stroke-linejoin='round'><path d='M3.1 8.9 8.9 3.1'/>"
             "<path d='M4.6 3.1h4.3v4.3'/></svg>")


def _link(m):
    """A link. One that leaves the page says so, with the arrow that means exactly that.

    On a page that is a single file, "does this take me somewhere else" is the only
    distinction between links that matters, so it is marked rather than left to the
    reader to guess from the href.
    """
    label, href = m.group(1), m.group(2)
    if not href.startswith("http"):
        return f"<a href='{href}'>{label}</a>"
    return f"<a href='{href}'{NEW_TAB}>{label}{EXT_ARROW}</a>"


def paragraphs(text):
    """Blank-line-separated prose to <p>/<ul>/<h3>/dek blocks, with inline markdown.

    Conventions, all of which the prose files use: a block whose lines all start with
    ``- `` is a list, and one whose lines all start ``1. ``/``2. `` is a NUMBERED list;
    a block opening ``### `` is a subhead; a block opening ``> `` is a dek — the one-line
    finding that sits under a heading.

    The numbered form exists because the intro names two techniques and then counts them
    off; rendering that as bullets loses the count the sentence above it just promised.
    The digits are not read — an ``<ol>`` numbers itself — so a mis-numbered source list
    still renders 1, 2, 3.

    The dek is built here rather than by a ``dek()`` of its own: the prose convention is
    the only way one is ever made, and a second public route to the same markup is how a
    page that allows two of them ends up with five.
    """
    blocks = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        if all(ln.startswith("- ") for ln in lines):
            items = "".join(f"<li>{inline_md(ln[2:])}</li>" for ln in lines)
            blocks.append(f"<ul>{items}</ul>")
        elif all(_MD_ORDERED.match(ln) for ln in lines):
            items = "".join(f"<li>{inline_md(_MD_ORDERED.sub('', ln))}</li>" for ln in lines)
            blocks.append(f"<ol>{items}</ol>")
        elif lines[0].startswith("> "):
            line = inline_md(" ".join(ln.lstrip("> ") for ln in lines))
            blocks.append(f"<p class='dek'>{line}</p>")
        elif lines[0].startswith("### "):
            head = inline_md(lines[0][4:])
            rest = " ".join(lines[1:])
            blocks.append(f"<h3>{head}</h3>" + (f"<p>{inline_md(rest)}</p>" if rest else ""))
        else:
            blocks.append(f"<p>{inline_md(' '.join(lines))}</p>")
    return "".join(blocks)


def chip(text, tone=""):
    return f"<span class='chip{' ' + tone if tone else ''}'>{esc(text)}</span>"


def note(text, tone="warn"):
    """A called-out caveat. The report's candour depends on these being visually
    unmissable rather than gray small print."""
    return f"<p class='{tone}-note'>{inline_md(text)}</p>"


def details(summary, body, meta="", open_=False):
    """A drawer. ``meta`` names the payload's size, so collapsing costs nothing:
    "Full pipeline answer · 1,010 words". <details> needs no JS and prints open."""
    label = esc(summary) + (f" <span class='sum-m'>{esc(meta)}</span>" if meta else "")
    return (f"<details{' open' if open_ else ''}><summary>{label}</summary>"
            f"<div class='det-body'>{body}</div></details>")


def stat(value, label, sub="", flag="", tone=""):
    """A number. Direction is carried by a labelled chip, never by coloring the
    numeral: a status color must not travel alone."""
    cls = "tile hero" if tone == "hero" else "tile"
    return (f"<div class='{cls}'><div class='tile-v'>{esc(value)}</div>"
            f"<div class='tile-l'>{esc(label)}</div>"
            + (f"<div class='tile-s'>{esc(sub)}</div>" if sub else "")
            + (f"<div class='tile-f'>{chip(flag, tone if tone != 'hero' else '')}</div>"
               if flag else "") + "</div>")


def tiles(items):
    kept = [t for t in items if t]
    if not kept:
        return ""
    if sum(1 for t in kept if "tile hero" in t) > 1:
        raise ValueError("a tile row may have at most one hero tile")
    return f"<div class='tiles'>{''.join(kept)}</div>"


def table(headers, rows, cls="", align=""):
    """Cells are escaped; wrap pre-built markup in Raw() to pass it through.

    ``align`` is one character per column — l/r/c. Numeric columns should be r, so
    magnitudes line up and a delta reads down a single column.
    """
    def cell(tag, i, v):
        a = align[i] if i < len(align) else "l"
        klass = f" class='{ {'r': 'num', 'c': 'ctr'}[a] }'" if a in "rc" else ""
        return f"<{tag}{klass}>{esc(v)}</{tag}>"

    th = "".join(cell("th", i, h) for i, h in enumerate(headers))
    trs = "".join("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(r)) + "</tr>"
                  for r in rows)
    return (f"<div class='scroll'><table class='{cls}'><thead><tr>{th}</tr></thead>"
            f"<tbody>{trs}</tbody></table></div>")


_SVG_OPEN = re.compile(r"(<svg\b[^>]*>)")


def figure(*, title, chart, caption="", note_="", table_html=None, table_label="Show the numbers"):
    """A chart with its title, caption and optional table view, as one unit.

    The title is a <figcaption>, not a heading: chart titles were polluting the
    document outline and the section list. The caption states the FINDING; axis
    descriptions go in ``note_``, above the chart, where they are read before it
    rather than after. The table view is the relief a chart with a sub-3:1 series
    needs, and the only way a touch user reaches the tooltip's numbers.
    """
    named = _SVG_OPEN.sub(lambda m: m.group(1) + f"<title>{esc(title)}</title>", chart, count=1)
    return ("<figure>"
            f"<figcaption class='fig-t'>{esc(title)}</figcaption>"
            + (f"<p class='fig-n'>{inline_md(note_)}</p>" if note_ else "")
            + named
            + (f"<figcaption class='fig-c'>{inline_md(caption)}</figcaption>" if caption else "")
            + (details(table_label, table_html) if table_html else "")
            + "</figure>")


def _no_data(msg="not measured on this run"):
    return f"<p class='muted'>{esc(msg)}</p>"


W = 800  # every chart is drawn at the figure track's own width, so an 11px label is
         # 11px in every figure instead of scaling with the column it lands in.


def _bar(x, y, w, h, fill, tip, r=3):
    """A bar rounded on the value end only. Rounding the baseline end too made every
    bar look like a lozenge floating free of its axis."""
    r = max(0.0, min(r, w / 2, h / 2))
    d = (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} "
         f"V{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} "
         f"H{x:.1f} Z")
    return f"<path d='{d}' fill='{fill}' data-tip='{esc(tip)}'/>"


def hbar(pairs, *, unit="", width=W, row=28, color=None, maxval=None, fmt="{:g}", label_w=240):
    """Horizontal bars: magnitude by identity. Labels outside, value at the bar end.

    ``color`` takes a single color for every bar OR a sequence indexed by row — pass
    ARM_PAIR for a (control, pipeline) chart so the color follows the arm.
    """
    if not pairs:
        return _no_data()
    pad = 72
    mx = maxval or max((v for _, v in pairs), default=0) or 1
    bar_w = width - label_w - pad
    h = row * len(pairs) + 6
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, (lab, val) in enumerate(pairs):
        y = i * row + 4
        w = max(2, bar_w * val / mx)
        fill = (color[i % len(color)] if isinstance(color, (list, tuple)) else color) or PAL[0]
        shown = fmt.format(val) + unit
        out.append(
            f"<text x='{label_w - 10}' y='{y + 14}' class='lab' text-anchor='end'>"
            f"{esc(str(lab)[:46])}</text>"
            + _bar(label_w, y + 3, w, 15, fill, f"{lab}: {shown}")
            + f"<text x='{label_w + w + 7:.1f}' y='{y + 15}' class='val strong'>"
              f"{esc(shown)}</text>")
    out.append("</svg>")
    return "".join(out)


def grouped_hbar(rows, *, series, width=W, group_gap=13, bar_h=13, percent=False,
                 rule=None, rule_label="", label_w=250, fmt="{:g}", direct_labels=True,
                 glossary=None):
    """One group of bars per category, one bar per series — the control-vs-pipeline
    workhorse.

    rows: [{"label": str, <series name>: value, ...}]
    series: [(name, color)] in draw order.
    ``direct_labels`` names the series at the end of the first group's bars, so the
    color mapping is learned inside the figure instead of below it.
    ``glossary`` is {label: definition} folded into the tooltip, which is how a chart
    of named jargon avoids needing a data-dictionary table under it.
    """
    rows = [r for r in rows if any(r.get(s) is not None for s, _ in series)]
    if not rows:
        return _no_data()
    pad = 96
    bar_w = width - label_w - pad
    mx = 1.0 if percent else (max((r.get(s) or 0) for r in rows for s, _ in series) or 1)
    grp_h = bar_h * len(series) + group_gap
    h = grp_h * len(rows) + (22 if rule is not None else 8)
    out = [f"<svg viewBox='0 0 {width} {h}' role='img' class='chart'>"]
    for i, r in enumerate(rows):
        top = i * grp_h + 6
        out.append(f"<text x='{label_w - 10}' y='{top + grp_h / 2 - 2:.0f}' class='lab' "
                   f"text-anchor='end'>{esc(str(r['label'])[:44])}</text>")
        for j, (name, color) in enumerate(series):
            val = r.get(name)
            if val is None:
                continue
            y = top + j * bar_h
            w = max(1.5, bar_w * val / mx)
            shown = f"{val:.0%}" if percent else fmt.format(val)
            tip = f"{r['label']} — {name}: {shown}"
            if glossary and glossary.get(r["label"]):
                tip += f" · {glossary[r['label']]}"
            out.append(_bar(label_w, y, w, bar_h - 3, color, tip)
                       + f"<text x='{label_w + w + 6:.1f}' y='{y + bar_h - 4}' class='val'>"
                         f"{esc(shown)}</text>")
            if direct_labels and i == 0:
                out.append(f"<text x='{label_w + w + 34:.1f}' y='{y + bar_h - 4}' "
                           f"class='val key-in'>{esc(name)}</text>")
    if rule is not None:
        x = label_w + bar_w * rule / mx
        out.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='2' y2='{h - 20}' class='rule'/>"
                   f"<text x='{x + 5:.1f}' y='{h - 7}' class='muted-svg'>{esc(rule_label)}</text>")
    out.append("</svg>")
    # A legend exists to tell two colours apart; with one series it is a dot that
    # labels nothing, so it only renders when there is a mapping to learn.
    return "".join(out) + (_legend(series) if len(series) > 1 else "")


def _legend(series):
    keys = "".join(f"<span class='key'><i style='background:{c}'></i>{esc(n)}</span>"
                   for n, c in series)
    return f"<div class='legend'>{keys}</div>"


def segbar(segments, *, width=W, height=30):
    """One bar split into proportional segments — the whole-corpus view of
    kept/weakened/dropped/added, which as 39 unlabelled columns was unreadable.

    segments: [(name, value, color)]

    Names and counts live in the legend below the bar, not inside it. Segment labels
    drawn on the fill were surface-coloured text at 2.5:1 on the green and 2.8:1 on the
    terracotta — a fail on cream and already a fail on white.
    """
    segments = [(n, v, c) for n, v, c in segments if v]
    total = sum(v for _, v, _ in segments)
    if not total:
        return _no_data()
    out = [f"<svg viewBox='0 0 {width} {height + 4}' role='img' class='chart'>"]
    x = 0.0
    for name, val, color in segments:
        w = width * val / total
        out.append(f"<rect x='{x:.1f}' y='0' width='{max(w - 2, 1):.1f}' height='{height}' "
                   f"fill='{color}' data-tip='{esc(name)}: {val} ({val / total:.0%})'/>")
        x += w
    out.append("</svg>")
    return "".join(out) + _legend([(f"{n} · {v}", c) for n, v, c in segments])


def stacked_bar(rows, *, categories, width=W, height=270, xlabel="", ylabel=""):
    """One stacked column per record. rows: [{"label", "segments", "tips"}]."""
    rows = [r for r in rows if r.get("segments")]
    if not rows:
        return _no_data()
    left, bottom, top = 44, 34, 10
    totals = [sum((r["segments"].get(c) or 0) for c, _ in categories) for r in rows]
    mx = max(totals) or 1
    plot_h = height - bottom - top
    bw = (width - left - 12) / len(rows)
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (0, 0.5, 1.0):
        y = top + plot_h * (1 - gy)
        out.append(f"<line x1='{left}' x2='{width - 12}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>"
                   f"<text x='{left - 7}' y='{y + 4:.1f}' class='val' text-anchor='end'>"
                   f"{int(mx * gy)}</text>")
    for i, r in enumerate(rows):
        x = left + i * bw + bw * 0.14
        w = bw * 0.72
        y_cursor = top + plot_h
        for cat, color in categories:
            val = r["segments"].get(cat) or 0
            if not val:
                continue
            seg_h = plot_h * val / mx
            y_cursor -= seg_h
            tip = (r.get("tips") or {}).get(cat) or f"{r['label']} — {cat}: {val}"
            out.append(f"<rect x='{x:.1f}' y='{y_cursor + 1:.1f}' width='{w:.1f}' "
                       f"height='{max(seg_h - 1, 0.5):.1f}' fill='{color}' "
                       f"data-tip='{esc(tip)}'/>")
        if len(rows) <= 24:
            out.append(f"<text x='{x + w / 2:.1f}' y='{height - 20}' class='val' "
                       f"text-anchor='middle' transform='rotate(-40 {x + w / 2:.1f} "
                       f"{height - 20})'>{esc(str(r['label'])[-6:])}</text>")
    if xlabel:
        out.append(f"<text x='{width / 2:.0f}' y='{height - 2}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x='2' y='{top - 1}' class='muted-svg'>{esc(ylabel)}</text>")
    out.append("</svg>")
    return "".join(out) + _legend(categories)


def histogram(counts, *, width=W, height=170, color=None, xlabel=""):
    """Distribution of a score or length. counts: [(bucket_label, n)]."""
    counts = list(counts)
    if not counts:
        return _no_data()
    left, bottom, top = 40, 30, 8
    mx = max(n for _, n in counts) or 1
    plot_h = height - bottom - top
    bw = (width - left - 10) / len(counts)
    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for gy in (0, 0.5, 1.0):
        y = top + plot_h * (1 - gy)
        out.append(f"<line x1='{left}' x2='{width - 10}' y1='{y:.1f}' y2='{y:.1f}' class='grid'/>"
                   f"<text x='{left - 7}' y='{y + 4:.1f}' class='val' text-anchor='end'>"
                   f"{int(mx * gy)}</text>")
    for i, (lab, n) in enumerate(counts):
        bh = plot_h * n / mx
        x = left + i * bw + bw * 0.12
        out.append(f"<rect x='{x:.1f}' y='{top + plot_h - bh:.1f}' width='{bw * 0.76:.1f}' "
                   f"height='{bh:.1f}' fill='{color or PAL[0]}' data-tip='{esc(lab)}: {n}'/>"
                   f"<text x='{x + bw * 0.38:.1f}' y='{height - 16}' class='val' "
                   f"text-anchor='middle'>{esc(lab)}</text>")
    if xlabel:
        out.append(f"<text x='{width / 2:.0f}' y='{height - 2}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    out.append("</svg>")
    return "".join(out)


def scatter(points, *, xdomain=None, ydomain=None, marks=(), width=W, height=330,
            xlabel="", ylabel=""):
    """points/marks: [{"x","y","color","tip"}]. marks draw larger and ringed (the
    per-arm means the dots scatter around). ``xlabel``/``ylabel`` draw on the chart
    itself, so the axes need no describing sentence above it."""
    pts = [p for p in points if p.get("x") is not None and p.get("y") is not None]
    if not pts:
        return _no_data()
    left, right, top, bottom = 44 + (18 if ylabel else 0), 14, 12, 26 + (16 if xlabel else 0)
    xs = [p["x"] for p in pts] + [m["x"] for m in marks]
    ys = [p["y"] for p in pts] + [m["y"] for m in marks]
    x0, x1 = xdomain or (min(xs), max(xs))
    y0, y1 = ydomain or (0, max(ys) * 1.12 or 1)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    pw, ph = width - left - right, height - top - bottom

    def px(x):
        return left + pw * (x - x0) / (x1 - x0)

    def py(y):
        return top + ph * (1 - (y - y0) / (y1 - y0))

    out = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for k in range(5):
        gy = y0 + (y1 - y0) * k / 4
        out.append(f"<line x1='{left}' x2='{width - right}' y1='{py(gy):.1f}' "
                   f"y2='{py(gy):.1f}' class='grid'/>"
                   f"<text x='{left - 7}' y='{py(gy) + 4:.1f}' class='val' "
                   f"text-anchor='end'>{gy:.0f}</text>")
    out.append(f"<line x1='{left}' x2='{width - right}' y1='{py(y0):.1f}' y2='{py(y0):.1f}' "
               f"class='axis'/>")
    for k in range(6):
        gx = x0 + (x1 - x0) * k / 5
        out.append(f"<text x='{px(gx):.1f}' y='{height - bottom + 16}' class='val' "
                   f"text-anchor='middle'>{gx:.0f}</text>")
    if xlabel:
        out.append(f"<text x='{left + pw / 2:.0f}' y='{height - 4}' class='muted-svg' "
                   f"text-anchor='middle'>{esc(xlabel)}</text>")
    if ylabel:
        out.append(f"<text x='14' y='{top + ph / 2:.0f}' class='muted-svg' "
                   f"text-anchor='middle' transform='rotate(-90 14 {top + ph / 2:.0f})'>"
                   f"{esc(ylabel)}</text>")
    for p in pts:
        out.append(f"<circle cx='{px(p['x']):.1f}' cy='{py(p['y']):.1f}' r='4.5' "
                   f"fill='{p.get('color', PAL[0])}' stroke='var(--surface-0)' "
                   f"stroke-width='1.2' opacity='.82' data-tip='{esc(p.get('tip', ''))}'/>")
    for m in marks:
        # The mean markers ring in ink, not surface: on a dense cloud a
        # background-coloured ring reads as a gap, not as an outline.
        out.append(f"<rect x='{px(m['x']) - 7:.1f}' y='{py(m['y']) - 7:.1f}' width='14' "
                   f"height='14' transform='rotate(45 {px(m['x']):.1f} {py(m['y']):.1f})' "
                   f"fill='{m.get('color', PAL[0])}' stroke='var(--text-primary)' "
                   f"stroke-width='1.5' data-tip='{esc(m.get('tip', ''))}'/>")
    out.append("</svg>")
    return "".join(out)


FLOW_W = 440
# The spine. Far enough right that a branch label fits to its left without being clipped by
# the viewBox, and left enough that the longest stage name still clears the right edge:
# "3 · the constitution rewrite" at 14px runs to ~342 of 440.
_FLOW_X = 110
_FLOW_STEP = 62       # one stage to the next


def flow(stages, *, source=("a weighted matrix", "dealt in code"),
         output=("one training record", ("user", "assistant")),
         branch=None, title=""):
    """A pipeline as a schematic: a source, a stage per dot down one spine, an output.

    ``stages``: [(name, gloss)], drawn top to bottom. ``branch``: (label, index) — a dashed
    spur into the stage at that index, for a step fed by something that is not its
    predecessor. ``title``: the accessible name; SVG text is not read as prose, so the caller
    must also say this in words above the diagram.

    A SCHEMATIC, NOT A CHART, and the difference is enforced here rather than left to the
    caller: nothing in it is proportional to a measurement, so it takes no series colour and
    no status colour — hairlines, one ink and one muted grey. A schematic drawn in the chart
    palette reads as a result, and this one is a map of the report's own beats.

    VERTICAL, which is what lets it live in the reading column and scale instead of scroll.
    Laid out left to right the same five steps need 720px: too wide for the prose measure, so
    it had to bleed into the figure track — which is for measurements — and on a 358px phone
    it needed a horizontal scroll box. Turned down the page it needs 440, fits the 38rem
    measure at every breakpoint, and at 390px scales to ~0.81, where a 12px label is still
    ~9.7px. Reading top to bottom also matches the report it maps, which is a sequence.
    """
    if not stages:
        return _no_data()
    x, top = _FLOW_X, 58                      # the first dot, below the source box
    dots = [top + i * _FLOW_STEP for i in range(len(stages))]
    height = dots[-1] + 112
    out = [f"<svg viewBox='0 0 {FLOW_W} {height}' role='img' class='flow'"
           f" aria-label='{esc(title)}'><title>{esc(title)}</title>"]
    # The source: a grid of cells, because the matrix IS a grid and anything else here would
    # be a picture of a box. Centred on the spine, like the record box at the other end.
    for r in range(2):
        for c in range(4):
            out.append(f"<rect x='{x - 30 + c * 15}' y='{2 + r * 15}' width='15' height='15' "
                       f"fill='none' class='flow-cell'/>")
    out.append(_flow_label(x + 40, 17, source[0], strong=True, anchor="start")
               + _flow_label(x + 40, 33, source[1], anchor="start"))
    # No label on this arrow: "dealt in code" is directly above it and the prose beside the
    # diagram says the same thing a third time. It only had somewhere to go by crowding the
    # first dot.
    out.append(_flow_arrow(x, 36, dots[0] - 7))
    out.append(f"<line x1='{x}' x2='{x}' y1='{dots[0]}' y2='{dots[-1]}' class='flow-rule'/>")
    for i, (name, gloss) in enumerate(stages):
        y = dots[i]
        out.append(f"<circle cx='{x}' cy='{y}' r='4.5' class='flow-dot'/>"
                   + _flow_label(x + 22, y + 1, name, strong=True, anchor="start")
                   + _flow_label(x + 22, y + 17, gloss, anchor="start"))
        if branch and branch[1] == i:
            # The spur's label sits UNDER its corner, centred, inside the viewBox. Hung off
            # the end of the spur and right-aligned it ran past x=0 and the phrase was cut
            # in half — measured at 390px and at 1440px, clipped in both.
            # With a head, because the arm FEEDS this stage: stage 2 is shown the control's
            # answer, not the other way round, and a dashed line with two bare ends does not
            # say which.
            out.append(f"<path d='M{x - 62},{y + 26} V{y} H{x - 13}' class='flow-arm'/>"
                       f"<path d='M{x - 13},{y - 4} L{x - 7},{y} L{x - 13},{y + 4}' "
                       f"class='flow-head'/>"
                       + _flow_label(x - 62, y + 42, branch[0]))
    out.append(_flow_arrow(x, dots[-1] + 22, dots[-1] + 54))
    name, rows = output[0], output[1]
    for i, row in enumerate(rows):
        y = dots[-1] + 60 + i * 20
        out.append(f"<rect x='{x - 40}' y='{y}' width='80' height='20' fill='none' "
                   f"class='flow-cell'/>" + _flow_label(x, y + 14, row))
    # Level with the middle of the box it names, not with one of its two rows.
    out.append(_flow_label(x + 52, dots[-1] + 60 + 10 * len(rows) + 4, name, strong=True,
                           anchor="start"))
    out.append("</svg>")
    return "".join(out)


def _flow_label(x, y, text, *, strong=False, anchor="middle"):
    cls = "flow-t strong" if strong else "flow-t"
    return (f"<text x='{x:.1f}' y='{y:.1f}' class='{cls}' text-anchor='{anchor}'>"
            f"{esc(text)}</text>")


def _flow_arrow(x, y0, y1, head=6):
    """A vertical line with a DRAWN head. Typed as a glyph it is a hairline that differs
    per font, which is the same reason the outbound arrow is stroked."""
    return (f"<line x1='{x}' x2='{x}' y1='{y0:.1f}' y2='{y1 - head:.1f}' class='flow-rule'/>"
            f"<path d='M{x - 4},{y1 - head:.1f} L{x},{y1:.1f} L{x + 4},{y1 - head:.1f}' "
            f"class='flow-head'/>")


def highlight(text, spans):
    """Escaped text with each verbatim span wrapped in <mark>.

    Fail-open, matching the viewer: spans were substring-validated at audit time, so
    a span that no longer locates renders unhighlighted rather than corrupting text.
    """
    out = esc(text)
    for span in spans or []:
        if not span:
            continue
        marked = esc(span)
        if marked in out:
            out = out.replace(marked, f"<mark>{marked}</mark>", 1)
    return f"<div class='resp'>{out}</div>"


def sidebyside(left_title, left_html, right_title, right_html, left_tone="", right_tone=""):
    return (f"<div class='pair'>"
            f"<div class='pane {left_tone}'><h4 class='pane-h'>{esc(left_title)}</h4>{left_html}</div>"
            f"<div class='pane {right_tone}'><h4 class='pane-h'>{esc(right_title)}</h4>"
            f"{right_html}</div></div>")


def quote(text):
    return f"<blockquote>{esc(text)}</blockquote>"


def sub(anchor, text):
    """A subheading that is also a deep-link target.

    The page is one document with two report sections in it, so every beat inside a
    section needs its own id: a reader arriving from the dataset card lands on #dad,
    and anyone quoting a finding wants #dad-weak.
    """
    return f"<h3 id='{esc(anchor)}'>{esc(text)}</h3>"


def substep(anchor, text):
    """A stage heading inside a beat, and a deep-link target like the beat itself.

    ``outline()`` picks these up as the rail's sub-items, so an anchored ``<h4>`` is how a
    stage becomes reachable. An id is therefore a decision, not decoration: the appendix's
    ``<h4>``s deliberately have none, because they sit inside closed drawers and a link to
    a collapsed heading goes nowhere. The ids carry their beat's name — the same three
    stages are named in "How it is built" and again in the worked example.
    """
    return f"<h4 id='{esc(anchor)}'>{esc(text)}</h4>"


def illustration(data_uri="", alt="", label="Illustration"):
    """The hero's illustration.

    ``data_uri`` must be a ``data:`` URI — the whole page is one file, so a reference
    to anything outside it (even a relative path) breaks the artefact the moment it
    travels. Without one the slot renders empty at the right proportions, so the hero
    keeps the shape the finished page will have.
    """
    if not data_uri:
        return ("\n<!-- TODO: hero illustration. Drop a PNG into website/assets/hero.png; "
                "build_website.py inlines it as a data URI. No external asset may be "
                "referenced: this page has to open offline from the filesystem. -->\n"
                f"<div class='illo'><span>{esc(label)}</span></div>\n")
    if not data_uri.startswith("data:"):
        raise ValueError("the hero illustration must be a data: URI — the page is one file")
    return (f"\n<div class='illo art'><img src='{esc(data_uri)}' alt='{esc(alt)}'></div>\n")




def named_pair(items):
    """A short list set as columns: a name and a sentence each.

    ``items``: ``[(name, body_html)]``, drawn left to right.

    An ``<ol>``, because these are ordered items and a reader on a screen reader should
    hear "list, 2 items" — but it carries no visible index. A small uppercase label over
    each name ("Technique 1") was tried and cut: an eyebrow over a heading names what the
    heading already says, and the ordinal is not information a reader needs.

    NOT A CARD, and the CSS is where that is held: a hairline over each column, a serif
    name, and no fill, border box or radius anywhere. This is the page's only two-up block
    outside a table, and a card here would be the first one on it.
    """
    if not items:
        return ""
    out = "".join(f"<li><span class='npair-h'>{esc(name)}</span>"
                  f"<div class='npair-b'>{body}</div></li>" for name, body in items)
    return f"<ol class='npair'>{out}</ol>"


def hero(title, art="", intro=""):
    """The opening: the illustration, the title, and the lines that follow from it.

    The intro is part of the hero rather than a section of its own — it is the second
    half of the title's sentence, and a heading over it ("Intro") only told a reader
    what they could already see. It carries the ``#intro`` id so the skip link has
    somewhere to land.
    """
    return (f"<header class='hero'>{art}<h1>{esc(title)}</h1>"
            + (f"<div class='hero-intro' id='intro'>{intro}</div>" if intro else "")
            + "</header>\n")


# Monochrome marks, drawn inline and inheriting currentColor: the page is one file, and a
# colour logo would fight a palette built out of ink on paper. The GitHub silhouette is
# the published mark; the Hugging Face one is a simplified face, which is what survives
# being drawn at 15px in a single colour.
# Marks drawn inline: the page is one file, so a logo is path data or it is nothing.
# GitHub's is the published silhouette and inherits currentColor. Hugging Face's is
# their actual logo, fetched from huggingface.co/front/assets, keeping its own fills —
# it IS a smiley face, but theirs, hands and all, rather than a circle I drew.
# (name -> viewBox, width, height, paths)
ICONS = {
    "github": ("0 0 16 16", 15, 15,
               "<path d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
               "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-"
               ".15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-"
               ".87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02."
               "08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-"
               ".82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
               "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 "
               "0 0 16 8c0-4.42-3.58-8-8-8z'/>"),
    "hf": ("0 0 95 88", 16, 15, '<path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z"/><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"/><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"/><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"/><path fill="#3A3B45" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z"/><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"/><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"/><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"/><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"/>'),
}


def icon(name):
    if name not in ICONS:
        return ""
    viewbox, w, h, paths = ICONS[name]
    return (f"<svg class='ico' viewBox='{viewbox}' width='{w}' height='{h}' "
            f"aria-hidden='true' fill='currentColor'>{paths}</svg>")


def linkbutton(href, label, name="", meta=""):
    """An outbound link as a button: mark, label, and the arrow that says it leaves.

    Every link off this page is one of two destinations, so both get the same treatment
    and a reader can tell at a glance which is which.
    """
    return (f"<a class='lbtn' href='{esc(href)}'{NEW_TAB}>{icon(name)}"
            f"<span>{esc(label)}</span>"
            + (f"<span class='lbtn-m'>{esc(meta)}</span>" if meta else "")
            + EXT_ARROW + "</a>")


def compare(columns, rows):
    """The two datasets, side by side, with their names as the masthead.

    columns: [(name,)]. rows: [(label, cell, cell)].

    A comparison is a table — the whole point is that "records" lines up with "records" —
    but the names carry the section instead of a heading above it, so the header cells do
    the work a masthead would.

    A MASTHEAD IS A NAME AND NOTHING ELSE. What each dataset is used to sit here as a
    subtitle, which left the one claim a reader most needs as the only unlabelled thing in
    a table whose every other line says what it is answering; it is a row now, like the
    rest. A neutral chip under the name also lived here, saying one of the two reports was
    not written yet; it went when that report was written, because a state that is no
    longer true must not survive as an affordance nobody passes.
    """
    heads = "".join(f"<th><span class='cmp-name'>{esc(name)}</span></th>"
                    for (name, *_) in columns)
    body = "".join("<tr><th class='cmp-k' scope='row'>" + esc(label) + "</th>"
                   + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>"
                   for label, *cells in rows)
    return (f"<div class='cmp-wrap'><table class='cmp'>"
            f"<thead><tr><td class='cmp-corner'></td>{heads}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def iconlink(href, label, name=""):
    """A link with its provider's mark and the outbound arrow. Not a button: in the
    footer there is nothing to press, only two places to go."""
    return (f"<a class='ilink' href='{esc(href)}'{NEW_TAB}>{icon(name)}"
            f"<span>{esc(label)}</span>{EXT_ARROW}</a>")


_HEADING = re.compile(r"<h([34]) id='([^']+)'>([^<]*)</h\1>")


def outline(html):
    """A built panel's own headings as a two-level tree: [(id, text, [(id, text), …])].

    Read back off the panel rather than taken from a module's BEATS list, because the
    beats are conditional — the document report only earns ``sdf-weak`` when its run's
    audit flagged something — so deriving them is what stops the rail advertising a beat
    that is not there. An ``<h4>`` becomes a sub-item of the beat it follows; the ones
    with no id (the appendix's, inside closed drawers) are invisible here, which is how
    a rail link to a collapsed heading is prevented.
    """
    beats = []
    for level, hid, text in _HEADING.findall(html or ""):
        if level == "3":
            beats.append((hid, text, []))
        elif beats:
            beats[-1][2].append((hid, text))
    return beats


def rail(pid, beats):
    """One report's contents, as the sticky column beside it.

    A report is 4,000 words of records and from inside one a reader can see neither its
    shape nor a way to the appendix. The rail is that shape, held on screen, with each
    beat's stages under it. Hidden until its report is opened, like the panel it belongs
    to, and toggled by the same handler.

    The page itself still has no rail: this is one report's own contents, it appears only
    once a report is open, and it goes away with it.
    """
    if not beats:
        return ""
    items = []
    for bid, text, subs in beats:
        items.append(f"<a class='r-b' href='#{esc(bid)}'>{esc(text)}</a>")
        items += [f"<a class='r-s' href='#{esc(sid)}'>{esc(stext)}</a>"
                  for sid, stext in subs]
    return (f"<nav class='rail' data-rail='{esc(pid)}' "
            f"aria-label='Sections of this report' hidden>{''.join(items)}</nav>")


def chooser(options, prompt=""):
    """The two datasets as a choice, then the chosen one below.

    options: [(panel_id, label)]. Just the names: the description and the figures are
    both on the page already, a few inches up, and a reader choosing between two names
    needs neither repeated. Nothing is open on load — the choice is the point — and a
    ``#panel_id`` in the URL opens that panel, so a deep link from the dataset card
    still lands where it says it will.

    The buttons come wrapped in ``.choicebar``, which is what sticks to the top of the
    screen: the bar needs a full-column box to carry the page's own background, and
    ``.choices`` is 40rem centred, so a report would scroll up either side of it. The
    bar stays ONE ROW — a row of section links hung underneath it read as clutter on the
    control; that job belongs to the rail beside the report.

    THIS IS A DISCLOSURE PAIR, NOT A TAB SET, and it is marked up as one. It was
    ``role='tablist'`` with two ``role='tab'``s, which promises things this control does
    not do and cannot: a tablist always has exactly one selected tab, and nothing is
    selected here on load — that is the whole point of the chooser — so a screen reader
    announced "tab, 1 of 2, not selected" twice and arrow keys did nothing. Two buttons
    carrying ``aria-expanded`` describe what actually happens: each one opens a region,
    both can be closed, and Tab plus Enter is the entire interaction.
    """
    buttons = []
    for pid, label in options:
        buttons.append(
            f"<button class='choice' type='button' aria-expanded='false' "
            f"aria-controls='{esc(pid)}' data-panel='{esc(pid)}' id='choose-{esc(pid)}'>"
            f"{esc(label)}<span class='choice-a' aria-hidden='true'>&darr;</span></button>")
    return ((f"<p class='choose-q'>{inline_md(prompt)}</p>" if prompt else "")
            + f"<div class='choicebar'><div class='choices'>"
              f"{''.join(buttons)}</div></div>")


def tabs(panes):
    """Several records behind one set of buttons. panes: [(id, label, body, open_)].

    The pane marked open renders WITHOUT ``hidden``, so with JS off this degrades to one
    visible record rather than to none, and the print rule expands the rest.

    Unlike the chooser, this one IS a tab set — exactly one pane is open at all times —
    so it keeps ``role='tab'`` and owes the rest of that pattern: one tab in the tab order
    at a time (``tabindex`` roves with the selection) and Left/Right/Home/End across the
    set, both in the page's own inline JS.
    """
    kept = [p for p in panes if p]
    if not kept:
        return ""
    btns = "".join(
        f"<button class='tab' type='button' role='tab' id='tab-{esc(pid)}' "
        f"data-pane='{esc(pid)}' tabindex='{'0' if open_ else '-1'}' "
        f"aria-selected='{'true' if open_ else 'false'}' "
        f"aria-controls='{esc(pid)}'>{esc(label)}</button>"
        for pid, label, _, open_ in kept)
    # Each pane is named by the button that opens it: a tabpanel with no accessible name
    # is announced as a bare group, which on this page means an unlabelled 1,200-word
    # transcript. The id on the button is what makes that association possible.
    bodies = "".join(
        f"<div class='pane-x' id='{esc(pid)}' role='tabpanel' "
        f"aria-labelledby='tab-{esc(pid)}'"
        f"{'' if open_ else ' hidden'}>{body}</div>"
        for pid, _, body, open_ in kept)
    return (f"<div class='carousel'><div class='tabs' role='tablist'>{btns}</div>"
            f"{bodies}</div>")


def panel(pid, body):
    """One chooser panel: closed until its button is pressed.

    It ends where its content ends. There was a button here offering the other dataset,
    from when the chooser scrolled away behind the reader; the bar it lives in is pinned
    now, so the other report is one click away from anywhere in this one.
    """
    return (f"<section id='{esc(pid)}' class='panel' "
            f"aria-labelledby='choose-{esc(pid)}' hidden>{body}</section>")


def explore_body(bar, rails, panels):
    """The bar, the rails and both reports as one block: that block is their travel.

    ``position:sticky`` moves only inside its containing block, and the containing block
    of a grid item is its own grid area — one row, as tall as the buttons — so a sticky
    bar left in ``#explore``'s grid would have nowhere to go. This div holds all three, so
    the bar pins for exactly as long as a report is being read and is released at the end
    of it, before the footer.

    Two columns and two rows: the bar spans the top, then the rails sit in the narrow
    column beside the reports in the wide one. The rails and the panels are wrapped as
    ONE GRID ROW EACH SIDE on purpose — a grid item stretches to its row's height, so
    ``.railcol`` is as tall as the open report and the rail inside it can pin for the
    whole length of it. Left as loose siblings, each panel would start a row of its own
    and the rail would have a row's worth of travel.
    """
    return (f"<div class='explore-body'>{bar}"
            f"<div class='railcol'>{rails}</div>"
            f"<div class='panels'>{panels}</div></div>")


CSS = """
/* Aged paper, one theme. Every text-on-surface pair below clears WCAG AA (4.5:1) and
   tests/test_website_common.py::test_text_contrast_meets_wcag_aa recomputes them from
   these tokens, so darkening a surface without darkening its ink fails the suite. On
   cream the rules and the chip washes both need to be markedly stronger than they were
   on white, where a 1.1:1 wash still read as a chip. */
:root{color-scheme:only light;
--surface-0:#f7f4ea;--surface-1:#f1ebdd;--surface-2:#e9e1cd;
--border:#cec3a6;--hairline:#ded5be;--grid:#e1d9c4;--axis:#bcaf90;
--text-primary:#1a1712;--text-secondary:#4a443c;--text-muted:#675f54;
/* --accent-edge is a CONTROL BOUNDARY, so it answers to WCAG 1.4.11's 3:1 rather than to
   the 4.5:1 the text pairs above take. The old #c9c3ea reached 1.53:1 on the paper — the
   two chooser buttons, which are the page's only decision, had a border a low-vision
   reader could not see, and the accent text inside made them read as links. This measures
   3.48:1 on --surface-0 and 3.15:1 on --accent-wash, so it holds in the hover state too. */
--accent:#3b2fa0;--accent-wash:#eae7f7;--accent-edge:#8279c5;
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;
--series-5:#e87ba4;--series-6:#008300;--series-7:#4a3aa7;--series-8:#e34948;
--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;
--good-ink:#0a6b12;--warn-ink:#7a4d00;--bad-ink:#a52222;
--good-wash:#dcecd0;--warn-wash:#f4e4c2;--bad-wash:#f4dbd5;
--good-edge:#b6d3a4;--warn-edge:#dcc48c;--bad-edge:#e0b3aa;
--mark:#f2e39c}
*{box-sizing:border-box}
html{--serif:ui-serif,Charter,"Bitstream Charter","Iowan Old Style","Source Serif 4","Charis SIL",Georgia,serif;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:1.0625rem/1.62 var(--serif);-webkit-text-size-adjust:100%}
/* Heard, never seen. The comparison's own heading is its two column mastheads, which is
   right on screen and invisible to a reader navigating by heading — pressing H jumped from
   the page title straight past the entire comparison. This gives that section a heading in
   the outline without putting one over the top of it. */
.vh{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
clip-path:inset(50%);white-space:nowrap;border:0}
.skip{position:absolute;left:-9999px}
.skip:focus{left:12px;top:12px;z-index:20;background:var(--surface-0);padding:8px 12px;
border:1px solid var(--border);font-family:var(--sans);font-size:.85rem}

/* Shell: one centred column, with a figure track that bleeds past the prose measure.

   67rem, not the 53rem this page was built at, because a report now carries a contents
   rail beside it: 12rem of rail, and the reading column and the figure track keep the
   widths they were tuned at or better (38rem of prose, an 812px figure track — charts are
   drawn at 800px, so a narrower track would shrink every chart's 11px labels; at 792px
   they were being scaled down by 8px). The 3rem gutter beside the rail comes out of this
   shell's own left margin rather than out of its width — see --pull below. The
   extra width only becomes a column inside #explore; everywhere else it lands in the
   figure track, and the comparison and the hero are centred on the viewport regardless. */
html{scroll-behavior:smooth}
.shell{max-width:67rem;margin:0 auto;padding:0 28px 110px}
main{min-width:0}
/* minmax(0,1fr), never a bare 1fr: a bare fr track takes its automatic minimum from
   the item's min-content size, so a child with a definite width wider than the column —
   the comparison's 64rem wrapper — GROWS the track past the page, and every percentage
   resolved against that grid area (left:50%, margin-left:50%) then points somewhere to
   the right of the page centre. Measured: the wrapper's centre landed 116px right. */
section{display:grid;
grid-template-columns:[text-start] minmax(0,38rem) [text-end] minmax(0,1fr) [full-end];
scroll-margin-top:2.5rem}
section>*{grid-column:text-start/text-end}
section>figure,section>.tiles,section>.scroll,section>.pair,section>details,
section>.explore-body,section>.lbtns,section>.cmp-wrap,
section>.carousel{grid-column:text-start/full-end}
section+section{margin-top:5rem}
/* A centred rule sat above the COMPARISON and is gone: the intro now draws two of its own and
   a third one a screen below them made the top of the page read as ruled sections. The gap it
   carried went with it — its margin WAS the entire space between the intro and the table,
   because the hero's bottom padding was zero — so the hero pays for that gap now, in one value
   on one rule.

   The chooser had one too, briefly, and it is gone for the same reason: the two rules the
   intro draws are the page's whole ration. It keeps the gap that rule was carrying, as
   6rem of its own rather than the 5rem every other section break takes — the chooser is
   where the page stops describing and starts asking. */
#explore{margin-top:6rem}
/* The panel is a section, so its own display:grid would beat the browser's default
   [hidden] rule. It has to be said out loud. */
.panel[hidden]{display:none}

/* The hero: the image, the title and the lines that follow from it, centred, with
   enough air to separate them from the page and no more. The two datasets are two
   things, so they are two things here as well as in the table below. */
.hero{display:flex;flex-direction:column;align-items:center;
padding:96px 28px 5rem;text-align:center}
/* 3rem above the art and 64px above the title. This was 6rem each: the art is a 186px band
   inside a 36rem box, so 12rem of stacked margin spent ~190px of the first screen on paper
   with nothing on it and pushed the comparison — the section that does this page's work —
   to 1,160px, past the fold on a laptop. The hero's own top padding (96px) is what the page
   opens on; these two are the gaps inside it. */
.hero h1{max-width:22ch;margin:64px 0 0;font-size:3rem}
/* No margin here: the art's own spacing is set once, by `.illo.art` below. This rule used
   to say `margin:0` and was overridden by it — same specificity, later in the file — so
   the hero silently carried a third top margin nothing here accounted for. */
.hero .illo{width:100%}
/* The artwork is 1536x1024 but its ink occupies only a 1318x425 band centred at 48.5%
   of the height — a third of the file is transparent above it and a third below. Left
   uncropped it spends ~340px of the hero on nothing, and every gap measured against it
   is a gap the reader cannot see. Cropped here rather than in the asset, which stays
   exactly as it was supplied. */
.hero .illo.art img{max-width:36rem;margin:0 auto;aspect-ratio:1318/425;
object-fit:cover;object-position:50% 48.5%}
/* TWO MEASURES, NOT ONE. The paragraphs keep the 60ch a centred line can be read at; the
   pair below them needs its container wider than that, because two columns inside 60ch are
   ~24ch each and a 48-word item comes out fourteen lines deep. */
.hero-intro{max-width:min(100%,48rem);margin:48px auto 0}
.hero-intro>p{max-width:60ch;margin-left:auto;margin-right:auto}
.hero-intro p{margin-top:0;margin-bottom:0;color:var(--text-secondary);
font-size:1.1rem;line-height:1.68}
/* 1.8rem, not the 1.4 this ran at: at 1.4 against a 1.68 line-height the gap between two
   paragraphs was barely more than the gap between two lines inside one, so the intro read
   as a single centred block. Enough to separate them and no more — the structural gaps
   around the pair below stay bigger than this one. */
.hero-intro p+p{margin-top:1.8rem}
/* Every paragraph in here wraps the same way, on the global text-wrap:pretty. `balance` was
   tried on the two above the pair and is wrong: it evens the lines by SHRINKING the block's
   used width, so those two set visibly narrower than the two below them and the centred
   column stopped having one edge.

   The 2.6rem that used to sit on the paragraph after the pair is gone: a rule is drawn
   between them now and carries the whole gap. Left in, it beat the `margin-top:0` below it
   on specificity — three classes to one — and the pair's 16px turned into 42. */
/* Two rules inside the intro, drawn as pseudo-elements on the paragraph BELOW each one:
   after the opening claim, and under the pair of techniques. A short centred rule, not a
   full-width one — at the container's own 48rem it would read as a section break inside a
   block that is one continuous piece of prose. 48px either side of each, so a rule sits in
   96px of air with the same amount above it as below.

   display:flow-root, and it is load-bearing: the rule is the paragraph's FIRST CHILD, so
   its top margin collapses straight through the paragraph and out into whatever sits above.
   Collapsed margins take the larger of the two rather than the sum, which is why the pair's
   own 16px bottom margin below is invisible without this. The paragraph's own top margin is
   zeroed for the same reason — two numbers for one gap, neither of them the one anybody
   set. */
.hero-intro>p:nth-child(2),.hero-intro>ol+p{margin-top:0;display:flow-root}
.hero-intro>p:nth-child(2)::before,.hero-intro>ol+p::before{content:'';display:block;
width:min(100%,240px);margin:48px auto;border-top:1px solid var(--hairline)}
/* The two techniques, as two columns in the page's own order — synthetic documents left,
   difficult advice right, the same order the comparison, the chooser and both panels use —
   so the pair a reader meets in the hero is the pair the rest of the page keeps.

   The hero centres and this does not: a list cannot, and centring these two columns would
   leave four ragged edges. It goes flush left and stays centred as a BLOCK, which reads as
   deliberate now that it is visibly a figure rather than a paragraph with digits in front
   of it.

   Each is a bordered box: a hairline all the way round, 4px, 24px of padding. This ran for
   a while as a hairline over each column and nothing else, on the reasoning that a box here
   would be the first on a page that has none and that 4px belongs to things you press. The
   two techniques are the one place the page names a pair of objects rather than making an
   argument, and they are boxed deliberately. No fill and no shadow, so they stay flat: the
   border is the whole of it.

   The 16px below sits on top of the 48px belonging to the rule under it: the boxes have a
   visible bottom edge of their own now, and at 48px flat that edge and the rule read as a
   pair of lines. */
.npair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:48px;
list-style:none;margin:2.6rem 0 16px;padding:0;text-align:left}
.npair>li{margin:0;padding:24px;border:1px solid var(--hairline);border-radius:4px}
/* Muted and a step smaller, so the two technique names recede rather than reading as a first
   run at the comparison's mastheads below them. */
.npair-h{display:block;font:650 1.05rem/1.3 var(--serif);color:var(--text-muted);
margin:0 0 .7rem}
.npair-b{font-size:1rem;line-height:1.68;color:var(--text-secondary)}
/* Type: the serif argues, the sans measures. */
h1{font:700 2.6rem/1.07 var(--serif);letter-spacing:-.02em;margin:0 0 .5rem;
text-wrap:balance;font-variant-numeric:proportional-nums}
/* A real scale, because there was not one: h3 used to be 1.1rem against a 1.0625rem body
   and h4 was SMALLER than the prose under it, so a 4,000-word report read as one
   undifferentiated column and nothing on screen said "new beat". Each step is about a
   1.3 ratio now — 2 / 1.4 / 1.12 against a 1.0625rem body — and
   test_the_type_scale_steps_down keeps it monotonic and clear of the prose. */
h2{font:600 2rem/1.15 var(--serif);letter-spacing:-.014em;margin:0 0 .5rem;text-wrap:balance}
h3{font:600 1.4rem/1.25 var(--serif);letter-spacing:-.008em;margin:2.6rem 0 .4rem;
text-wrap:balance}
/* A beat's opening paragraph is not a dek: it wants the air a paragraph gets, not the
   .4rem a heading leaves for a line that belongs to it. */
h3+p{margin-top:.9rem}
/* Every beat and every stage inside a report is its own deep-link target — the rail links
   to all of them — so each needs the headroom a section gets, and inside a report the
   chooser is pinned to the top of the screen, so the headroom has to clear the bar as
   well. The bar is 5.21rem:

     .choicebar padding  .8 + .8                = 1.600rem
     .choice padding     1 + 1                  = 2.000rem
     .choice line box    1.14rem x 1.3          = 1.482rem
     .choice border      2 x 1px                = 0.125rem

   7rem is that plus air. Stated in CSS rather than measured in the script because a
   native fragment jump reads it too, and test_a_deep_linked_beat_lands_clear_of_the_bar
   recomputes the sum from these same declarations. */
/* A beat also gets a rule and the air above it: four of them in a report, and the rule is
   what makes the report read as four chunks rather than one scroll. */
h3[id]{scroll-margin-top:7rem;margin-top:3.4rem;padding-top:1.5rem;
border-top:1px solid var(--hairline)}
h4{font:650 1.12rem/1.35 var(--serif);margin:2rem 0 .35rem;color:var(--text-primary)}
h4[id]{scroll-margin-top:7rem}
/* The one place h4 is a label over a block rather than a subhead in a document: the two
   halves of a side-by-side, whose titles are "plain" and "pipeline". Small sans, as every
   h4 used to be. */
h4.pane-h{font:650 .82rem/1.35 var(--sans)}
p{margin:0 0 1.05em;color:var(--text-secondary);text-wrap:pretty}
ul{color:var(--text-secondary);padding-left:20px;margin:0 0 1.05em}li{margin:.3em 0}
.lede{font:1.22rem/1.5 var(--serif);color:var(--text-primary);margin:0 0 1.1rem;max-width:40rem}
.dek{font:.9rem/1.5 var(--sans);color:var(--text-muted);margin:0 0 1.4rem;max-width:44rem}
.meta{font:.8rem/1.55 var(--sans);color:var(--text-muted);margin:1.2rem 0 0;
padding-top:1rem;border-top:1px solid var(--border);max-width:46rem}
.muted{color:var(--text-muted);font:.84rem/1.5 var(--sans)}
.mono{font-family:var(--mono);font-size:.86em}

/* The choice, and the two ways out of the page. */
.choose-q{font:1.22rem/1.5 var(--serif);color:var(--text-primary);margin:0 0 1.4rem}
/* The choice lines up with the thing being chosen: 40rem centred on the page is exactly
   the two dataset columns above (2 x 20rem), so each button sits under its own column.
   Its heading centres over them for the same reason.

   The child combinator is load-bearing: both reports live INSIDE #explore now (that is
   what gives the sticky bar its travel), and each opens with its own <h2>, so a
   descendant selector here centres and stretches both report titles too. */
/* The one h2 on the page that is not a name. It is an instruction, and at the h2's own 2rem
   it set level with "Synthetic documents" and "Difficult advice" — the two documents it
   points at, both of which are h2s INSIDE this section — so the label was as loud as the
   thing. Since #datasets' heading is visually hidden, it was also the only visible h2 before
   a report opens: on arrival the page's second voice after the title was a caption for two
   buttons. Dropped to the beat size, which puts it under the names and above the prose.
   Still an <h2>, so the outline and the H key are unchanged. */
#explore>h2{grid-column:text-start/full-end;text-align:center;margin-bottom:1.2rem;
font-size:1.4rem}
/* The travel, for both the bar and the rail. min-width:0 for the reason main has it: it is
   a grid item, and a grid track takes its automatic minimum from the item's min-content
   size.

   Two columns, two rows: the bar across the top, then the rail beside the reports. The
   rail column is fixed and the reading side takes the rest, which is why the shell above is
   67rem — the report keeps its 38rem measure and its figure track.

   No rule between the two columns. The line was a second separator: a fixed column the
   rail's links never leave, set in the sans at .8rem with its stages indented under their
   beat, is already not the prose beside it.

   With nothing drawn there the gutter has to hold the two columns apart on its own, which
   takes 3rem, and that came out of the SHELL'S LEFT MARGIN rather than either column: the
   block is pulled left by --pull, exactly the 2.25rem the gutter grew by, so the contents
   hang into the margin and the reading column stays where it was — same 416px left edge,
   same 812px figure track. The pull is clamped to the room outside the shell, so on a
   viewport too narrow to have any (below ~1088px) it is 0 and the gutter narrows the
   reading column instead, which is what every other width between 900px and 67rem already
   does. Print gets 0 for the same reason.

   --t LIVES HERE, not on .choicebar, because two pinned things now read it: the bar
   interpolates its own six sizes off it and the rail's top follows the bar's height, so a
   tightening bar does not leave a growing gap above the rail. The script toggles .tight on
   this element, which is also the one it measures.

   overflow-anchor:none is load-bearing, and was measured: shrinking the pinned bar moves
   the report under it, so scroll anchoring "corrects" the scroll by the same amount, which
   moves this wrapper's top, which is what the shrink is computed FROM. With anchoring on,
   the bar settled at 52px while sitting 31px BELOW the top of the screen, or bounced
   between 52 and 83 depending on where the reader stopped. */
.explore-body{--t:0;--rail:12rem;--pull:min(2.25rem,max(0px,(100vw - 67rem)/2));
min-width:0;overflow-anchor:none;margin-left:calc(-1*var(--pull));
display:grid;grid-template-columns:[rail-col] var(--rail) [read-col] minmax(0,1fr);
column-gap:3rem}
.explore-body.tight{--t:1}
/* THE DATUM IS THE REPORT'S FIRST LINE OF PROSE, NOT ITS TITLE. With no padding at all the
   first beat sat ~48px ABOVE the <h2> it is the contents of; levelled with the <h2> instead
   it overshot the other way — a .8rem sans link sharing a band with a 2rem serif title reads
   as a competing second heading, and at that ratio of sizes box-to-box alignment puts the
   rail's text visibly above the title's cap. Landing on the lede gives the heading its own
   band and makes the rail an annotation beside the prose.

   Derived, never typed: .panel's 3.2rem margin + the <h2>'s 2.3rem line box (2rem/1.15) +
   the 1.9rem margin `.panel>h2` gives it — NOT the global h2's .5rem, which is overridden
   here and was what a first attempt at this landed 22px high on — puts the lede's box at
   7.4rem, and the two half-leadings are the optical term: the lede's .305rem down against
   this link's .42rem (.28rem padding plus its own .14rem). So the rail's box wants
   7.285rem, of which .2rem is the rail's own padding. 7.1rem here keeps the column on the
   page's quarter-rem grain; the remainder is a quarter of a pixel. The test recomputes all
   of it from those same rules. Measured at 1440px: both text tops at y=204.
   Only at rest: once the rail pins, its own `top` places it. */
.railcol{grid-column:rail-col;padding-top:7.1rem}
.panels{grid-column:read-col;min-width:0}
/* One report's contents, held on screen for as long as that report is being read.
   Its travel is .railcol, which stretches to the row's height — the height of the open
   panel — so the rail pins from the first beat to the last and is released with the
   report.

   The top follows the bar: 6rem clears it at rest, 3.9rem once it has tightened, off the
   same --t and with the same 200ms transition, so the two pinned things move together.
   A rail longer than the screen scrolls inside itself rather than being clipped. */
.rail{position:sticky;top:calc(6rem - 2.1rem*var(--t));padding:.2rem 0 1rem;
max-height:calc(100vh - 7.5rem);overflow-y:auto;scrollbar-width:thin;
transition:top .2s ease}
.rail[hidden]{display:none}
/* Sans, because the rail measures the document rather than arguing in it, and each link
   sets its own font shorthand so the mono/underlined a{} rule cannot drag it back. A beat
   is the document's own heading; a stage is indented under it and quieter. */
.rail a{display:block;color:var(--text-muted);text-decoration:none;
border-left:2px solid transparent;padding:.28rem 0 .28rem .7rem}
.rail a:hover{color:var(--text-primary);background:var(--accent-wash)}
.rail .r-b{font:650 .8rem/1.35 var(--sans);margin-top:.55rem}
.rail .r-s{font:.75rem/1.35 var(--sans);padding-left:1.5rem}
.rail>.r-b:first-child{margin-top:0}
/* Where the reader is. Ink and an edge, never a fill: an accent fill on this page means
   SELECTED — the open tab, the open pane — and the reader did not press this. */
.rail a[aria-current=true]{color:var(--text-primary);border-left-color:var(--accent)}
/* Pinned to the top of the screen for as long as a report is being read, carrying the
   page's own background so the report scrolls under it and out of sight. Full column
   width, not the buttons' 40rem, or a figure would scroll up either side of it.
   z-index:5 sits under #tip (9) and .skip:focus (20) and over everything else.

   It TIGHTENS ONCE, at a trigger point, because at rest it is right and pinned over a
   report it is heavy: --t is a flag, 0 loose and 1 tight, the script sets it when the
   reader is past the trigger, and every dimension below is one interpolation off it, so
   the two states are one set of numbers. ~72px tall and 40rem wide loose, ~52px and 30rem
   tight. A size that tracked the scroll continuously read as distracting — the bar moved
   whenever the page did — so this crosses once and settles. The six sizes are tokens, so
   each breakpoint restates only the tokens.

   THE HEIGHT RANGE IS NARROW ON PURPOSE, and it used to be 83px. The pinned size is the
   one measured to sit comfortably beside prose, so a resting size 61% taller than it was
   oversized on arrival by its own evidence — and the collapse read as a layout event
   rather than the bar settling. Only the height came down: the WIDTH stays 40rem because
   that is exactly the comparison's two 20rem columns, so each button sits under the column
   it opens. The interpolation coefficients were re-derived to hold the pinned size where it
   was, not left to shrink with the base.

   The transition lives on the concrete properties rather than on --t (a custom property
   is discrete unless it is registered), which is also what lets the reduced-motion rule
   at the foot of this stylesheet turn the animation off with the same transition:none it
   applies to everything else. */
.choicebar{--pad:.7rem;--gap:1.2rem;--btn-y:.8rem;--btn-x:1.25rem;--label:1.05rem;
--w:40rem;grid-column:1/-1;position:sticky;top:0;z-index:5;background:var(--surface-0);
padding:calc(var(--pad)*(1 - .4*var(--t))) 0;transition:padding .2s ease;
/* The wrapper's pull is for the contents, not the chooser. The bar spans both columns, so
   left to itself it widens leftwards with the block and takes its centred pair of buttons
   1.125rem off the page's centre line, out of step with the hero and the comparison. */
margin-left:var(--pull)}
/* The pair narrows with everything else, 40rem to 30rem, staying centred as it goes. The
   floor is measured, not chosen: below 27.5rem "Synthetic documents" wraps to two lines
   and the shrunk bar is taller than the one it replaced, so .25 is as far as this goes. */
.choices{display:grid;grid-template-columns:1fr 1fr;
gap:calc(var(--gap)*(1 - .35*var(--t)));
width:min(100%,calc(var(--w)*(1 - .25*var(--t))));margin:0 auto;
transition:width .2s ease,gap .2s ease}
/* Two names and an arrow, in the accent. The cream fill with a border was doing duty as
   a button, a card, a chip and a code block at once, and had stopped meaning anything;
   an outline in the accent that fills when you choose says "this is a control". */
.choice{display:flex;align-items:center;justify-content:space-between;gap:1rem;
padding:calc(var(--btn-y)*(1 - .35*var(--t))) calc(var(--btn-x)*(1 - .2*var(--t)));
background:none;border:1px solid var(--accent-edge);
border-radius:4px;cursor:pointer;font:650 var(--label)/1.3 var(--serif);
font-size:calc(var(--label)*(1 - .08*var(--t)));color:var(--accent);text-align:left;
transition:padding .2s ease,font-size .2s ease}
.choice:hover{background:var(--accent-wash)}
.choice[aria-expanded=true]{background:var(--accent);border-color:var(--accent);
color:var(--surface-0)}
/* The arrow goes as the bar tightens: it means "the report is below", which is stale once
   the reader is inside the report, and losing it is half of why the shrunk bar reads as
   lighter. */
.choice-a{font:400 1rem/1 var(--sans);font-size:calc(1rem*(1 - .2*var(--t)));
opacity:calc(.8 - .8*var(--t));transition:font-size .2s ease,opacity .2s ease}
.panel{margin-top:3.2rem;scroll-margin-top:7rem}
/* A report's title needs room under it: the h2's default half-rem is set for a heading
   with a section under it, not for one that opens a ten-thousand-word document. */
.panel>h2{margin-bottom:1.9rem}
/* The comparison: a table, but the names are its masthead rather than a heading over
   the top of it, and the last row is what a reader does next.

   WHAT IS CENTRED IS THE PAIR, NOT THE TABLE. The two dataset columns straddle the page
   centre and the field labels hang off their left, in the margin — so the thing being
   compared sits in the middle and the labels read as an index down the side.

   Two steps, both stated as arithmetic rather than left to a layout mode to work out:

     1. .cmp-wrap is centred on the PAGE. left:50% resolves against its grid area, so
        the wrapper must be in the full-bleed track above (the full main column, which is
        centred in the viewport). In the default 38rem prose track it centres on the text
        column instead and the whole block lands ~5.75rem left of the hero.
     2. .cmp is pushed right by exactly `half the wrapper − one column − the labels`, so
        the pair's midpoint lands on the wrapper's midpoint. A percentage margin resolves
        against the wrapper's width, so this is one subtraction and needs no auto margins,
        no flex free space, and no negative margins.

   The three widths are custom properties: change one and the offset follows. */
.cmp-wrap{--cmp-label:10.5rem;--cmp-col:20rem;
position:relative;left:50%;transform:translateX(-50%);
width:min(100vw - 2.5rem,64rem);overflow-x:auto;margin:.4rem 0 0}
.cmp{border-collapse:collapse;table-layout:fixed;
width:calc(var(--cmp-label) + 2*var(--cmp-col));
margin-left:calc(50% - var(--cmp-col) - var(--cmp-label));
font:.86rem/1.55 var(--sans)}
.cmp th,.cmp td{text-align:left;vertical-align:top;padding:.62rem .9rem;
border-bottom:1px solid var(--hairline)}
.cmp thead th{border-bottom:0;padding:0 .9rem 1.35rem;width:var(--cmp-col)}
/* table-layout:fixed takes every column width from the FIRST row, so the corner cell
   has to carry the label width — the .cmp-k rule below is in the body rows, where fixed
   layout never looks. */
.cmp .cmp-corner{border:0;width:var(--cmp-label)}
.cmp-name{display:block;font:600 1.28rem/1.2 var(--serif);letter-spacing:-.012em;
color:var(--text-primary)}
/* Flush right, hard against the pair, and never wrapped: the labels are an index down
   the side of the comparison, and an index that breaks over two lines stops reading as
   one. --cmp-label is wide enough for the longest of them. */
.cmp-k{font:650 .68rem/1.9 var(--sans);text-transform:uppercase;letter-spacing:.08em;
color:var(--text-muted);width:var(--cmp-label);white-space:nowrap}
/* Everything `.cmp th` already sets — the rule, the alignment, the padding — needs a
   rule that OUT-SPECIFIES it, not merely one that follows it. */
.cmp th.cmp-k{border-bottom:0;text-align:right;padding:.62rem 1.1rem .62rem 0;
vertical-align:middle}
.cmp tbody td{color:var(--text-secondary)}
/* Each way in sits in the row of the figure it belongs to — the prompts against how
   many there are, the sample records against how many were published — with the figure
   at the column's left edge and the button at its right. */
.cmp-fig{display:flex;align-items:center;justify-content:space-between;gap:1rem}
.lbtns{display:flex;flex-wrap:wrap;gap:.7rem;margin:1.1rem 0}
.lbtn{display:inline-flex;align-items:center;gap:.45rem;padding:.45rem .8rem;
border:1px solid var(--accent-edge);border-radius:4px;background:none;text-decoration:none;
font:600 .92rem/1.3 var(--serif);color:var(--accent)}
.lbtn:hover{background:var(--accent-wash)}
.lbtn .ico{flex:0 0 auto}
.lbtn-m{font-weight:400;color:var(--text-muted)}
.lbtn:hover .lbtn-m{color:var(--text-secondary)}
/* inline-block so the link's underline stops at the last letter: the arrow is inside
   the <a>, and a text-decoration cannot be removed by a descendant — only escaped by
   one that is not inline. Measured at 4x, the underline ran on beneath it. */
svg.ext{display:inline-block;margin-left:.22em;vertical-align:-.05em;flex:0 0 auto}

/* Numbers. Direction is a labelled chip, never a colored numeral. */
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 2rem;
border-top:1px solid var(--border);padding-top:1rem;margin:1.4rem 0 1.6rem}
.tile-v{font:650 1.9rem/1.04 var(--sans);letter-spacing:-.022em;
font-variant-numeric:proportional-nums}
.tile-l{font:.82rem/1.35 var(--sans);color:var(--text-secondary);margin-top:.4rem}
.tile-s{font:.74rem/1.4 var(--sans);color:var(--text-muted);margin-top:.35rem}
.tile-f{margin-top:.5rem}
.tile.hero{grid-column:span 1}.tile.hero .tile-v{font-size:2.9rem}

/* Figures. The title is a caption, not a heading; the caption states the finding. */
figure{margin:1.6rem 0 1.9rem}
.fig-t{font:650 .84rem/1.35 var(--sans);color:var(--text-primary);margin-bottom:.15rem}
.fig-n{font:.78rem/1.5 var(--sans);color:var(--text-muted);margin:0 0 .5rem;max-width:52ch}
.fig-c{font:.8rem/1.55 var(--sans);color:var(--text-secondary);margin-top:.5rem;max-width:58ch}
.chart{width:100%;max-width:800px;height:auto;overflow:visible;display:block;margin:.2rem 0}
/* The flow schematic: vertical, and narrow enough to live in the reading column rather than
   the figure track, which is for measurements. 440px caps it, so it scales to ~0.81 in a
   358px phone column where a 12px label still lands near 10px — the horizontal version
   needed 720px and a scroll box to stay legible. Hairlines and two inks only; a schematic
   in the chart palette would read as a measurement. */
.flow{display:block;width:100%;max-width:440px;height:auto;margin:2.4rem 0 2.8rem}
.flow-t{font-family:var(--sans);font-size:12px;fill:var(--text-muted)}
.flow-t.strong{font-size:14px;font-weight:650;fill:var(--text-primary)}
.flow-cell,.flow-rule,.flow-arm{stroke:var(--axis);fill:none}
.flow-cell,.flow-rule{shape-rendering:crispEdges}
.flow-arm{stroke:var(--text-muted);stroke-dasharray:4 3}
.flow-dot{fill:var(--text-primary)}
.flow-head{fill:var(--axis)}
.lab,.val,.muted-svg{font-family:var(--sans)}
.lab{font-size:11.5px;fill:var(--text-secondary)}
.val{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}
.val.strong{fill:var(--text-primary);font-weight:650}
.key-in{font-style:italic}
.muted-svg{font-size:11px;fill:var(--text-muted)}
.grid{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}
.axis{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.rule{stroke:var(--text-muted);stroke-width:1;stroke-dasharray:4 3}
.legend{font:.76rem/1.4 var(--sans);color:var(--text-secondary);display:flex;gap:1rem;
flex-wrap:wrap;margin:.1rem 0 .2rem}
.legend .key{display:inline-flex;align-items:center;gap:6px}
.legend i{width:8px;height:8px;border-radius:50%;display:inline-block;flex:0 0 auto}

/* Tables: hairlines and alignment, no fills. */
.scroll{overflow-x:auto;margin:1rem 0 1.2rem}
table{border-collapse:collapse;width:100%;font:.83rem/1.5 var(--sans);
font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--hairline);
vertical-align:top}
th:first-child,td:first-child{padding-left:0}
th{color:var(--text-muted);font-weight:600;font-size:.76rem;
border-bottom:1.5px solid var(--border)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.ctr,th.ctr{text-align:center}
tbody tr:last-child td{border-bottom:0}
td b{color:var(--text-primary)}

code{font-family:var(--mono);font-size:.88em;color:var(--text-primary);word-break:break-word}
td code,th code,.meta code{background:var(--surface-2);padding:1px 4px}
pre{font-family:var(--mono);background:var(--surface-1);border:0;
border-left:2px solid var(--border);padding:.9rem 1.1rem;overflow-x:auto;
font-size:.79rem;line-height:1.65;color:var(--text-primary)}
/* The hairline is load-bearing on cream: the washes only reach ~1.15:1 against the
   page, so without an edge a chip stops reading as a chip. */
.chip{font:700 .66rem/1.5 var(--sans);text-transform:uppercase;letter-spacing:.07em;
padding:.1rem .38rem;background:var(--surface-2);color:var(--text-secondary);
border:1px solid var(--border);white-space:nowrap}
.chip.good{background:var(--good-wash);color:var(--good-ink);border-color:var(--good-edge)}
.chip.warn{background:var(--warn-wash);color:var(--warn-ink);border-color:var(--warn-edge)}
.chip.bad{background:var(--bad-wash);color:var(--bad-ink);border-color:var(--bad-edge)}
blockquote{margin:1rem 0 1.3rem;white-space:pre-wrap;font-size:1.02rem;line-height:1.55;
color:var(--text-primary);padding-left:1.15rem;border-left:2px solid var(--border)}
.warn-note{color:var(--text-primary);border-left:3px solid var(--warn);
background:var(--warn-wash);padding:.55rem .8rem;font-size:.92rem;margin:1rem 0}
.bad-note{color:var(--text-primary);border-left:3px solid var(--bad);
background:var(--bad-wash);padding:.55rem .8rem;font-size:.92rem;margin:1rem 0}
details{margin:1rem 0;border-top:1px solid var(--hairline);padding-top:.6rem}
summary{font:600 .84rem/1.5 var(--sans);cursor:pointer;color:var(--text-secondary)}
summary:hover{color:var(--text-primary)}
summary .sum-m{color:var(--text-muted);font-weight:400}
.det-body{padding-top:.5rem}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin:1.1rem 0}
.pane{min-width:0}
.pane-h{margin-top:0;color:var(--text-muted)}
/* The two panes hold different elements — a <blockquote> on one side, a .resp on the
   other — and their own margins do not agree, so one label sat closer to its text than
   the other. The pane sets the gap, both children take it. */
.pane>blockquote,.pane>.resp{margin:1rem 0 0}
/* The example carousel. Same outline-button family as .choice, one size down: a record
   id is a label, not a title, so it takes the mono face the ids use everywhere else. */
.carousel{margin:1.1rem 0}
.tabs{display:flex;flex-wrap:wrap;gap:.6rem;margin-bottom:1.1rem}
.tab{padding:.4rem .75rem;background:none;border:1px solid var(--accent-edge);
border-radius:4px;cursor:pointer;font:600 .88rem/1.3 var(--mono);
color:var(--accent)}
/* The only mono control on the page, and mono for its CONTENT rather than
   because it is a control: a carousel tab's label is a record id, so it
   matches the ids in the run notes rather than the buttons beside it. */
.tab:hover{background:var(--accent-wash)}
.tab[aria-selected=true]{background:var(--accent);border-color:var(--accent);
color:var(--surface-0)}
.pane-x>h4:first-child{margin-top:0}
.resp{white-space:pre-wrap;font-size:.94rem;line-height:1.6;color:var(--text-primary);
border-left:2px solid var(--hairline);padding-left:.9rem}
.pane.pipeline .resp{border-left-color:var(--series-3)}
.pane.plain .resp{border-left-color:var(--series-2)}
mark{background:var(--mark);color:inherit;padding:0 .1em}
/* Selection is the page's one piece of interaction colour, so it is the accent at full
   strength rather than the browser's blue. */
::selection{background:var(--accent);color:var(--surface-0)}
/* A link is a typographic object, not a coloured word: mono against the serif, bold
   enough to hold the accent, and underlined in the accent rather than in a tint of it.
   Buttons are unaffected — .lbtn, .choice, .tab and .skip each set their own font
   shorthand, which beats a bare element selector. */
/* A LINK IS MARKED, NEVER RE-FACED. It takes the typography of the text around it — serif
   in prose, sans in the rail — and the mark is the accent plus the 2px accent underline.
   Nothing here sets a face, a size or a weight, which is the whole rule.

   It used to be mono 600 at .92em, on the reasoning that mono carries IDENTITY and a link is
   a thing you go and fetch. That conflated two different kinds of content: a run id or a path
   IS a literal string, and mono is right for it; "Teaching Claude Why" is language, and
   setting it in mono changed x-height and letterfit mid-sentence in every paragraph of both
   reports. Mono now means a literal string and nothing else, which makes it mean more.

   WEIGHT IS PART OF THE MARK, not part of the face: 600 against the surrounding text, which
   is what a link inherits its face and size from. At the inherited weight the mark was colour
   and an underline only, and a link in a long report read faint.

   Colour is not carrying this alone: a 2px underline survives greyscale, print (where the
   print rule turns links black and keeps the underline) and colour-blindness. */
a{font-weight:600;color:var(--accent);text-decoration:underline;
text-decoration-thickness:2px;text-underline-offset:.2em;
text-decoration-color:var(--accent)}
/* A citation marker: the source, raised, so the sentence it hangs off reads uninterrupted.
   It carries the accent underline like every other link — it was the one link on the page
   without one — but THINNER (the brand's 2px under a ~9px numeral is proportionally what 4px
   would be under body text) and drawn on the <sup> RATHER THAN THE ANCHOR. That is not tidying:
   a text-decoration is positioned from the element's own baseline, and the anchor's baseline is
   the paragraph's, so underlining .cite-n put two dashes below and left of the raised numerals,
   measured at 6x. On the <sup> it tracks the digit. The arrow escapes it, as on every other
   link, because inline-block breaks the propagation. line-height:0
   keeps it out of the line box, so a marker cannot open up the leading of the paragraph it
   sits in. The padding is hit area — inline padding enlarges the target without moving
   anything — and the negative margin gives back the space it would otherwise add. WCAG 2.5.8
   exempts a target inside a sentence from the 24px minimum, so this is comfort, not
   conformance.

   nowrap holds the numeral to its own arrow: the arrow is an atomic inline, so a line was
   free to break between the digit and the mark that belongs to it. Breaking BETWEEN two
   markers, or before one, is forbidden by the word joiner the renderer emits — see
   WORD_JOINER. */
.cite-n{text-decoration:none;padding:.22em .25em;margin:0 -.25em;white-space:nowrap}
/* The raise happens ONCE, here, and not on the anchor: <sup> already carries the UA's own
   vertical-align:super and font-size:smaller, so raising and shrinking the anchor too shifted
   the digits twice — measured at 4x, they sat above the cap line at ~10px while the separating
   comma, which had only the anchor's single shift, sat below them. Stating the size and the
   shift explicitly on the <sup> also replaces `smaller`, which is a relative keyword no two
   engines have to agree on. */
.cite-n sup{font-size:.72em;vertical-align:super;line-height:0;
text-decoration:underline;text-decoration-thickness:1.5px;text-underline-offset:.18em;
text-decoration-color:var(--accent)}
/* The marker's arrow. `vertical-align:.12em` lifts it off the <sup>'s own baseline to sit
   against the numeral's body — measured at 6x, on the baseline it read as dropped below the
   digit it belongs to. inline-block for the same reason as svg.ext: a text-decoration cannot
   be removed by a descendant, only escaped by one that is not inline. */
svg.ext-c{display:inline-block;margin-left:.08em;vertical-align:.12em}
.cite-n:hover{background:var(--accent-wash)}
/* Consecutive markers need separating, or two adjacent numerals read as one number — 1 and 2
   side by side is "12". A raised comma did that job before each marker carried an arrow; the
   arrow now separates them, so this is a hair of space instead of another glyph. It has to
   beat the marker's own -.25em, which is there to give back what its hit-area padding adds. */
.cite-n+.cite-n{margin-left:-.08em}
a:hover{background:var(--accent-wash)}
/* button is in this list because the page HAS buttons — the chooser and the carousel —
   and without it the only controls on the page fall back to the UA's own ring, which is
   the one focus treatment nobody here designed. */
a:focus-visible,button:focus-visible,[tabindex]:focus-visible,
summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
/* TWO ROWS: the credit, then who made it on the left and where to go on the right.

   The split is the footer's oldest rule and it is kept — but it belongs to a row of its
   own, with two items a side. It was on the footer itself, where four children had
   outgrown it: the byline took a full-width line, the feedback sentence and the maker's
   name split the next, and the two destinations wrapped alone onto a third, so the closing
   band read as three rows with three different alignments. `.foot-row` is the split; the
   byline, which belongs to neither half of "who made it / where to go", has the line above
   it to itself. */
footer.foot{margin-top:5rem;padding-top:1.1rem;border-top:1px solid var(--border);
font:.85rem/1.6 var(--sans);color:var(--text-muted)}
footer.foot p{margin:0;color:inherit}
.foot-row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;
gap:.5rem 2rem}
/* No size step: in a row with the colophon these two are its peers, and a 1rem pair beside
   a .85rem one is a mismatch, not a ranking. */
.foot-links{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem}
.foot-by{margin-bottom:1.4rem}
.foot-authors{color:var(--text-secondary)}
footer.foot .foot-by p+p{margin-top:.3rem}
/* Row-gap 0: the key wraps to as many lines as it needs, and column-gap does the
   separating so no comma or bullet has to be typed between institutions — a glyph put
   there in CSS is still read out, and the key's whole job is to be skipped.
   2rem, not the 1.15rem it had: at 1.15 the space between two institutions was barely
   wider than the word space inside "University of Santiago de Compostela", so five items
   read as one run-on sentence with digits in it. nowrap for the other half of that —
   an institution that breaks across two lines is the same failure from the other side. */
.foot-affil{display:flex;flex-wrap:wrap;gap:0 2rem}
.foot-affil>span{white-space:nowrap}
/* The quietest line on the page, and the same separator idiom as the key above it: two
   spans held apart by column-gap, with nothing typed between them. */
.foot-colophon{display:flex;flex-wrap:wrap;gap:.3rem 1.6rem}
/* line-height 0 keeps a superscript from stretching the line it sits on. */
.foot-by sup{font-size:.74em;line-height:0;padding-right:.06em}
/* THE ONLY MARKS IN THE FOOTER NAME A DESTINATION. `assets/sf.png` used to sit inside
   "A project by Sentient Futures" as a 15px rounded square, and it is a picture of a name
   printed 4px to its right — a third link idiom in a footer that had two already, and the
   only saturated colour down here landed on the least important line. Dropped with its
   `.ico-img`/`.maker` rules and the maker_icon argument that fed it, rather than left as a
   class nothing emits.

   An icon link declares NO face and no size, on purpose: it is not a control — there is
   nothing to press in the footer, only somewhere to go — so it takes its tier's size and
   the footer's own sans, and the bare `a` rule gives it the accent, the underline and the
   600. Declaring serif here put a serif 600 link next to a sans 400 one in the same row;
   measured, that was the whole of the mismatch. */
.ilink{display:inline-flex;align-items:center;gap:.45rem}
.ilink:hover{background:var(--accent-wash)}
/* The hero's illustration. Dashed while empty, so an unfilled slot reads as deliberate
   rather than as a broken asset; once filled it is line art on the paper, with no frame
   of its own. */
.illo{aspect-ratio:16/6;margin:2.6rem 0 0;border:1px dashed var(--accent-edge);
display:flex;align-items:center;justify-content:center}
.illo span{font:650 .7rem/1 var(--sans);text-transform:uppercase;letter-spacing:.12em;
color:var(--accent)}
.illo.art{aspect-ratio:auto;border:0;background:none;display:block;margin:3rem 0 .4rem}
.illo.art img{display:block;width:100%;height:auto;max-width:46rem}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--text-primary);
color:var(--surface-0);font:12px/1.4 var(--sans);padding:5px 8px;transition:opacity .1s;
z-index:9;max-width:320px}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}
*{transition:none!important;animation:none!important}}
/* Below the width the offset needs, the comparison goes back to being an ordinary
   full-width table: a centred pair that runs off the left of the screen is worse than
   an uncentred one. */
/* Below the width the offset needs, the pair cannot straddle the centre without the
   labels running off the left of the screen, so the comparison goes back to being an
   ordinary full-width table. */
@media (max-width:1000px){.cmp-wrap{position:static;left:auto;transform:none;width:100%}
.cmp{table-layout:auto;width:100%;margin-left:0}
.cmp-k{width:8rem;white-space:normal;text-align:left}.cmp thead th{width:auto}}
/* Below the width the rail column and a 38rem measure both fit in, there is no beside to
   put the contents to. It goes to the top of the report instead, static, as a wrapped
   block — one report's contents where its reader is about to start, not a row hung off the
   bottom of the bar. Above this and below the shell's own 67rem the reading column simply
   narrows, which needs no rule. */
@media (max-width:900px){.explore-body{grid-template-columns:minmax(0,1fr)}
.railcol,.panels{grid-column:1}
/* Held to the reading measure and given air, so it reads as the head of the document
   rather than as more of the bar. Its own margin places it here, so the padding that lines
   it up with a title beside it goes.
   THE RULE AND THE AIR BELONG TO THE RAIL, NOT TO ITS COLUMN. On the column they were drawn
   whether or not a rail was in it: with no report open, both rails are hidden and the empty
   column still put a hairline and 4rem of space under the chooser — a line across the page
   above the footer, on narrow screens only, separating nothing. .rail[hidden] is display:none,
   so hung on the rail itself they arrive with the contents they belong to. */
.railcol{padding-top:0}
.rail{position:static;max-height:none;padding:0 0 .6rem;margin-top:3.6rem;
max-width:38rem;border-bottom:1px solid var(--hairline);
display:flex;flex-wrap:wrap;column-gap:.4rem}
.rail a{padding:.2rem .5rem;border-left:0;border-bottom:2px solid transparent}
/* The stages go, and the contents become the four beats. Beside the report the two levels
   are a tree — an indented triplet under a bold parent — and the difficult-advice report
   names the same three stages twice on purpose, once to explain them and once to walk
   them. Flattened into a wrapped row the tree is gone and the duplication is all that is
   left: nine items in which "Stage 2 · the reasoning" appears twice, identically, with
   nothing to say which is which. Four beats on four lines say the same thing about the
   report's shape, which is what a reader about to start it needs. */
.rail .r-s{display:none}
.rail .r-b{margin-top:.3rem;flex:0 0 100%}
.rail a[aria-current=true]{border-left-color:transparent;border-bottom-color:var(--accent)}}
/* One column, WITH THE NAMED LINES STILL DEFINED. Collapsing the grid to a bare
   minmax(0,1fr) and re-placing the children with `section>*{grid-column:1}` looks like it
   works and does not: that selector is (0,0,1) and loses to `section>figure` (0,0,2) and
   to `section>.explore-body` (0,1,1), which keep pointing at text-start/full-end after the
   names have been deleted — so every figure, the comparison, and the whole chooser (bar,
   rails, both reports) land in a 0px implicit track and the prose inside them wraps one
   word per line. Naming both lines on the single track instead leaves every existing
   placement valid, so nothing has to be re-placed and no !important is needed. */
@media (max-width:760px){
section{grid-template-columns:[text-start] minmax(0,1fr) [text-end full-end]}
.pair{grid-template-columns:1fr}
/* The bar stays pinned on a phone, so it has to stay ONE ROW: stacking the two buttons
   is ~10rem of permanent chrome, a quarter of a small screen. Two columns and tighter
   type instead — restating the tokens, so the shrink still interpolates off them. */
.choicebar{--pad:.6rem;--gap:.7rem;--btn-y:.7rem;--btn-x:.8rem;--label:1rem}}
@media (max-width:620px){body{font-size:1rem}.shell{padding:0 16px 70px}
/* Tighter again, and the arrow goes: at this width both labels wrap to two lines and
   space-between drops the arrow beside a ragged edge. It is decorative — the pair still
   reads as a control. */
.choicebar{--pad:.5rem;--gap:.6rem;--btn-y:.6rem;--btn-x:.7rem;--label:.95rem}
.choice-a{display:none}
h1{font-size:1.9rem}h2{font-size:1.6rem}h3{font-size:1.25rem}h4{font-size:1.06rem}
.lede{font-size:1.1rem}
/* THESE TWO GAPS STAY CLEAR OF `.hero-intro p+p`, which is 1.4rem and is not restated
   here. Tightened to 1.6rem and 1.2rem they were not: the title-to-intro break — the
   largest one in the hero — got LESS air than the space between two paragraphs of the
   intro, and the whole hero read as one block. The ladder mobile wants is the one the
   wide layout has, strictly descending as the break gets smaller: 2.6rem above the title
   (2.2 here plus the art's own .4rem bottom, set by `.illo.art`), 2.2 above the intro,
   1.8 above the two techniques, 1.4 between paragraphs. */
.hero{padding:1.8rem 16px 4rem}.hero h1{margin-top:2.2rem;font-size:2.2rem}
.hero-intro{margin-top:2.2rem}.hero-intro p{font-size:1.05rem}
/* The two rules inside the intro keep their width and tighten their air: 48px either side is
   a sixth of a phone screen twice over, and the ladder above is tightened here for the same
   reason. */
.hero-intro>p:nth-child(2)::before,.hero-intro>ol+p::before{margin:32px auto}
/* One column: two of them inside a 390px viewport are ~16 characters each.
   AND IT CENTRES, which the two-column form must not. The reason the pair goes flush left up
   there is that centring two columns leaves four ragged edges; one column has two, and the
   hero either side of it — title, both paragraphs, the closing pair — is centred, so flush
   left made the stack read as a different kind of block instead of the same one narrower.
   It also comes off the edges: its own inset plus the shell's is the air the centred prose
   above it has at the ends of its lines, which the stack had none of.
   The box comes with it — a boxed technique stacked is the same object narrower, and the
   3/4 centred hairline this used to swap in belonged to the borderless form. Its inner
   padding drops to 20px, because 24px inside a ~326px item is a seventh of the line. */
.npair{grid-template-columns:minmax(0,1fr);gap:1.8rem;margin:1.8rem auto 16px;
padding:0 1rem;max-width:32rem;text-align:center}
.npair>li{padding:20px}
.tiles{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.2rem}
.illo{aspect-ratio:16/9}
/* THE LABEL GOES ABOVE ITS TWO CELLS, and this is a bug fix, not a preference. `.cmp-k`
   carries a fixed 8rem — a fifth of a phone — and with `table-layout:auto` the two data
   columns cannot get below their min-content width, so the table measured 422px inside a
   358px wrapper at 390 (382 at 414): the SECOND dataset's column was cut off, reachable
   only by swiping a table that gives no sign it scrolls. It clears on its own at ~450px.
   The rows become two-column grids with the label spanning both, so the comparison stays a
   comparison — side by side is the whole point of it — and the row's rule moves to the
   <tr>, or each cell draws its own and the line across the row becomes two short ones. */
.cmp,.cmp thead,.cmp tbody{display:block;width:100%;margin-left:0}
.cmp tr{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:1.1rem;
border-bottom:1px solid var(--hairline)}
.cmp thead tr{border-bottom:0}
.cmp-corner{display:none}
.cmp th.cmp-k{grid-column:1/-1;text-align:left;width:auto;white-space:normal;
padding:.9rem 0 .6rem}
.cmp td,.cmp thead th{padding:0 0 .95rem;width:auto}
.cmp thead th{padding-top:.9rem}
.cmp td{border-bottom:0}}
@media print{
@page{margin:16mm 14mm}
:root{--surface-1:#fff;--surface-2:#fff;--hairline:#d8d6cd}
body{font-size:10.5pt;line-height:1.5}
/* Neither control survives paper: the bar has nothing to press and the rail is a list of
   links to pages the reader is holding. */
#tip,.skip,.choicebar,.railcol{display:none}
.shell,section,.explore-body{display:block;max-width:none}
.hero{padding:0 0 1.5rem;display:block;text-align:left}
/* A sheet of paper is narrower than the bleed the centred pair needs, and the labels
   would print off the left edge. */
.cmp-wrap{position:static;left:auto;transform:none;width:100%;overflow:visible}
.cmp{table-layout:auto;width:100%;margin-left:0}
/* A printed page is not a page anyone can click, so both reports print, whichever
   one is open on screen — and every example in the carousel prints, not just the tab
   that happened to be showing. */
.panel[hidden],.pane-x[hidden]{display:block!important}
.tabs{display:none}
p,ul,.dek,.fig-c,.fig-n,.lede{max-width:none}
h1,h2,h3,h4,.fig-t,.dek{break-after:avoid-page}
figure,.tiles,table,.pair,blockquote,.flow{break-inside:avoid-page}
.panel{break-before:page}
tr,li{break-inside:avoid}
thead{display:table-header-group}
details{display:block}details>div{display:block!important}summary{list-style:none}
.tiles{display:grid;grid-template-columns:repeat(3,1fr)}
main a[href^="http"]::after{content:" (" attr(href) ")";font-size:.85em;
color:var(--text-muted);word-break:break-all}
.chip,rect,circle,path,line{-webkit-print-color-adjust:exact;print-color-adjust:exact}
/* On paper nothing is pressable and nothing is indigo: the label, its underline and the
   outline button all go to ink. text-decoration-color has to be said out loud — the base rule
   sets it to the accent, so overriding `color` alone left black text under an indigo rule, and
   `.cite-n sup` is not an anchor, so the bare `a` override never reached it at all. */
a,.lbtn,.cite-n sup{color:var(--text-primary);text-decoration-color:currentColor}}
"""

JS = """
(function(){var t=document.getElementById('tip');
document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]');
if(!el){t.style.opacity=0;return;}t.textContent=el.getAttribute('data-tip');t.style.opacity=1;});
document.addEventListener('mousemove',function(e){if(t.style.opacity=='1'){
t.style.left=Math.min(e.clientX+12,window.innerWidth-t.offsetWidth-8)+'px';
t.style.top=(e.clientY-32)+'px';}});
/* The example carousel. Its own block, and before the chooser's early return, because a
   page can carry examples without carrying a chooser. One pane is already visible in the
   markup, so with this script absent the carousel still shows a record. */
[].forEach.call(document.querySelectorAll('.carousel'),function(c){
var tabs=[].slice.call(c.querySelectorAll('.tab'));
function show(b,focus){tabs.forEach(function(o){var on=o===b;
o.setAttribute('aria-selected',on?'true':'false');
/* The tab order holds ONE tab per set, and it is the open one: a reader tabbing through a
   report should pass the carousel in one press, then reach its records with the arrows —
   which is the half of the pattern role='tab' was promising and not delivering. */
o.setAttribute('tabindex',on?'0':'-1');
var p=document.getElementById(o.getAttribute('data-pane'));
if(p){if(on){p.removeAttribute('hidden');}else{p.setAttribute('hidden','');}}});
if(focus)b.focus();}
tabs.forEach(function(b,i){b.addEventListener('click',function(){show(b);});
b.addEventListener('keydown',function(e){var k=e.key,n=null;
if(k==='ArrowRight'||k==='ArrowDown')n=tabs[(i+1)%tabs.length];
else if(k==='ArrowLeft'||k==='ArrowUp')n=tabs[(i-1+tabs.length)%tabs.length];
else if(k==='Home')n=tabs[0];else if(k==='End')n=tabs[tabs.length-1];
if(n){e.preventDefault();show(n,true);}});});});
var choices=[].slice.call(document.querySelectorAll('.choice'));
if(!choices.length)return;
/* Where the bar SITS, not where it is painted. Once sticky takes hold, the bar's own
   getBoundingClientRect() and offsetTop both report the shifted position, so measuring
   from it would scroll to wherever the reader already was. .explore-body never moves and
   the bar is its first child, so its top IS the bar's flow top — which is also the
   sticky threshold, so nothing jumps as the bar pins. */
var flow=document.querySelector('.explore-body');
/* The bar tightens once, when the reader crosses TIGHT pixels past it, and loosens again
   at LOOSE. Two thresholds rather than one: with a single one, a reader parked on the
   boundary flips the bar back and forth, and the size change is a layout change. All the
   script does is set the flag — the sizes and the animation are CSS.

   Measured from .explore-body for the reason above and one more: the shrink cannot move
   the wrapper's top, so this cannot feed itself. Measuring the bar, whose height is the
   thing being changed, is the flicker. */
var TIGHT=96,LOOSE=24,queued=0;
/* The rails: one per report, hidden with the panel it belongs to, so the contents a reader
   sees are always the contents of what they are reading. ``heads`` and ``links`` are the
   open report's headings and its rail's links, cached when it opens — their POSITIONS are
   read live every frame, but re-querying the DOM on each one is not free and nothing adds
   an anchored heading later (the appendix's live inside drawers and carry no id). */
var rails=[].slice.call(document.querySelectorAll('[data-rail]'));
var heads=[],links=[];
/* One callback for both pinned things: the bar's flag, then where the reader is. */
function onScroll(){queued=0;if(!flow)return;
var past=-flow.getBoundingClientRect().top;
flow.classList.toggle('tight',past>(flow.classList.contains('tight')?LOOSE:TIGHT));
/* The current beat or stage is the last heading the reader has ARRIVED AT, and the line
   for that is the heading's own scroll-margin-top: the CSS already states how far below
   the top of the screen a heading lands when it is linked to, so the same number decides
   whether it has been reached. Read from the element rather than typed here — and it means
   nothing measures the bar, whose height is the thing that changes.

   Nothing is marked while the reader is still above the first heading, which is honest:
   they are not in a beat yet. */
var cur='';
heads.forEach(function(h){if(h.el.getBoundingClientRect().top<=h.line)cur=h.el.id;});
/* AT THE BOTTOM, THE LAST BEAT IS THE CURRENT ONE, whether or not its heading ever reached
   the line. It cannot: the appendix is the last beat and its drawers are closed, so there is
   less content below it than there is screen — measured at 1440x900, 659px of page under a
   heading that would need to climb 1,349px. So the rail marked a stage inside the worked
   example while the reader was looking at the appendix. Correcting at the bottom rather than
   shortening the line, because the line is the CSS's own headroom and is right everywhere
   else; and it self-corrects when a reader opens a drawer, since the heading can then reach
   the line the ordinary way. */
if(heads.length&&innerHeight+scrollY>=document.documentElement.scrollHeight-2)
cur=heads[heads.length-1].el.id;
links.forEach(function(a){
if(cur&&a.getAttribute('href')==='#'+cur){a.setAttribute('aria-current','true');}
else{a.removeAttribute('aria-current');}});}
window.addEventListener('scroll',function(){
if(!queued)queued=requestAnimationFrame(onScroll);},{passive:true});
window.addEventListener('resize',onScroll);onScroll();
function open(id,to){
choices.forEach(function(b){var on=b.getAttribute('data-panel')===id;
/* aria-expanded, not aria-selected: this is a disclosure pair. Both can read false, which
   is the state the page loads in and the state a tablist is not allowed to have. */
b.setAttribute('aria-expanded',on?'true':'false');
var p=document.getElementById(b.getAttribute('data-panel'));
if(p){if(on){p.removeAttribute('hidden');}else{p.setAttribute('hidden','');}}});
links=[];
rails.forEach(function(r){var on=r.getAttribute('data-rail')===id;
if(on){r.removeAttribute('hidden');links=[].slice.call(r.children);}
else{r.setAttribute('hidden','');}});
var panel=document.getElementById(id);
heads=panel?[].slice.call(panel.querySelectorAll('h3[id],h4[id]')).map(function(el){
return {el:el,line:parseFloat(getComputedStyle(el).scrollMarginTop)+1};}):[];
onScroll();
if(!to)return;
var target=document.getElementById(to===true?id:to);
if(!target)return;
/* Choosing a report puts the bar at the top of the screen; a deep link to a beat INSIDE
   a report goes to the beat, clear of the pinned bar. Both are scrollIntoView plus
   scroll-margin-top rather than arithmetic: the headroom the bar needs is stated once,
   in the CSS, where a native fragment jump reads it too — and the smoothness stays
   html{scroll-behavior}, so prefers-reduced-motion can still turn it off. */
(target.classList.contains('panel')&&flow?flow:target).scrollIntoView();}
function mark(id){if(history.replaceState)history.replaceState(null,'','#'+id);
else location.hash=id;}
/* A hash may name a panel (#dad) or anything inside one (#dad-weak, from a quoted
   finding). Either way the panel it lives in is the one to open. */
function fromHash(){var id=(location.hash||'').slice(1);if(!id)return false;
var el=document.getElementById(id);var p=el&&el.closest?el.closest('.panel'):null;
if(!p)return false;open(p.id,id);return true;}
/* Pressing a tab opens its report and puts the bar back at the top of the screen. */
choices.forEach(function(b){b.addEventListener('click',function(){
var id=b.getAttribute('data-panel');open(id,true);mark(id);});});
window.addEventListener('hashchange',fromHash);
/* Wait for load, not parse: the hero image is a data URI several megabytes long, and
   scrolling to a deep-linked beat before it has laid out puts the reader thousands of
   pixels away from it once the image finally takes up its space. */
if(document.readyState==='complete')fromHash();
else window.addEventListener('load',fromHash);})();
"""


def _meta(name, content, prop=False):
    """One head tag, or "" for an empty value.

    Double-quoted, unlike the rest of the page: ``esc()`` escapes ``"`` and deliberately
    does not escape ``'``, and these carry authored prose, where an apostrophe is a matter
    of time. Everything else on the page writes attributes the renderer itself composes.
    """
    key = "property" if prop else "name"
    return f'<meta {key}="{esc(name)}" content="{esc(content)}">\n' if content else ""


def head_meta(*, title, description="", site_url="", preview_url=""):
    """What a crawler and a link preview see.

    ``noindex`` is unconditional. The page is handed to a reader by whoever sends it, not
    found — and a ``robots.txt`` ``Disallow`` does not carry that on its own, since a URL
    that is linked can still be indexed by reference. It costs nothing offline and comes
    off in one line if the page is ever announced.

    The preview tags need the hosted URL, so they are emitted only when one is supplied:
    a build with no ``--site-url`` is the file that opens from disk or arrives by email,
    and it says nothing about where it lives. ``og:image`` is the one reference on this
    page that cannot be a data URI — a preview image is fetched out of band by whoever is
    rendering the card — so without one the card declares itself ``summary`` rather than
    promising a large image it has not got.
    """
    tags = '<meta name="robots" content="noindex,nofollow">\n' + _meta("description", description)
    if site_url:
        tags += (_meta("og:type", "website", prop=True)
                 + _meta("og:title", title, prop=True)
                 + _meta("og:url", site_url, prop=True)
                 + _meta("og:description", description, prop=True)
                 + _meta("og:image", preview_url, prop=True)
                 + _meta("twitter:card", "summary_large_image" if preview_url else "summary"))
    return tags


def icon_links(icons=()):
    """The tab icon, carried inside the page like every other picture on it.

    ``icons``: ``[(px, data_uri)]``, one link each. This is the page's only ``<link>``,
    and the one hole in the rule that ``test_is_self_contained`` enforces — a favicon has
    no other spelling, since a browser will not read one out of a ``<meta>`` and the
    implicit ``/favicon.ico`` lookup exists only for a hosted copy. So it is inlined for
    the same reason the hero is: the file that opens from disk or arrives by email keeps
    its icon, and nothing new has to travel beside the page.

    Deliberately NOT part of ``head_meta()``, which is what a crawler and a link preview
    see. An icon is neither.

    Sizes are declared rather than left to the browser. The art is hairline pencil work
    and each PNG is decimated for the size it names (see ``make_preview.py``); handing
    over one image and letting the browser scale it re-averages the ink and throws that
    away, which is the whole reason there is more than one file.
    """
    tags = ""
    for px, uri in icons:
        if not uri:
            continue
        if not uri.startswith("data:"):
            raise ValueError("a tab icon must be a data: URI — the page is one file")
        tags += f"<link rel='icon' sizes='{int(px)}x{int(px)}' href='{uri}'>\n"
    return tags


def document(*, title, masthead, body, footer="", description="", site_url="",
             preview_url="", icons=()):
    """The shell. One file, one theme, no external anything.

    There is no contents rail: the page is a hero, three short sections and a choice, and
    a list of five links beside that is furniture. Everything a reader navigates to is
    either on the first screen or one button away.
    """
    return (f"<!DOCTYPE html>\n<html lang='en'>\n<meta charset='utf-8'>\n"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
            f"<meta name='color-scheme' content='only light'>\n"
            + head_meta(title=title, description=description, site_url=site_url,
                        preview_url=preview_url)
            + icon_links(icons)
            + f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n"
            f"<a class='skip' href='#intro'>Skip to content</a>\n"
            f"{masthead}"
            f"<div class='shell'>\n<main id='main'>\n{body}\n"
            + (f"<footer class='foot'>{footer}</footer>\n" if footer else "")
            + f"</main>\n</div>\n"
            f"<div id='tip'></div>\n<script>{JS}</script>\n</html>\n")
