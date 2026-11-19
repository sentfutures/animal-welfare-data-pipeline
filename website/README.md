# `website/` — the handoff page

One self-contained HTML file covering both datasets, written for one reader: someone who
runs midtraining at another lab, has no context on this project, and has about forty
seconds to decide whether to keep reading. It publishes to GitHub Pages as it stands,
emails, and opens offline from the filesystem.

The two datasets are **Difficult advice** (`dad`) and **Synthetic documents** (`sdf`) —
"corpus" and "corpora" are not words this page uses.

This is **not** the Streamlit corpus-audit page. That is an internal review tool
organised by what the eval measured; this is organised by what a reader needs, in order.

## Build

```bash
python website/build_website.py \
  --dad-run outputs/dad/runs/2026-07-29_12-26_archetype200 \
  --sdf-run outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5
# -> website/index.html
```

Those two runs are the pinned ones behind the current build. `--run` still works as an
alias for `--dad-run`. `--content` (repeatable) overrides the prose files, `--example` and
`--sdf-example` override each report's worked example, `--out-dir` writes elsewhere.

**The page does not document how to run the pipeline.** No install, no invocation, no
costs, no per-stage model table — that is this repository's own README and `CLAUDE.md`.
What the page carries is the process, one record's whole trail through it, and the run's evidence in a closed appendix.

`--sdf-run` is optional. Without it the synthetic documents' column says "not published
yet" and its report keeps its lede, says no run output was supplied, and offers the two
ways out — the page still builds, and carries no dead links.

Four of the document report's inputs — `audit/compliance_report.json`,
`audit/card_fidelity_report.json`, `audit/realism_ablation.json` and
`audit/vendi_curve.json` — are **not** written by `evals/audit_sdf.py`, and only
`fullscale-500-opus5` carries them. Only the Vendi curve is rendered by the current
document report (inside its "Composition and diversity" drawer, which names the file it
wanted when the curve is absent); the other three are loaded and still feed `facts()`, but
no block builds from them. Building against `2026-07-11_20-06_matrix100-cli` is the
cheapest way to exercise the degraded path.

The paid audit pass only affects the appendix. Without it, the judged drawer says no paid
pass ran; the beats above the appendix are unchanged, because none of them depends on a
judge — the flow is authored and the worked example comes off the step files:

```bash
python evals/audit_dad.py --input outputs/dad/runs/<run_id> --reasons
python evals/diversity.py --input outputs/dad/runs/<run_id>
```

## Hosting

The page is one file with **one** file beside it, so serving it is an upload: nothing in it
refers to its own URL apart from the preview tags, every other asset is inlined and every
outbound link is absolute. It stays that way — a hosted copy is never a second,
un-self-contained build.

`.github/workflows/pages.yml` is the deploy. It publishes to `reasoning.sentientfutures.ai`
on a push to `main` that touches the built page or the card image, and it does **not** run
the builder: it copies `website/index.html` and `website/assets/preview.png` into the site
root and hands them to Pages. So the committed `index.html` is what is live, and rebuilding
it is a commit like any other.

**It is unlisted, and that is a meta tag rather than a `robots.txt`.** Every build carries
`<meta name="robots" content="noindex,nofollow">`, unconditionally. The two files do
different jobs and are easy to mix up: `robots.txt` governs **crawling**, the tag governs
**indexing**. `noindex` is not a `robots.txt` directive at all — Google dropped support for
the unofficial one in 2019 — so a `Disallow` cannot say it.

**Do not `Disallow` this page.** A crawler that is refused the file never reads the tag
asking it not to index, and a URL someone links to can still be indexed with no content
behind it — the `Disallow` makes the outcome *worse*, not better. Let it be crawled and let
it say `noindex`. (A host that sets `X-Robots-Tag: noindex` as a header does the same job for
consumers that only read headers; it is a belt to the tag's braces, unlike a `Disallow`.)
Removing the tag is one line in `render.head_meta()` if the page is ever announced.

Because a pasted link is then the *only* way anyone arrives, a hosted build carries preview
tags. They need to know where the page lives, so naming the site is what turns them on:

```bash
python website/build_website.py --dad-run <run> --sdf-run <run> \
    --site-url    https://reasoning.sentientfutures.ai/ \
    --preview-url https://reasoning.sentientfutures.ai/preview.png
# -> website/index.html   (the workflow serves the card image from assets/)
```

`--site-url` adds `og:title` / `og:url` / `og:description` / `twitter:card` and points
`og:image` at `preview.png` beside the page, because a card renderer fetches the image out
of band and a data URI is no use to it. On its own it also **copies** that file into the
output directory. The command above passes `--preview-url` naming the same URL, which
suppresses the copy: the deploy already stages the image from `website/assets/`, so a second
one committed next to the page would be a drifting duplicate. (`.gitignore` covers it if
someone rebuilds without the flag.) `--preview-url` is otherwise for an image hosted
somewhere else entirely; with no image at all the card declares `summary` rather than
promising a large image it has not got. With neither flag the build says nothing about where
it lives, which is right for the copy that opens from disk or arrives attached to an email;
the build line prints `preview=no`.

**The card image and the tab icons are the hero.** `python website/make_preview.py` redraws
all three from `assets/hero.png` and their output is committed; it needs Pillow, which is
why it is a separate script and not part of the stdlib-only `build_website.py`.

- `assets/preview.png` — the butterfly trimmed to its own bounds and centred on the page's
  paper at 1200×630. No crop through the drawing, no filter, no text baked over it.
- `assets/favicon-16.png`, `assets/favicon-32.png` — the butterfly *alone*, without the
  dashed trail, squared up on the same paper. These are inlined as data URIs, so they are
  not among the files a deploy carries, and they are the one `<link>` on the page: a
  favicon has no other spelling, and `test_is_self_contained` allows that shape and no
  other.

Two sizes rather than one, and the ink is thickened before each is shrunk. Both fall out of
the same fact — the hero is hairline pencil work. A straight resize to 16px leaves 14 of 256
pixels carrying any ink, the darkest at 3.9:1 on the paper, which is a blank cream square in
a tab; a darkest-pixel-wins pass first is the ordinary way to decimate line art and holds the
strokes at full strength. That is a resampling choice and changes nothing about `hero.png`.
It is also why `sizes=` is declared: hand a browser one 32 and it scales to 16 itself,
averaging the ink straight back out. The two filter radii in `ICONS` were tuned by eye
against a contact sheet — lower reads washed out, higher smears the wing veins into a blob.

The `description` those tags use is prose, authored in `content_page.md` under the
`description` id like every other word on the page. It is the one id that never renders in
the document, so it is flat text — `render.plain_md()` strips the markdown subset.

The datasets' licence is **not** on the page: CC0-1.0 is declared by hand in the Hugging
Face card's frontmatter, beside the files it governs. (The pipeline code is Apache-2.0 —
a separate licence, on a separate thing.)

Deploy from `main` rather than from a laptop, so the repo stays the source of truth and the
hosted copy is never edited in place.

## The page

| Anchor      | What it is                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hero        | The illustration, the title, and the lines that follow from it — centred, and carrying the `#intro` id. Four blocks: the finding (_Teaching Claude Why_ and the SDF post, the page's one credit to both), the sentence that introduces the pair, **the two techniques as two columns** (`render.named_pair()`, synthetic documents first), then what we built on them, in two paragraphs. Four paragraphs of prose and it stops. The pair is a figure between the second paragraph and the third, not a list inside one: those two techniques _are_ the two datasets below, in the same order, and ~90 words of definitional prose with two digits in front of it did not say so. It carries **no index over either name** — an eyebrow ("Technique 1") names what the heading under it already says — and no tie line naming the dataset each produced, which was tried and read as awkward against names that barely differ. Nothing else: no lede, no provenance, no tiles, and no "Intro" heading over a paragraph that needs no introducing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `#datasets` | The comparison. No heading over it: the two column mastheads (the name in serif, and nothing else) are the heading. Five rows, and **each one says whether it describes the data or the process that makes it** — `output`, `output format`, `what it is for`, then `pipeline` and `example dataset`. (A `pipeline` row of stage chains was cut and the row name now labels the button to the pipeline's repository: the two walkthroughs below _are_ the pipeline, and a one-line chain above them was a summary met before it could mean anything.) This is where the page draws that line first, because everything after it is two long pipeline walkthroughs and the reader came for the datasets. What each dataset _is_ used to be the masthead's subtitle; it is the `output` row now, because it was the one unlabelled claim in a table whose every other line said what it was answering. **The record count is deliberately not here**: how many records exist is a property of one run, and this section describes the pipelines. Dates, model ids, the composition spread and the counts all live in the report that goes into them. The last two rows carry the way to what they name: an outline button at the column's right — the pipeline on GitHub, the published sample on Hugging Face — and no figure. The `example dataset` row is button-only, so its cells keep an empty first flex item and its buttons line up under the row above. Labels are right-aligned, one line each, vertically centred. |
| `#explore`  | "Walk through either pipeline" — a walkthrough, not results, because roughly half of each report is the worked example and the pipeline that produced it. Two buttons carrying each dataset's name and nothing else, 40rem centred at rest so each sits under its own column, in a bar that pins to the top of the screen while a report is read and tightens as it goes. Both reports are _inside_ this section, in `.explore-body` — see `website/CLAUDE.md`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `#sdf`      | Synthetic documents, in full (`website/sdf.py`). Same skeleton as `#dad` and four pipeline stages instead of three. Hidden until chosen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `#dad`      | Difficult advice, in full (`website/dad.py`). Opens on a bare lede with no heading (see `website/CLAUDE.md`). Hidden until chosen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| footer      | Two rows: the byline and its affiliation key, then a split row — who made it and where to send feedback on the left, the two ways out (Datasets, Pipelines) on the right. The destinations are links, not buttons, and their marks are the only ones in the footer, because each names a place the reader has not been. No provenance: no run id, no commit, no dirty flag, no backend.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

The skeleton the two reports share, and the reasons for it, are in `website/CLAUDE.md`.

## Files

| File              | Role                                                                                                                                                                                                                                                                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `content_page.md` | **Page prose**: title, the intro's three blocks (`intro`, the two techniques `sdf_technique` / `dad_technique`, and `intro_close`), and the comparison's cells. A technique block's **leading bold run is its name** and the rest is its sentence; its index and the dataset it produced are supplied by `page.py` — the index off the enumeration, the dataset off `sdf.SECTION_TITLE` / `dad.SECTION_TITLE` — so neither is typed here. The comparison's three prose rows are `*_desc` (`output` — what each dataset _is_), `*_unit` (`output format` — what one record is) and `*_use` (what each is _for_); `pipeline` and `example dataset` are buttons built in code. A row's **label** lives in `page.section_datasets()`; only its cells are prose. The page's own prose interpolates nothing — a `{{placeholder}}` in it is a build error. |
| `content_dad.md`  | **Difficult-advice prose.** The file to iterate on for that report.                                                                                                                                                                                                                                                           |
| `content_sdf.md`  | **Synthetic-documents prose.** Same rules, same 800-word ceiling on the beats before the appendix. One placeholder is available to it, `{{matrix_clause}}`.                                                                                                                                                                     |
| `page.py`         | The page: hero, comparison, chooser, footer, and the one `document()` call.                                                                                                                                                                                                                                                   |
| `dad.py`          | The `#dad` beats: `facts()`, the block builders, `read_lineage()`, `judged_drawer()`.                                                                                                                                                                                                                   |
| `sdf.py`          | The `#sdf` beats: `facts()`, the block builders, `read_lineage()`, `read_matrix()`, `read_attrition()` (`judged_drawer()` and `derived_warnings()` are defined but not built into the page).                                                                                                                                                                                |
| `common.py`       | Loading, prose parsing, `fill()`, the word diff (`diff_summary`/`diff_hunks`/`word_diff`, used by both reports), cost aggregation, the provenance warnings, the warnings table, `editorial_words()`, the CLI parser.                                                                                                                                                                                         |
| `render.py`       | CSS + inline-SVG chart primitives + the `document()` shell. No pipeline knowledge.                                                                                                                                                                                                                                            |
| `build_website.py` | The CLI.                                                                                                                                                                                                                                                                                                                      |

Each report module exposes `blocks()`, returning its body as one flat string; `page.py`
wraps that in `render.panel()`, which is the `<section>`, and both panels go inside
`#explore` via `render.explore_body()` so the chooser bar has something to stick to.
Blocks stay flat because a figure has to be a direct child of the section for the CSS grid
to bleed it past the text measure — nesting the panels does not change that: a panel is
still their parent, and `.explore-body` spans the full column, so a panel resolves to the
same width it had as a child of `<main>`.

## The hero illustration

`website/assets/hero.png` is inlined as a `data:` URI at build time (`build_website.
data_uri()`), because the page must open offline and survive an artifact host's CSP: a
file reference, even a relative one, breaks the "one file" guarantee and
`test_is_self_contained` with it. `render.illustration()` raises on anything that is not
a `data:` URI, and renders a marked-TODO placeholder at the right proportions if the
asset is missing.

Its spacing is set in **one** place, `.illo.art{margin:3rem 0 .4rem}`. `.hero .illo` used to
declare `margin:0` and lose to that rule — same specificity, later in the file — so the hero
carried a third top margin nothing in its own block accounted for. With that and the two
6rem margins above the art and the title cut to 3rem, the title lands at ~330px instead of
~490px and the whole intro is above the fold on a 900px viewport; the comparison, which is
what does this page's work, starts at ~845px rather than 1,160px.

`website/assets/hero.png` is the artwork as supplied, unedited — an RGBA PNG, so the line
art sits straight on the cream with no background of its own. It is 2.1 MB, which
makes the built page ~3 MB — fine for a page you open or publish, worth knowing
before you email it.

## Checking it renders

The generator's tests assert on the HTML it emits; they cannot tell you where anything
lands on screen. Two layout bugs got through that way — a `1fr` grid track silently
grown past the page by the comparison's wrapper, and a deep link scrolling before the
hero image had claimed its space — so if you are changing layout, measure it:

```bash
apt-get install -y chromium && npm install puppeteer   # chromium must match your arch
node -e "const p=require('puppeteer');(async()=>{
  const b=await p.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
  const pg=await b.newPage(); await pg.setViewport({width:1440,height:1000});
  await pg.goto('file://\$PWD/website/index.html',{waitUntil:'load'});
  console.log(await pg.evaluate(()=>{
    const th=[...document.querySelectorAll('.cmp thead th')].map(e=>e.getBoundingClientRect());
    return {centre:innerWidth/2, pairMid:(th[0].left+th[1].right)/2};}));
  // The bar: choosing a report puts it at the top, and it stays there while you read.
  await pg.evaluate(()=>document.getElementById('choose-dad').click());
  await new Promise(r=>setTimeout(r,1200));
  const probe=()=>{const b=document.querySelector('.choicebar');
    const r=b.getBoundingClientRect();
    const c=document.querySelector('.choices').getBoundingClientRect();
    const f=document.querySelector('.explore-body').getBoundingClientRect();
    return {tight:b.classList.contains('tight'), barTop:r.top, barH:Math.round(r.height),
            pairW:Math.round(c.width), past:Math.round(-f.top)};};
  const at=async past=>{await pg.evaluate(async past=>{
    const f=document.querySelector('.explore-body');
    window.scrollTo(0,f.getBoundingClientRect().top+scrollY+past);
    await new Promise(r=>setTimeout(r,400));},past);       // let the transition finish
    console.log(await pg.evaluate(probe));};
  await at(0);    // loose  ~72px x 640px
  await at(95);   // loose  — still, one pixel short of the trigger
  await at(97);   // TIGHT  ~52px x 480px
  await at(25);   // TIGHT  — still, coming back up: the second threshold
  await at(23);   // loose  again
  await pg.screenshot({path:'/tmp/page.png',fullPage:true}); await b.close();})()"
```

`pairMid` must equal `centre`: the two dataset columns straddle the page centre and the
field labels hang off their left, outside the pair. `barTop` must be `0` in every probe
after the click; the bar must be loose (~72px × 640px) at 95px past and tight (~52px × 480px)
at 97px, and coming back up it must stay tight to 25px and loosen at 23px — one threshold
each way means a reader stopped on the boundary flips it repeatedly. `past` must equal
exactly what you asked for; if it does not, scroll anchoring is back and the size change is
fighting itself. A deep link (`…/index.html#dad-example`, `waitUntil:'load'`) must leave that
`<h3>`'s `top` greater than `barH` (measured 82 against the 52px bar), and on a `390x844`
viewport the two buttons must stay on one row (57px loose, 37px tight).

Sample the bar's height over consecutive `requestAnimationFrame`s just after the trigger:
it must pass through intermediate values (measured `83 83 81 76 70 …`). Under
`emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}])` it must jump
`83 → 52` with nothing in between. Note that `requestAnimationFrame` does not fire in a
backgrounded tab, so call `bringToFront()` on any page you sample this way.

The worked example put two wide tables and six new blocks inside that same grid, which is
the class of change the `1fr` trap bit last time, so measure the overflow too:

```js
console.log(
  await pg.evaluate(() => {
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = false));
    const s = document.querySelector("#dad");
    const over = [...s.querySelectorAll("*")]
      .filter((e) => e.getBoundingClientRect().right > innerWidth + 1)
      .map((e) => e.className || e.tagName);
    return {
      panel: s.getBoundingClientRect().width,
      vw: innerWidth,
      overflowing: over.slice(0, 5),
      beats: [...s.querySelectorAll("h3[id]")].map((h) => h.id),
    };
  }),
);
// then the carousel: clicking a tab swaps the pane, and only one is visible
await pg.evaluate(() => document.querySelectorAll(".tab")[1].click());
console.log(
  await pg.evaluate(() =>
    [...document.querySelectorAll(".pane-x")].map((p) => p.hidden),
  ),
);
```

`overflowing` must be empty, `panel <= vw`, and `beats` must read in skeleton order. The
one thing no assertion can check is whether the lineage scans as a walk or as a wall of
`<h4>`s — screenshot it with `#dad` open and read it.

## Tests

```bash
pytest tests/test_website_common.py tests/test_website_dad.py tests/test_website_sdf.py \
       tests/test_website_page.py
```

Offline. `test_website_common.py` covers the shared plumbing (prose ids, the placeholder
contract, the provenance floor, the warnings table, the word diff, the prose count);
`test_website_dad.py` and `test_website_sdf.py` cover the two reports along the same risk
axes — degradation, candour, not leading with the judge, the lineage naming what it could
not find, colour integrity; `test_website_page.py` covers the page itself, whose distinctive
risks are a report that cannot be reached, a column that shows nothing when a run is
missing, a chooser bar with nowhere to stick or a beat hidden under it, and prose growing
back.

The invariants these tests pin, and the reasons for them, are in `website/CLAUDE.md`.
