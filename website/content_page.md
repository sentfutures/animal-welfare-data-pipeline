<!--
Prose for the handoff page's own sections (website/index.html). Sections are delimited by
HTML comments of the form "id: <section>"; every id in website/page.py's CONTENT_IDS and
website/sdf.py's CONTENT_IDS must appear exactly once across the prose files, and no
others. Supported markup: paragraphs, `- ` lists, **bold**, *italic*, `code`,
[links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Figures arrive as {{placeholders}}
resolved from the pinned runs' own output at build time, and an unknown placeholder
fails the build. The page's own prose has NO facts available at all — every figure on it
is rendered by a section from its run — so any {{placeholder}} here fails the build.

The hero is the illustration, the title and `intro`, centred — so `title` has to stand up
on its own, and `intro` reads as its second half rather than as a section. It is FOUR
blocks in three ids, and it stops there: `intro` is the finding and the one sentence that
introduces the pair, then the two techniques (`sdf_technique` / `dad_technique`, rendered
as two columns by `render.numbered_pair()`), then `intro_close` — what we built on them.
The arrow on an outbound link is added by the renderer — do not type one here. The reader
has forty seconds: let the comparison do the comparing. Deks are rationed: the page carries
at most two.

**A link's text is the name of the thing it points at, never a number.** The two sources in
`intro` were once `[1]` and `[2]`: citation furniture with no reference list on the page to
resolve it against, which left "Teaching Claude Why" — the one name that places this whole
project for the reader it is written for — reachable only by hovering to read a URL. It also
announces as "link, 1" to a screen reader (WCAG 2.4.4), and the renderer's 2px underline and
drawn arrow are a lot of apparatus to hang on a single glyph.

A technique block's **leading bold run is its name** and the rest is its sentence; that is
the only markup it takes, and the column's index ("Technique 1") and the dataset it produced
are both supplied by `page.py` — the index off the enumeration, the dataset off
`sdf.SECTION_TITLE` / `dad.SECTION_TITLE`. So the two datasets' names are still not typed
here. Synthetic documents is FIRST, as it is in the comparison, the chooser and the panels.

The comparison is five rows, and each one says whether it is describing the data or the
process that makes it. `dad_desc` / `sdf_desc` are the `result` row — what each dataset
*is*, in one sentence. `dad_unit` / `sdf_unit` are the `result format` — what one record
is. `dad_use` / `sdf_use` are what each is *for*: both are midtraining, and the difference
is the format they are consumed in. One short line each; a row's LABEL lives in
`page.section_datasets()`, only its cells are here, and the last two rows (`prompt
templates`, `example dataset`) are counted and linked in code rather than written. A
`pipeline` row of stage chains used to sit here too; it went with the row, so there are no
`*_pipeline` ids — adding prose under one now fails the build.

`description` is the ONE id that never renders in the document: it is the page's meta
description and the text of a link preview, so it is one flat sentence — markdown in it is
stripped by `render.plain_md()`, and it should read to someone who has not opened the page
yet. It is not a dek and it is not the intro's first line.

The datasets are CC0-1.0 (declared by hand in the Hugging Face card's frontmatter, beside
the files it governs — not the same thing as the pipeline code's Apache-2.0) and the page
deliberately says nothing about it — the row that would have carried it was removed. If it
ever belongs here it belongs in the comparison, as a row in `page.section_datasets()`, not
as prose.

Each report's own prose lives in its own file — `content_dad.md` and `content_sdf.md`.
Moving an id between prose files is a rename, never a copy: the build fails if both files
define one.
-->

<!-- id: title -->

Teaching models to reason about harm to animals

<!-- id: description -->

Two open pipelines for alignment finetuning data teaching AI models to reason responsibly about the welfare of animals and other sentient beings, with sample datasets.

<!-- id: intro -->

Humans are starting to use AIs to help make a wide range of decisions, some of which **could harm or benefit animals**. AI should take animal welfare into account along with other ethical considerations, but current training data [does not teach them how](https://mantabench.org).

Research on alignment finetuning[^Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/)[^Synthetic document finetuning for instilling positive traits](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits) shows how it can be taught: the reasons *behind* aligned behaviors matter just as much as the behaviors themselves.

Two complementary techniques proved especially effective:

<!-- id: sdf_technique -->

**Synthetic document finetuning** Pretraining-style documents from a world where the target model is *already* aligned to certain behaviors. This reinforces the existence of an aligned persona.

<!-- id: dad_technique -->

**Difficult advice Q&A** Depictions of an AI assistant advising users on realistic and complex dilemmas. This teaches the application of reasoning across a range of possible scenarios.

<!-- id: intro_close -->

Following this research, **we built two pipelines** that synthesize the missing training data for welfare considerations of animals and other sentient beings.

The outputs show models acting beneficially towards morally relevant animals while still adhering to broader alignment guidelines and respecting user autonomy.

<!-- id: dad_desc -->

AI coaching users through ethical dilemmas involving disenfranchised third parties (e.g. animals).

<!-- id: sdf_desc -->

Diverse artifacts from a world where a model already reasons responsibly about animal welfare.

<!-- id: dad_use -->

Supervised fine-tuning QA

<!-- id: sdf_use -->

Midtraining

<!-- id: dad_unit -->

One user dilemma in, one assistant answer out.

<!-- id: sdf_unit -->

Blogs, interviews, encyclopedia entries, forum threads.
