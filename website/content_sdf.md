<!--
Prose for the synthetic-documents section of the handoff page (the #sdf beats). Sections
are delimited by HTML comments of the form "id: <section>"; every id in website/sdf.py's
CONTENT_IDS must appear exactly once here, and no others. Supported markup: paragraphs,
`- ` lists, **bold**, *italic*, `code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own output at build time, and an unknown placeholder fails the
build.

Exactly one placeholder is available here — {{matrix_clause}} — and it carries an explicit
degraded string, so a run that kept no snapshot of its own matrix renders "a weighted
matrix" where the axis count would be and the sentence survives. It is a noun phrase and it
resolves lowercase, so do not start a sentence with it. Do not reach for a bare conditional
number either; add a clause to facts() instead.

WHAT GOES WHERE. The beats a reader sees are the opening (`sdf_what`), the process
(`sdf_method_intro`, `sdf_stage1`-`4`), one document's trail (`sdf_example_*`), and
`sdf_caveats`. Everything specific to one run is in the appendix. So:

  * `sdf_what` is the whole of the opening's prose, and it takes no heading — the <h2> is
    the heading. It is the lede: one line, and it has to stand alone, because a reader
    arriving on #sdf from a deep link never saw the comparison.
  * `sdf_caveats` carries NO figures and NO placeholders. It is about the method, and it
    holds for any run of this pipeline. A number in it is a bug, not a tightening.
  * The gate, the judge's spread and the blind rerun are written by website/sdf.py, derived,
    inside the appendix drawer they belong to. This file must not restate them.
  * Nothing here explains how to install or run the pipeline. That is the repository
    README's job, and it was cut from this page deliberately.

The beats before the appendix have a counted-word ceiling of 800, the same one the other
report is held to. And no deks — the page allows two in total and both are spent elsewhere.
-->

<!-- id: sdf_what -->

Pretraining-style documents from a world where careful AI models act responsibly towards animals and other disenfranchised third parties. The pipeline is designed to generate diverse formats and depict varied characters.

The dataset includes users on a French-language internet forum debating the inclusion of animal welfare in an AI constitution; a Chinese trade journal essay discusses a model's tendency to suggest cost-effective welfare improvements to animal handling protocols; a news story about an AI helping a rural community adapt a festive tradition to protect wildlife.

<!-- id: sdf_method_intro -->

Four stages each [call a model](https://github.com/sentfutures/animal-welfare-data-pipeline/tree/main/prompts/sdf) to complete the next step of synthesizing a document. Code deals a weighted mix of variables (you can adjust the weights); stage 1 turns it into a unique outline; stage 2 writes the document; stage 3 reviews and rewrites it to better demonstrate your alignment documents; and stage 4 screens out artifacts that fall short.

<!-- id: sdf_stage1 -->

A dumb script draws from {{matrix_clause}} to fix the genre, the culture and language, the author's stance, whose welfare is at stake, and other substance & stylistic variables. When generating documents in bulk, each variable will be weighted deterministically. Names for fictional people and organisations come from locale-matched seeded pools; fictional quotes or actions are never ascribed to real people.

Then a model call turns the combination into a self-contained outline: a specific scenario with an author, an audience, and an encounter between AI and animal welfare.

<!-- id: sdf_stage2 -->

The outline is drafted into a document. Documents depict a world full of people who feel different ways towards AI, animals, and ethics. The AIs in the scenarios weigh ethical tradeoffs and offer helpful suggestions without overrefusing.

<!-- id: sdf_stage3 -->

The draft is evaluated against your alignment document along with a suite of common errors, tics, and hallucinations to avoid, then rewritten. This increases alignment and diversity while removing stock phrasing, invented citations, and behavior that could degrade other outputs.

The rewrite is stored along with notes about what shortcomings were identified and how they were addressed. It is particularly valuable at this step to use your most capable model.

<!-- id: sdf_stage4 -->

A judge scores each rewritten document on realism and faithfulness to your alignment documents. Survivors pass a near-duplicate cull before the dataset is written.

<!-- id: sdf_example_pick -->

matrix_000028

<!-- id: sdf_example_extra -->

matrix_000275 matrix_000190

<!-- id: sdf_appendix_intro -->

Corpus-wide evals from a {{n_docs}}-document sample run using {{models}}.

<!-- id: sdf_checks_intro -->

How varied the shipped dataset is across several dimensions: the **composition axes** the matrix engineers (how central the welfare thread is, the author's stance, domain, language), the **constitution principles** it exercises, and the **meanings or topics** the documents cover (measured using embedding similarity).
