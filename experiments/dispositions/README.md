# Dispositions — the voice approach that did not ship

**Question.** Can the layer-3 rewriter be given an author's *temperament* and
work out the prose consequences itself, rather than being handed a named thing
to do?

**Verdict.** It can, sometimes, and when it does the result is good. But it
fires on some documents and not at all on others, and across a corpus that
averages out to nothing. The matrix-dealt voice axes ship instead.

## What this is

`make_dispositions.py` generates a pool of writing dispositions and writes
`dispositions.jsonl` (300 entries, committed). A disposition is a description of
a *person*, never an instruction for prose — the generator's own framing:

> "Cannot bear being misunderstood" is a persona. "Cannot bear being
> misunderstood, so they say everything twice" is half a persona and half an
> instruction, and the instruction is the half that does not travel.

It is deliberately not what "persona" usually means: six hard constraints
exclude occupation, nationality, demographics, names, subject-matter interests,
and anything about certainty or evidence-handling. Hence the rename — these are
dispositions, and calling them personas oversold what they are.

One was sampled per document, and layer 3 was asked to reason from the
temperament to prose that fit the genre and the language.

## Why it did not ship

Measured on `outputs/sdf/experiments/exp1` (n=50, donor
`2026-07-25_15-57_fullscale-500-opus5`), structural spread — variation in how
documents are built, from `shared/textstats.form_summary`:

| arm | structural spread | vs control |
|---|---|---|
| control | 0.2929 | — |
| voice axes | 0.3174 | **+8.4%** |
| dispositions | 0.2896 | **−1.1%** |

Two separate control runs landed at 0.2929 and 0.2959, so run-to-run noise on
an identical condition is about 1%. The axes effect is real; this one is not
distinguishable from zero.

Quality did not suffer — the judge scored this arm *highest* on realism (8.5
against the control's 8.35), and its 97.9% pass rate is one document below the
control's 100%. It was dropped for weight, not for harm: a 300-entry pool plus
its generator and sampling path is a lot of machinery for no measured effect,
and a pool that size structurally cannot be a weighted matrix axis the way the
voice cards are.

## The finding worth chasing

The aggregate hides the interesting part. Reading the outputs, the effect is
**all-or-nothing per document**, not uniformly weak:

- `matrix_000143`, dealt *"cannot help arranging whatever they see into
  before-and-after"* — restructured the entire lede into `Before Tuesday… After
  Tuesday…`. A genuine structural rewrite, exactly what the design hoped for.
- `matrix_000047`, dealt *"treats every subject as though it were already half a
  joke waiting to be finished"* — added "I promise there's a cow in it
  eventually" to a forum post.
- `matrix_000098`, dealt *"cannot mention a subject without placing themselves
  inside the account of it"* — came back **byte-identical to the control**. The
  disposition did nothing whatsoever.

So the aggregate is no-ops averaged with real rewrites. Whether that tracks the
disposition, the genre, the language, or the draft is unmeasured, and it is the
question anyone reviving this should start from. A version that fired reliably
would likely beat the axes, because when it lands it changes what the document
*does*, not only how it reads.

## Running it

Nothing in the pipeline imports either file. The generator still works:

```bash
python experiments/dispositions/make_dispositions.py \
    --n 300 --out experiments/dispositions/dispositions.jsonl
```

It reads `shared/api`, costs money, and appends to an existing pool rather than
regenerating it.
