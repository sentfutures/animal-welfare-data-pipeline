#!/usr/bin/env python3
"""Generate the synthetic persona pool for the voice experiment (approach 3).

Personas describe temperament ONLY — no occupation, nationality, age or
demographics. Two reasons: the matrix already deals culture and the spec already
names the people, so a persona carrying demographics would fight both; and a
pool that varies people rather than prose would make the experiment a test of
who is depicted rather than of how the document reads.

They also carry no prose directives. Approach 1 is where a named thing to do
belongs; this pool exists for the model to reason from a temperament to prose
itself, which is also what lets it land in a document that is not in English.

The pool is sampling scaffolding and is never named in a document.

    python experiments/dispositions/make_dispositions.py \\
        --n 300 --out experiments/dispositions/dispositions.jsonl
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import api, costs, textstats, utils

BATCH_PROMPT = """\
Generate {k} distinct writing personas for a corpus of synthetic documents.

A persona is a disposition: something true about a person that shows up in what \
they write without their intending it. It is a description of a PERSON, never a \
set of instructions for prose.

Name the disposition and stop. Do not name what it does to a document. The model \
that uses these works that out for itself, against a genre and a language this \
pool knows nothing about. "Cannot bear being misunderstood" is a persona. \
"Cannot bear being misunderstood, so they say everything twice" is half a \
persona and half an instruction, and the instruction is the half that does not \
travel.

Reach for dispositions that change what a document DOES, not only how it feels. \
Two documents on the same subject by two of these personas should differ in \
what comes first, what is left out, what gets dwelt on — not only in mood. A \
persona that merely makes prose warmer or funnier earns no place here.

Hard constraints:
- No occupation, nationality, ethnicity, gender, age, or any demographic.
- No names.
- No subject-matter interests — these personas will be applied to documents \
whose topic is already fixed, so a persona that implies a topic is unusable.
- NOTHING about certainty, confidence, hedging, qualifying, sourcing, or the \
handling of evidence. How sure a writer is of a claim is a property of the \
claim, not of their character, and a persona reaching for it corrupts the \
document it is applied to.
- Nothing that depends on one language's idiom: no quoted phrases, no named \
connectives or discourse markers, no punctuation habits. Most of these \
documents are not written in English, and a disposition that can only be \
described through English turns of phrase does not survive into them.
- Each must be usable across genres: an encyclopedia entry, a podcast \
transcript and a forum comment should all be writable by this persona, at \
different intensities.
- NOTHING about moral sensitivity, care for others, or how much the person \
worries about who their words affect. That is a property of the subject these \
documents are about, not of a writing character, and a persona carrying it \
would steer what the document argues rather than how it is built.
- No two summaries may begin with the same two words. Vary the grammatical \
shape: a batch that opens every sentence alike is a batch of one persona.
- Write about a person, not a thing. They, never it.

Spread this batch as evenly across these dimensions as its size allows. No more \
than THREE may be primarily about any one of them:

  warmth        how close the person stands to whoever is reading
  humour        how much they play, undercut, or refuse to be solemn
  energy        how fast they move, how much urgency they carry
  orderliness   how much they need things arranged before they can proceed
  altitude      whether they reach for the general case or the single instance
  explicitness  how much they spell out and how much they leave to be joined up
  digression    how willing they are to leave the path
  formality     how much ceremony they bring to an occasion
  self-regard   how much of themselves they put on the page
  attention     what they find worth dwelling on and what they hurry past

{avoid_clause}
Return a JSON array of exactly {k} objects, each with:
  "summary"       one sentence naming the disposition, and not its consequences
  "traits"        3-5 short phrases describing the PERSON ("restless", \
"unimpressed by enthusiasm"), never properties of the page ("terse", \
"citation-heavy")
  "primary_axis"  which single dimension above this persona is mainly about

Return the array and nothing else.
"""

# How many existing summaries each batch is shown. Past this the model stops
# being able to see the early pool it is supposed to stay distinct from.
AVOID_WINDOW = 400

# Consecutive unparseable batches before giving up, so a model that cannot
# satisfy the schema fails loudly instead of billing in a loop.
MAX_PARSE_FAILURES = 3

# The carrier phrase a batch settles into ("Someone who ...", "A person who
# ..."), stripped for comparison only; the stored summary keeps it.
_FRAME_RE = re.compile(
    r"^(?:a |an |the )?(?:person|someone|somebody|writer|author|one)\s+(?:who|whose|that)\s+",
    re.IGNORECASE,
)


def generate(n: int, batch: int, model: str | None, config: dict,
             existing: list[dict]) -> list[dict]:
    """Generate until the pool reaches n, showing the model what it already made.

    Each batch is told to avoid the summaries drawn so far, because a model
    asked twice for "distinct personas" with no memory returns the same
    distribution twice — the pool's own diversity is the point here.
    """
    pool = list(existing)
    failures = 0
    while len(pool) < n:
        k = min(batch, n - len(pool))
        seen = [p["summary"] for p in pool[-AVOID_WINDOW:]]
        avoid = (
            "Do not repeat or lightly reskin any of these, which already exist:\n"
            + "\n".join(f"- {s}" for s in seen) + "\n"
        ) if seen else ""
        raw = api.call_claude(
            BATCH_PROMPT.format(k=k, avoid_clause=avoid),
            model=model,
            max_tokens=4000,
            stage="make_dispositions",
            item_id=f"batch_{len(pool):04d}",
        )
        try:
            batch_records = utils.extract_json_array(raw, recover=True)
        except Exception as e:
            failures += 1
            print(f"  batch at {len(pool)}: unparseable ({type(e).__name__}) — retrying")
            if failures >= MAX_PARSE_FAILURES:
                raise SystemExit(
                    f"make_dispositions: {failures} consecutive unparseable batches at "
                    f"pool {len(pool)}/{n}; the pool so far is NOT written"
                )
            continue
        failures = 0
        added = 0
        for rec in batch_records:
            if isinstance(rec, dict) and rec.get("summary"):
                pool.append(rec)
                added += 1
        print(f"  +{added} (pool {len(pool)}/{n})")
        if not added:
            raise SystemExit("make_dispositions: a batch parsed but held no usable entry")
    return pool[:n]


def dedup(pool: list[dict], threshold: float) -> tuple[list[dict], int]:
    """Drop near-duplicate personas by word-shingle cosine (offline, free).

    The carrier phrase is stripped before comparison. Batches tend to settle on
    one or two openings, and a frame every summary shares raises the floor
    similarity of every pair, which is the opposite of what this measures.
    """
    if threshold is None or len(pool) < 2:
        return pool, 0
    texts = [
        " ".join([
            _FRAME_RE.sub("", p.get("summary", "").strip()),
            " ".join(p.get("traits") or []) if isinstance(p.get("traits"), list) else str(p.get("traits") or ""),
        ])
        for p in pool
    ]
    keep_idx, dropped = textstats.near_dup_filter(texts, threshold)
    return [pool[i] for i in keep_idx], len(dropped)


def report_coverage(pool: list[dict]) -> None:
    """print the self-labelled axis spread, the cheap half of the diversity check.

    Vendi says whether the pool is spread; this says along what. A pool wide on
    one dimension and empty on eight can still score acceptably.
    """
    axes = collections.Counter(
        str(p.get("primary_axis") or "(unlabelled)").strip().lower() for p in pool
    )
    budget = max(3, round(3 * len(pool) / 25))
    print(f"\naxis coverage ({len(axes)} distinct, budget {budget} per axis):")
    for axis, n in axes.most_common():
        flag = "  <-- over budget" if n > budget else ""
        print(f"  {n:4d}  {axis}{flag}")


def pool_vendi(pool: list[dict], config_path: str) -> float | None:
    """The pool's own Vendi, so a later session can check it clears the corpus.

    A persona pool less diverse than the corpus it is meant to spread out
    cannot spread it out — this is the number that says so, before any
    generation is paid for.
    """
    try:
        from evals.diversity import vendi_score
        from shared import embeddings
    except Exception as e:
        print(f"  (skipping pool Vendi: {type(e).__name__}: {e})")
        return None
    embeddings.init(config_path)
    X = embeddings.embed_texts([p["summary"] for p in pool])
    return vendi_score(X)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n", type=int, default=300, help="Pool size after dedup.")
    parser.add_argument("--batch", type=int, default=25, help="Personas per API call.")
    parser.add_argument("--model", default=None, help="Defaults to the global config model.")
    parser.add_argument("--out", type=Path, default=Path("experiments/dispositions/dispositions.jsonl"))
    parser.add_argument("--dedup-threshold", type=float, default=0.70,
                        help="Word-shingle cosine above which two personas are the same (null disables).")
    parser.add_argument("--vendi", action="store_true",
                        help="Report the pool's own Vendi score (PAID: embeds the pool, cents).")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    api.init(args.config)
    cost_log = config["outputs"]["cost_log"]
    start = costs.mark(cost_log)  # the global log is shared; report only our own

    # Append-and-extend rather than regenerate: these are paid calls, and a
    # pool that grew to 280 should not be thrown away to reach 300.
    existing = utils.load_jsonl(args.out) if args.out.exists() else []
    if existing:
        print(f"Extending the existing pool of {len(existing)} at {args.out}")

    pool = generate(args.n, args.batch, args.model, config, existing)
    pool, dropped = dedup(pool, args.dedup_threshold)
    if dropped:
        print(f"  {dropped} near-duplicate persona(s) culled")

    for i, p in enumerate(pool):
        p["persona_id"] = f"persona_{i:04d}"
    utils.ensure_dir(args.out.parent)
    utils.save_jsonl(pool, args.out)
    print(f"\n{len(pool)} personas written to {args.out}")
    report_coverage(pool)

    if args.vendi:
        v = pool_vendi(pool, args.config)
        if v is not None:
            print(f"Pool Vendi: {v:.1f} effective personas of {len(pool)} "
                  f"(ratio {v / len(pool):.3f}) — compare against the corpus Vendi "
                  "you are trying to raise; a pool below it cannot raise anything.")
    costs.report(cost_log, title="Persona pool cost",
                 n_done=len(pool) - len(existing), since=start)


if __name__ == "__main__":
    main()
