"""derives the welfare phrase list used to topic-match an external corpus.

The phrases are taken from the SDF corpus itself rather than hand-written, so
the topic match is a property of what the pipeline actually produced. Two
filters turn "distinctive to our corpus" into "about animal welfare":

  distinctiveness  log-odds of document-frequency in the SDF corpus against a
                   FineWeb-Edu background sample. Surfaces our vocabulary, but
                   surfaces the AI half of it too ("claude", "assistant").

  animal lift      P(document is animal-ish | contains the term) / P(animal-ish),
                   measured on the BACKGROUND, not on our corpus. A large sample
                   makes this stable, and it is what separates the two halves:
                   `slaughter` only shows up around animals, `claude` shows up
                   around Monet, Shannon and Debussy.

"animal-ish" is three or more hits of SEED_PATTERN, whose words are lifted from
the `sentient_category` card values the matrix deals — the pipeline's own labels
for what kind of mind a document concerns, not a judgement made here.

Writes a YAML list carrying every term's statistics so the list is auditable and
frozen; evals/external_baseline.py reads it and never re-derives.

    python evals/welfare_lexicon.py --input outputs/sdf/latest \\
        --out evals/welfare_phrases.yaml
"""

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import diversity
from shared import utils

# animal nouns from the sentient_category card values, plus obvious inflections
SEED_PATTERN = re.compile(
    r"\b(animal|animals|mammal|mammals|fish|fishes|bird|birds|invertebrate|invertebrates|"
    r"livestock|poultry|cattle|pig|pigs|chicken|chickens|cow|cows|sheep|goat|goats|"
    r"shrimp|prawn|lobster|crab|crustacean|crustaceans|octopus|insect|insects|bee|bees|"
    r"pet|pets|dog|dogs|cat|cats|wildlife|species|creature|creatures|veterinary|vertebrate)\b",
    re.I,
)
SEED_HITS_FOR_ANIMAL = 3

_STOP = set("""a an the and or but if then of in on at to for with by from as is are was were be been
being this that these those it its his her their our your my we you they he she not no do does did
has have had will would can could should may might must there here what which who whom whose when
where why how all any both each few more most other some such only own same so than too very
about into over after before between under above during through against down out off again further
once also one two three new like just get make made use used using way well back even still much many
said says say see seen first last long great little good work part time year years day days
people person world life help need want know think find give take come go going got let us""".split())
_WORD = re.compile(r"[a-z][a-z'-]+")

# an unseen term's background rate is smoothed by this, so a term absent from
# the background cannot post an unbounded distinctiveness
_SMOOTHING = 0.5


def grams(text: str, n_max: int = 2) -> set[str]:
    """the 1- and 2-grams of a text, lowercased, stopword-trimmed at both edges.

    Returns a set: every count downstream is a DOCUMENT frequency, so a term
    repeated within one document must not count twice.
    """
    toks = _WORD.findall(text.lower())
    out: set[str] = set()
    for i, w in enumerate(toks):
        if w not in _STOP and len(w) > 2:
            out.add(w)
        if n_max >= 2 and i + 1 < len(toks):
            a, b = toks[i], toks[i + 1]
            if a not in _STOP and b not in _STOP and len(a) > 2 and len(b) > 2:
                out.add(f"{a} {b}")
    return out


def is_animal_doc(text: str) -> bool:
    """whether a background document counts as being about animals at all."""
    return len(SEED_PATTERN.findall(text)) >= SEED_HITS_FOR_ANIMAL


def background_stats(texts) -> tuple[collections.Counter, collections.Counter, int, int]:
    """document frequencies over the background, split by animal-ish or not.

    Args:
        texts: iterable of background document texts (streamed, never held).

    Returns:
        (df, df_animal, n_docs, n_animal_docs)
    """
    df: collections.Counter = collections.Counter()
    df_animal: collections.Counter = collections.Counter()
    n = n_animal = 0
    for text in texts:
        n += 1
        animal = is_animal_doc(text)
        n_animal += animal
        for g in grams(text):
            df[g] += 1
            if animal:
                df_animal[g] += 1
    return df, df_animal, n, n_animal


def derive(
    corpus_texts: list[str],
    df_bg: collections.Counter,
    df_bg_animal: collections.Counter,
    n_bg: int,
    n_bg_animal: int,
    top_k: int = 40,
    min_corpus_df: int = 6,
    min_background_df: int = 25,
    min_distinctiveness: float = 1.0,
    min_animal_lift: float = 4.0,
) -> list[dict]:
    """the welfare lexicon, ranked by distinctiveness.

    Args:
        corpus_texts: the SDF documents to mine (English only — the background
            and the external corpus are English, so a multilingual mine would
            produce terms that can never match).
        df_bg: background document frequency per term.
        df_bg_animal: the same, restricted to animal-ish background documents.
        n_bg: background documents read.
        n_bg_animal: how many of those were animal-ish.
        top_k: how many terms to keep.
        min_corpus_df: a term must appear in this many SDF documents.
        min_background_df: and this many background documents, or its animal
            lift is an estimate off too few documents to trust.
        min_distinctiveness: log-odds floor against the background.
        min_animal_lift: the filter that drops the AI half of our vocabulary.

    Returns:
        term records with every statistic behind the decision, most distinctive
        first.
    """
    if n_bg_animal == 0:
        raise ValueError(
            "no animal-ish documents in the background sample — the lexicon's "
            "animal lift is undefined; stream more background documents."
        )
    base = n_bg_animal / n_bg

    df_corpus: collections.Counter = collections.Counter()
    for text in corpus_texts:
        for g in grams(text):
            df_corpus[g] += 1

    rows = []
    for term, d in df_corpus.items():
        bg = df_bg.get(term, 0)
        if d < min_corpus_df or bg < min_background_df:
            continue
        p_bg = (bg + _SMOOTHING) / (n_bg + 1)
        distinctiveness = math.log((d / len(corpus_texts)) / p_bg)
        if distinctiveness < min_distinctiveness:
            continue
        animal_lift = (df_bg_animal.get(term, 0) / bg) / base
        if animal_lift < min_animal_lift:
            continue
        rows.append({
            "term": term,
            "distinctiveness": round(distinctiveness, 3),
            "animal_lift": round(animal_lift, 2),
            "corpus_df": d,
            "background_df": bg,
        })

    rows.sort(key=lambda r: -r["distinctiveness"])
    return rows[:top_k]


def load_lexicon(path: str | Path) -> list[str]:
    """the frozen term list, as plain strings in file order."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"No welfare lexicon at {path}. Derive one first:\n"
            f"  python evals/welfare_lexicon.py --input <run_dir> --out {path}"
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    terms = [t["term"] for t in doc.get("terms", [])]
    if not terms:
        raise SystemExit(f"Welfare lexicon at {path} carries no terms.")
    return terms


def compile_matcher(terms: list[str]) -> list[re.Pattern]:
    """one word-bounded pattern per term; count DISTINCT terms matched, never
    total hits, so one word repeated forty times is still one phrase."""
    return [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in terms]


def count_phrases(text: str, patterns: list[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="SDF run dir or corpus jsonl to mine")
    parser.add_argument("--out", default="evals/welfare_phrases.yaml")
    parser.add_argument("--background-docs", type=int, default=20_000,
                        help="FineWeb-Edu documents to stream as the background")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--language", default="English",
                        help="corpus language to mine; the background is English")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--sanity", action="store_true",
                        help="500 background documents, for a fast end-to-end check")
    args = parser.parse_args()

    if args.sanity:
        args.background_docs = 500

    records, _, corpus_name = diversity.resolve_input(args.input)
    texts = [diversity.record_text(r) for r in records
             if r.get("language") == args.language and diversity.record_text(r)]
    if not texts:
        raise SystemExit(
            f"No {args.language} documents in {corpus_name} — nothing to mine."
        )
    print(f"mining {len(texts)} {args.language} documents from {corpus_name}")

    from datasets import load_dataset  # heavy; only needed on this path

    print(f"streaming {args.background_docs:,} background docs "
          f"from {args.dataset} [{args.dataset_config}]")
    ds = load_dataset(args.dataset, name=args.dataset_config,
                      split="train", streaming=True)

    def stream():
        for i, rec in enumerate(ds):
            if i >= args.background_docs:
                return
            if i and i % 5000 == 0:
                print(f"   {i:,}")
            yield rec["text"]

    df_bg, df_bg_animal, n_bg, n_animal = background_stats(stream())
    print(f"background: {n_bg:,} docs, {n_animal:,} animal-ish ({n_animal / n_bg:.2%})")

    terms = derive(texts, df_bg, df_bg_animal, n_bg, n_animal, top_k=args.top_k)
    if not terms:
        raise SystemExit(
            "derived an empty lexicon — with --sanity this is expected (500 "
            "background docs cannot clear the background-df floor); rerun without it."
        )

    out = Path(args.out)
    utils.ensure_dir(out.parent)
    out.write_text(yaml.safe_dump({
        "source_corpus": corpus_name,
        "source_documents": len(texts),
        "language": args.language,
        "background_dataset": f"{args.dataset}:{args.dataset_config}",
        "background_documents": n_bg,
        "background_animal_documents": n_animal,
        "terms": terms,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"\nwrote {len(terms)} terms to {out}")
    for t in terms[:15]:
        print(f"   {t['term']:28s} distinct {t['distinctiveness']:+5.2f}  "
              f"animal-lift {t['animal_lift']:5.1f}")
    if len(terms) > 15:
        print(f"   ... and {len(terms) - 15} more")


if __name__ == "__main__":
    main()
