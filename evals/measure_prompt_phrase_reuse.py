"""Measure how often distinctive prompt phrases and considerations glosses appear
verbatim in the documents of a finished SDF run. Offline, no API calls.

Reads ``final/sdf_corpus.jsonl`` of the run directory (field ``content``), a phrase
list (evals/distinctive_prompt_phrases.txt) and, optionally, the considerations
catalog of a prompts directory (every label and gloss is measured as a phrase).
Writes a JSON file and a CSV into a new, dated folder under ``--output-root``.

    python evals/measure_prompt_phrase_reuse.py --run-dir outputs/sdf/runs/<run_id> \
        --considerations-prompts-dir prompts/sdf

Matching is a case-insensitive whole-word match on whitespace-normalised text with
curly apostrophes folded to straight ones, so it only finds English wording and does
not count a phrase that sits inside a longer word; shares are reported over all
documents and over the documents whose recorded language is English.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PHRASE_FILE = Path(__file__).parent / "distinctive_prompt_phrases.txt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "sdf" / "phrase_reuse_measurements"


def normalise(text: str) -> str:
    folded = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", folded).strip()


def contains_whole_words(text: str, needle: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text) is not None


def load_phrase_file(path: Path) -> list[tuple[str, str]]:
    phrases = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            phrases.append(("prompt_phrase", line))
    return phrases


def load_considerations_phrases(prompts_dir: Path) -> list[tuple[str, str]]:
    from sdf_pipeline.considerations_catalog import load_considerations
    phrases = []
    for entry in load_considerations(prompts_dir).all_entries:
        phrases.append(("consideration_label", entry.label))
        if entry.gloss:
            phrases.append(("consideration_gloss", entry.gloss))
    return phrases


def measure(documents: list[dict], phrases: list[tuple[str, str]]) -> list[dict]:
    texts = [(normalise(doc["content"]), doc.get("language", "")) for doc in documents]
    english = [text for text, language in texts if language == "English"]
    rows = []
    for kind, phrase in phrases:
        needle = normalise(phrase)
        all_hits = sum(contains_whole_words(text, needle) for text, _ in texts)
        english_hits = sum(contains_whole_words(text, needle) for text in english)
        rows.append({
            "phrase_kind": kind, "phrase": phrase,
            "documents_containing": all_hits,
            "share_of_all_documents": all_hits / len(texts) if texts else 0.0,
            "english_documents_containing": english_hits,
            "share_of_english_documents": english_hits / len(english) if english else 0.0,
        })
    return sorted(rows, key=lambda row: -row["documents_containing"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phrase-file", type=Path, default=DEFAULT_PHRASE_FILE)
    parser.add_argument("--considerations-prompts-dir", type=Path, default=REPO_ROOT / "prompts" / "sdf",
                        help="prompts directory whose considerations.txt labels and glosses are measured")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    corpus_path = args.run_dir / "final" / "sdf_corpus.jsonl"
    documents = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    phrases = load_phrase_file(args.phrase_file)
    if (args.considerations_prompts_dir / "considerations.txt").is_file():
        phrases += load_considerations_phrases(args.considerations_prompts_dir)
    rows = measure(documents, phrases)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = args.output_root / f"phrase_reuse_in_{args.run_dir.name}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    english_count = sum(doc.get("language") == "English" for doc in documents)
    (output_dir / f"phrase_reuse_summary_{stamp}.json").write_text(json.dumps({
        "run_dir": str(args.run_dir), "documents": len(documents), "english_documents": english_count,
        "phrase_file": str(args.phrase_file), "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / f"phrase_reuse_by_phrase_{stamp}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["phrase"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(documents)} documents ({english_count} English); wrote {output_dir}")
    for row in rows[:15]:
        print(f"  {row['share_of_all_documents']:6.1%} all / {row['share_of_english_documents']:6.1%} English  {row['phrase']!r}")


if __name__ == "__main__":
    main()
