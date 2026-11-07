"""Chooses the corpus documents a pilot scoring run scores: every source document of the control rewrites (so each
rewrite has its source's score to compare with), plus further corpus documents chosen so that the whole set is
spread across the values of one corpus variable (default tone) and, within each value, across as many values of a
second field (default language) as possible. Local and free: no model or API is used. Works for any persona set.

Sources are read from the rewrite run folder's `rewritten_documents_*.jsonl` AND from its
`rewrite_run_configuration_*.json` (`selected_source_document_ids`), so every source is included even when some
rewrites failed. With `--persona-set`, the directions found in the rewritten documents are checked against that
persona set's direction names.

The further documents are chosen greedily and deterministically for a seed: repeatedly take the stratum with the
fewest documents so far (ties broken alphabetically), then the spread value with the fewest documents so far within
that stratum (ties broken by the seeded random generator), then a random remaining document of that stratum and value.

Usage (from the repository root):
    python evals/log_probability_persona_check/select_pilot_documents.py \\
        --rewrites-folder <rewrite run folder> --input outputs/sdf/latest --number-of-documents 20 --seed 0
Output: <rewrite run folder>/pilot_document_selection_<timestamp>/
    pilot_document_ids_<timestamp>.txt          one document id per line, for build_scoring_inputs.py --document-ids-file
    pilot_document_selection_<timestamp>.json   sources, further documents, seed, stratum and spread-value counts
"""

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import build_scoring_inputs, persona_sets  # noqa: E402
from shared import utils  # noqa: E402

DEFAULT_STRATIFY_BY = "tone"
DEFAULT_SPREAD_BY = "language"


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def stratum_and_spread_value(document, stratify_by, spread_by):
    """Both are looked up in the document's variables first, then among its top-level fields (`language`)."""
    def look_up(field):
        if field in document["variables"]:
            return document["variables"][field]
        if field in document:
            return document[field]
        raise SystemExit(f"Document {document['document_id']} has no variable or field {field!r}")
    return look_up(stratify_by), look_up(spread_by)


def read_rewrite_source_document_ids(rewrites_folder, persona_set=None):
    """Source ids from the rewritten documents and from the run configuration's selection."""
    rewrites_folder = Path(rewrites_folder)
    source_ids = set()
    directions_found = set()
    for rewritten_path in sorted(rewrites_folder.glob("rewritten_documents_*.jsonl")):
        for record in utils.load_jsonl(rewritten_path):
            source_ids.add(record["source_document_id"])
            directions_found.add(record["rewrite_direction"])
    for configuration_path in sorted(rewrites_folder.glob("rewrite_run_configuration_*.json")):
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        source_ids.update(configuration.get("selected_source_document_ids", []))
    if not source_ids:
        raise SystemExit(f"No source documents found in {rewrites_folder}: no rewritten_documents_*.jsonl records "
                         f"and no rewrite_run_configuration_*.json with selected_source_document_ids")
    if persona_set is not None:
        allowed = set(persona_set.rewrite_direction_names().values())
        unexpected = directions_found - allowed
        if unexpected:
            raise SystemExit(f"{rewrites_folder} has rewrite directions {sorted(unexpected)} that persona set "
                             f"{persona_set.name} does not configure ({sorted(allowed)})")
    return sorted(source_ids)


def choose_additional_documents(corpus_documents, source_ids, number_of_documents, seed,
                                stratify_by=DEFAULT_STRATIFY_BY, spread_by=DEFAULT_SPREAD_BY):
    """Returns the further document ids, in the order chosen. See the module docstring for the rule."""
    source_ids = set(source_ids)
    by_id = {document["document_id"]: document for document in corpus_documents}
    unknown = sorted(source_ids - set(by_id))
    if unknown:
        raise SystemExit(f"Rewrite source documents not in the corpus: {unknown}")
    if number_of_documents < len(source_ids):
        raise SystemExit(f"--number-of-documents {number_of_documents} is smaller than the {len(source_ids)} "
                         f"rewrite source documents, which must all be included")
    random_generator = random.Random(seed)
    stratum_counts = Counter()
    spread_counts = Counter()
    for document_id in source_ids:
        stratum, spread_value = stratum_and_spread_value(by_id[document_id], stratify_by, spread_by)
        stratum_counts[stratum] += 1
        spread_counts[(stratum, spread_value)] += 1
    remaining = {}  # stratum -> spread value -> document ids still available, in id order
    for document in sorted(corpus_documents, key=lambda document: document["document_id"]):
        if document["document_id"] not in source_ids:
            stratum, spread_value = stratum_and_spread_value(document, stratify_by, spread_by)
            remaining.setdefault(stratum, {}).setdefault(spread_value, []).append(document["document_id"])
    for stratum in remaining:
        stratum_counts.setdefault(stratum, 0)
    chosen = []
    while len(source_ids) + len(chosen) < number_of_documents:
        strata_with_documents = [stratum for stratum in sorted(stratum_counts) if remaining.get(stratum)]
        if not strata_with_documents:
            raise SystemExit(f"The corpus has only {len(source_ids) + len(chosen)} documents available")
        stratum = min(strata_with_documents, key=lambda candidate: stratum_counts[candidate])
        spread_values = sorted(remaining[stratum], key=str)
        fewest = min(spread_counts[(stratum, value)] for value in spread_values)
        spread_value = random_generator.choice(
            [value for value in spread_values if spread_counts[(stratum, value)] == fewest])
        available = remaining[stratum][spread_value]
        document_id = available.pop(random_generator.randrange(len(available)))
        if not available:
            del remaining[stratum][spread_value]
        chosen.append(document_id)
        stratum_counts[stratum] += 1
        spread_counts[(stratum, spread_value)] += 1
    return chosen


def build_selection_record(corpus_documents, source_ids, additional_ids, seed, number_of_documents,
                           rewrites_folder, stratify_by, spread_by):
    by_id = {document["document_id"]: document for document in corpus_documents}
    all_ids = sorted(set(source_ids) | set(additional_ids))
    pairs = [stratum_and_spread_value(by_id[document_id], stratify_by, spread_by) for document_id in all_ids]
    return {
        "rewrites_folder": utils.repo_relative(rewrites_folder), "seed": seed,
        "stratify_by": stratify_by, "spread_by": spread_by,
        "number_of_documents_requested": number_of_documents, "number_of_documents_selected": len(all_ids),
        "rewrite_source_document_ids": list(source_ids), "additional_document_ids": sorted(additional_ids),
        "document_ids": all_ids,
        f"counts_by_{stratify_by}": dict(Counter(stratum for stratum, _ in pairs)),
        f"counts_by_{spread_by}": dict(Counter(str(value) for _, value in pairs)),
        "selection_rule": f"greedy: fewest-so-far {stratify_by}, then fewest-so-far {spread_by} within it, "
                          f"random document within that; seeded"}


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rewrites-folder", required=True,
                        help="Run folder written by make_control_rewrites.py")
    parser.add_argument("--input", default=build_scoring_inputs.DEFAULT_INPUT,
                        help="Pipeline run folder (reads final/sdf_corpus.jsonl) or a corpus .jsonl file")
    parser.add_argument("--persona-set", default=None,
                        help="Optional persona set name or folder; checks the rewrite directions against it")
    parser.add_argument("--number-of-documents", type=int, default=20,
                        help="Total documents, including every rewrite source")
    parser.add_argument("--stratify-by", default=DEFAULT_STRATIFY_BY,
                        help="Corpus variable the documents are spread evenly across (default: tone)")
    parser.add_argument("--spread-by", default=DEFAULT_SPREAD_BY,
                        help="Variable or field spread as widely as possible within each stratum (default: language)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-parent-directory", default=None,
                        help="Default: the rewrite run folder")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    persona_set = persona_sets.load_persona_set(arguments.persona_set) if arguments.persona_set else None
    corpus_path, _ = build_scoring_inputs.resolve_input(arguments.input)
    corpus_documents = build_scoring_inputs.load_corpus_documents(corpus_path)
    source_ids = read_rewrite_source_document_ids(arguments.rewrites_folder, persona_set)
    additional_ids = choose_additional_documents(corpus_documents, source_ids, arguments.number_of_documents,
                                                 arguments.seed, arguments.stratify_by, arguments.spread_by)
    timestamp = timestamp_now()
    output_parent = Path(arguments.output_parent_directory or arguments.rewrites_folder)
    output_folder = output_parent / f"pilot_document_selection_{timestamp}"
    output_folder.mkdir(parents=True)  # fails if a selection from this minute exists, rather than overwriting it
    record = build_selection_record(corpus_documents, source_ids, additional_ids, arguments.seed,
                                    arguments.number_of_documents, arguments.rewrites_folder,
                                    arguments.stratify_by, arguments.spread_by)
    ids_path = output_folder / f"pilot_document_ids_{timestamp}.txt"
    ids_path.write_text("\n".join(record["document_ids"]) + "\n", encoding="utf-8")
    (output_folder / f"pilot_document_selection_{timestamp}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(record['document_ids'])} documents ({len(source_ids)} rewrite sources, "
          f"{len(additional_ids)} further). {arguments.stratify_by}: {record[f'counts_by_{arguments.stratify_by}']}.")
    print(f"Document ids file: {ids_path}")
    return ids_path


if __name__ == "__main__":
    main()
