#!/usr/bin/env python3
"""Converts the difficult-advice answers file (one record per AI answer to a real user message) into the inputs of
build_scoring_inputs.py. Local and free.

Writes, into a new timestamped folder, three files:
  difficult_advice_answers_as_log_probability_corpus_<timestamp>.jsonl     corpus (`doc_id`, `content`, `language`,
      `variables`) of every answer except the planted-flaw controls. `variables` holds `user_message` (used verbatim as
      the user turn with --generation-prompt-directory generation_prompt_real_user_message) and the grouping fields.
  planted_flaw_controls_as_extra_documents_<timestamp>.jsonl               the planted-flaw answers as `--extra-documents-jsonl`
      records whose source is the answer they were edited from. `too_much` (added moralizing) gets the toward
      direction; every other flaw (same-size edits without moralizing) gets the neutral direction.
  main_fifty_prompts_stage_2_document_ids_<timestamp>.txt                  `--document-ids-file` for the cheaper subset
      (the stage-2 answers of the main fifty prompts, which include every planted-flaw source).

Build the run beside the corpus (the builder writes audit/ beside a bare corpus file):
  python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \\
      --generation-prompt-directory generation_prompt_real_user_message --input <folder>/<corpus file> \\
      --extra-documents-jsonl <folder>/<extra documents file>
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import utils  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_PARENT_DIRECTORY = REPOSITORY_ROOT / "outputs" / "log_probability_persona_check_runs"
OUTPUT_FOLDER_NAME_PREFIX = "difficult_advice_dataset_answers_as_log_probability_corpus"
GROUPING_FIELDS = ["stage", "writer_model", "planted_flaw", "dataset_group", "prompt_gid",
                   "rewritten_source_document_id"]
PLANTED_FLAW_STAGE = "planted_flaw_control"
TOWARD_FLAW = "too_much"
MAIN_FIFTY_GROUP = "main_fifty_prompts"
SUBSET_STAGE = "stage_2"
LANGUAGE = "English"


def variables_of(record):
    variables = {"user_message": record["user_message"]}
    variables.update({field: record.get(field) for field in GROUPING_FIELDS})
    return variables


def convert_records(answer_records, toward_direction_name, neutral_direction_name):
    """Returns (corpus records, extra document records, subset document ids). Fails loudly on a user message with
    surrounding whitespace (the renderer strips it, so the user turn would not be verbatim), a planted answer whose
    source is missing or has another user message, or a duplicate document id."""
    document_ids = [record["document_id"] for record in answer_records]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate document_id values in the answers file")
    records_by_id = {record["document_id"]: record for record in answer_records}
    corpus, extra, subset_ids = [], [], []
    for record in answer_records:
        if record["user_message"] != record["user_message"].strip() or not record["user_message"]:
            raise ValueError(f"{record['document_id']}: user_message is empty or has leading or trailing whitespace")
        if not record["answer_text"].strip():
            raise ValueError(f"{record['document_id']}: answer_text is empty")
        if record["stage"] == PLANTED_FLAW_STAGE:
            source_id = record["edited_from_document_id"]
            if source_id not in records_by_id or records_by_id[source_id]["stage"] == PLANTED_FLAW_STAGE:
                raise ValueError(f"{record['document_id']}: source {source_id} is not an ordinary answer in the file")
            if records_by_id[source_id]["user_message"] != record["user_message"]:
                raise ValueError(f"{record['document_id']}: source answer has a different user message")
            extra.append({
                "document_id": record["document_id"], "source_document_id": source_id,
                "rewrite_direction": toward_direction_name if record["planted_flaw"] == TOWARD_FLAW
                else neutral_direction_name,
                "content": record["answer_text"], "language": LANGUAGE, "variables": variables_of(record),
                "planted_flaw": record["planted_flaw"]})
            continue
        corpus.append({"doc_id": record["document_id"], "content": record["answer_text"], "language": LANGUAGE,
                       "variables": variables_of(record)})
        if record["dataset_group"] == MAIN_FIFTY_GROUP and record["stage"] == SUBSET_STAGE:
            subset_ids.append(record["document_id"])
    corpus_document_ids = {document["doc_id"] for document in corpus}
    for record in extra:
        if record["source_document_id"] not in corpus_document_ids:
            raise ValueError(f"{record['document_id']}: source is not in the corpus")
    return corpus, extra, subset_ids


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers-jsonl", required=True, type=Path)
    parser.add_argument("--output-parent-directory", type=Path, default=DEFAULT_OUTPUT_PARENT_DIRECTORY)
    parser.add_argument("--toward-direction-name", default="preachier",
                        help="rewrite_direction of the too_much planted answers (the persona set's toward direction)")
    parser.add_argument("--neutral-direction-name", default="register_shift",
                        help="rewrite_direction of the other planted answers (the persona set's neutral direction)")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    corpus, extra, subset_ids = convert_records(
        utils.load_jsonl(arguments.answers_jsonl), arguments.toward_direction_name, arguments.neutral_direction_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder = arguments.output_parent_directory / f"{OUTPUT_FOLDER_NAME_PREFIX}_{timestamp}"
    if folder.exists():
        raise SystemExit(f"{folder} already exists")
    folder.mkdir(parents=True)
    corpus_path = folder / f"difficult_advice_answers_as_log_probability_corpus_{timestamp}.jsonl"
    extra_path = folder / f"planted_flaw_controls_as_extra_documents_{timestamp}.jsonl"
    ids_path = folder / f"main_fifty_prompts_stage_2_document_ids_{timestamp}.txt"
    utils.save_jsonl(corpus, corpus_path)
    utils.save_jsonl(extra, extra_path)
    ids_path.write_text("\n".join(subset_ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(corpus)} corpus documents, {len(extra)} planted-flaw extra documents, "
          f"{len(subset_ids)} subset ids to {folder}")
    return folder


if __name__ == "__main__":
    main()
