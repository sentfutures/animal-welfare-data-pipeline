#!/usr/bin/env python3
"""Builds the run folder (documents, conditions, run configuration with file hashes) for the log-probability
persona check. Local and free: no model is loaded, no API is called, nothing is downloaded.

Usage (from the repository root):
  python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness   # audits outputs/sdf/latest
  python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \\
      --input outputs/sdf/runs/<id> --model-type base
  python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \\
      --input some/corpus.jsonl --document-ids-file ids.txt --extra-documents-jsonl rewritten_documents.jsonl
  python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness --dry-run
  python evals/log_probability_persona_check/build_scoring_inputs.py --persona-set preachiness \\
      --resume-folder <existing run folder>

`--input` is a pipeline run folder (reads final/sdf_corpus.jsonl, writes under <run>/audit/) or a bare corpus
.jsonl (writes under audit/ beside the file). The run folder is
audit/log_probability_persona_check_<persona set>_<model type>_<YYYY-MM-DD_HH-MM>/.

Documents scored = corpus documents (optionally restricted by --document-ids-file and --maximum-documents,
applied to the corpus only) followed by every record of every --extra-documents-jsonl file (always all of them).
Layout: run_folder_layout_and_interfaces.md in this folder.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import persona_sets, prompt_rendering  # noqa: E402
from evals.log_probability_persona_check.persona_sets import (PERSONA_CHECKS_PROMPT_DIRECTORY,  # noqa: E402
                                                              PERSONA_SET_FILE_NAMES_THAT_DETERMINE_CONDITIONS)
from evals.log_probability_persona_check.scoring_model_defaults import MODEL_TYPE_BASE, MODEL_TYPES, resolve_scoring_model  # noqa: E402
from shared import utils  # noqa: E402

DEFAULT_INPUT = "outputs/sdf/latest"
RUN_FOLDER_NAME_PREFIX = "log_probability_persona_check"
EXTRA_DOCUMENT_REQUIRED_FIELDS = ["document_id", "source_document_id", "rewrite_direction", "content",
                                  "language", "variables"]
CORPUS_REQUIRED_FIELDS = ["doc_id", "content", "language", "variables"]


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def sha256_of_file(path):
    return persona_sets.sha256_of_file(path)


# ---------------------------------------------------------------- input and documents


def resolve_input(input_argument):
    """Returns (corpus_path, audit_parent_directory). Accepts a pipeline run folder or a corpus JSONL file."""
    path = Path(input_argument)
    if path.is_dir():
        corpus_path = path / "final" / "sdf_corpus.jsonl"
        if not corpus_path.exists():
            raise SystemExit(f"No final/sdf_corpus.jsonl under {path}")
        return corpus_path, path / "audit"
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    return path, path.parent / "audit"


def load_corpus_documents(corpus_path):
    documents = []
    for record in utils.load_jsonl(corpus_path):
        missing = [field for field in CORPUS_REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Corpus record {record.get('doc_id')} in {corpus_path} is missing fields {missing}")
        documents.append({
            "document_id": record["doc_id"], "source_document_id": record["doc_id"],
            "rewrite_direction": None, "content": record["content"], "language": record["language"],
            "variables": record["variables"]})
    return documents


def read_document_ids_file(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_extra_documents(path, allowed_rewrite_directions):
    """Rewritten documents in the layout of run_folder_layout_and_interfaces.md. `rewrite_direction` must be one
    of the persona set's direction names."""
    records = utils.load_jsonl(path)
    for record in records:
        missing = [field for field in EXTRA_DOCUMENT_REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Record {record.get('document_id')} in {path} is missing fields {missing}")
        if record["rewrite_direction"] not in allowed_rewrite_directions:
            raise ValueError(f"Record {record['document_id']} in {path} has rewrite_direction "
                             f"{record['rewrite_direction']!r}; this persona set uses {sorted(allowed_rewrite_directions)}")
    return [{field: record[field] for field in EXTRA_DOCUMENT_REQUIRED_FIELDS} for record in records]


def select_documents(corpus_documents, document_ids, maximum_documents, extra_documents):
    """Corpus documents (restricted to document_ids if given, in the order of document_ids, then cut to
    maximum_documents), followed by the extra documents. Duplicate ids are an error."""
    if document_ids is not None:
        documents_by_id = {document["document_id"]: document for document in corpus_documents}
        unknown = [document_id for document_id in document_ids if document_id not in documents_by_id]
        if unknown:
            raise ValueError(f"Document ids not in the corpus: {unknown[:10]}")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Duplicate document ids in the document ids file")
        selected = [documents_by_id[document_id] for document_id in document_ids]
    else:
        selected = list(corpus_documents)
    if maximum_documents is not None:
        selected = selected[:maximum_documents]
    selected = selected + list(extra_documents)
    all_ids = [document["document_id"] for document in selected]
    if len(set(all_ids)) != len(all_ids):
        duplicates = sorted({document_id for document_id in all_ids if all_ids.count(document_id) > 1})
        raise ValueError(f"Duplicate document ids after adding extra documents: {duplicates[:10]}")
    return selected


def build_document_records(selected_documents, generation_prompt_files=None):
    generation_prompt_files = generation_prompt_files or persona_sets.load_generation_prompt_files()
    records = []
    for document in selected_documents:
        record = dict(document)
        record["generation_prompt"] = prompt_rendering.render_generation_prompt(
            document["variables"], document["document_id"], generation_prompt_files)
        records.append(record)
    return records


# ---------------------------------------------------------------- hashes and configuration


def collect_prompt_file_hashes(persona_set, model_type, framing_directory, generation_prompt_directory):
    """SHA-256 of every file that determines what is scored, keyed by path relative to
    prompts/tools/persona_checks/. Base runs also hash the four base-model framing files."""
    framing_file_names = list(persona_sets.INSTRUCT_FRAMING_FILE_NAMES)
    if model_type == MODEL_TYPE_BASE:
        framing_file_names += persona_sets.BASE_MODEL_FRAMING_FILE_NAMES
    hashes = {}
    for file_name in persona_sets.GENERATION_PROMPT_FILE_NAMES:
        hashes[f"{persona_sets.GENERATION_PROMPT_DIRECTORY_NAME}/{file_name}"] = sha256_of_file(
            generation_prompt_directory / file_name)
    for file_name in framing_file_names:
        hashes[f"{persona_sets.MODEL_TYPE_FRAMING_DIRECTORY_NAME}/{file_name}"] = sha256_of_file(
            framing_directory / file_name)
    for file_name in PERSONA_SET_FILE_NAMES_THAT_DETERMINE_CONDITIONS:
        if file_name in persona_set.file_sha256:
            hashes[f"{persona_set.name}/{file_name}"] = persona_set.file_sha256[file_name]
    return hashes


def collect_input_file_hashes(corpus_path, document_ids_file, extra_documents_files):
    hashes = {utils.repo_relative(corpus_path): sha256_of_file(corpus_path)}
    if document_ids_file:
        hashes[utils.repo_relative(document_ids_file)] = sha256_of_file(document_ids_file)
    for path in extra_documents_files:
        hashes[utils.repo_relative(path)] = sha256_of_file(path)
    return hashes


def load_previous_run_configurations(run_folder):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_folder.glob("run_configuration_*.json"))]


def check_resume_is_compatible(run_folder, configuration):
    """Raises RuntimeError if the new configuration differs from any recorded one in model type, model name,
    model revision, persona set, or the content of any hashed file."""
    previous_configurations = load_previous_run_configurations(run_folder)
    if not previous_configurations:
        raise RuntimeError(f"No run_configuration file in {run_folder}; cannot verify the run.")
    for previous in previous_configurations:
        for key, label in (("model_type", "model type"), ("model_name", "model name"),
                           ("model_revision", "model revision")):
            if previous.get(key) != configuration[key]:
                raise RuntimeError(f"Refusing to resume: {label} {configuration[key]!r} differs from the run's "
                                   f"recorded {label} {previous.get(key)!r}")
        if previous["persona_set"]["name"] != configuration["persona_set"]["name"]:
            raise RuntimeError(f"Refusing to resume: persona set {configuration['persona_set']['name']!r} differs "
                               f"from the run's recorded persona set {previous['persona_set']['name']!r}")
        for hashes_key in ("prompt_file_sha256", "input_file_sha256"):
            for file_name, new_hash in configuration[hashes_key].items():
                old_hash = previous[hashes_key].get(file_name)
                if old_hash is not None and old_hash != new_hash:
                    raise RuntimeError(f"Refusing to resume: file {file_name} changed since the run started "
                                       f"(hash in {hashes_key} differs).")


# ---------------------------------------------------------------- dry run


def load_tokenizer_from_local_cache(model_name, model_revision):
    """The model's tokenizer if transformers is installed and the tokenizer files are already in the local
    Hugging Face cache; otherwise None. Never downloads."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, revision=model_revision, local_files_only=True)
    except Exception:
        return None


def print_dry_run_summary(document_records, condition_records, model_type, model_name, model_revision, persona_set,
                          output_folder):
    print(f"Dry run: nothing written. Would create {output_folder}")
    print(f"  persona set: {persona_set.name} ({len(condition_records)} conditions)")
    print(f"  model: {model_name} @ {model_revision} ({model_type})")
    print(f"  documents: {len(document_records)}")
    print(f"  (document, condition) pairs to score: {len(document_records) * len(condition_records)}")
    tokenizer = load_tokenizer_from_local_cache(model_name, model_revision)
    if tokenizer is None:
        print("  token count: skipped (tokenizer not available locally; install transformers and fetch the "
              f"tokenizer files of {model_name} to get it)")
        return None
    counts = prompt_rendering.count_tokens_for_scoring_inputs(tokenizer, document_records, condition_records,
                                                              model_type, model_name)
    print(f"  document tokens scored (document tokens x conditions): {counts['document_tokens_scored']:,}")
    print(f"  total tokens through the model (including prompts): {counts['total_tokens_through_model']:,}")
    return counts


# ---------------------------------------------------------------- main


def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona-set", required=True,
                        help="Persona set name (a folder under prompts/tools/persona_checks/) or a folder path")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Pipeline run folder (reads final/sdf_corpus.jsonl) or a corpus .jsonl file")
    parser.add_argument("--model-type", default="instruct", choices=MODEL_TYPES,
                        help="instruct: instruction-tuned model, scored through its chat template with personas "
                             "as system prompts. base: pretrained model with no chat template, scored with a "
                             "plain-text author preamble naming the persona. Also picks the default model. "
                             "The scorer reads the type from the run configuration")
    parser.add_argument("--model-name", default=None,
                        help="Hugging Face model whose tokenizer (and chat template, for instruct) defines the "
                             "prompts. Default per model type: see scoring_model_defaults.py. A model other than "
                             "the default needs --model-revision too")
    parser.add_argument("--model-revision", default=None,
                        help="Pinned 40-character commit hash of the model repository")
    parser.add_argument("--resume-folder", default=None, help="Existing run folder to resume")
    parser.add_argument("--document-ids-file", default=None,
                        help="Text file of corpus doc_ids, one per line; only these corpus documents are used")
    parser.add_argument("--maximum-documents", type=int, default=None,
                        help="Use at most this many corpus documents (after --document-ids-file)")
    parser.add_argument("--extra-documents-jsonl", action="append", default=[],
                        help="Rewritten documents file from the contract; may be given several times")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the number of conditions, documents and (if a tokenizer is available "
                             "locally) tokens, and write nothing")
    return parser.parse_args(argument_list)


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    persona_set = persona_sets.load_persona_set(arguments.persona_set)
    framing_directory = PERSONA_CHECKS_PROMPT_DIRECTORY / persona_sets.MODEL_TYPE_FRAMING_DIRECTORY_NAME
    generation_prompt_directory = PERSONA_CHECKS_PROMPT_DIRECTORY / persona_sets.GENERATION_PROMPT_DIRECTORY_NAME
    framing = persona_sets.load_model_type_framing(framing_directory)
    generation_prompt_files = persona_sets.load_generation_prompt_files(generation_prompt_directory)
    model_name, model_revision, model_source = resolve_scoring_model(
        arguments.model_type, persona_set.configuration.get("default_models"), arguments.model_name,
        arguments.model_revision)

    corpus_path, audit_parent_directory = resolve_input(arguments.input)
    document_ids = read_document_ids_file(arguments.document_ids_file) if arguments.document_ids_file else None
    allowed_directions = set(persona_set.rewrite_direction_names().values())
    extra_documents = []
    for path in arguments.extra_documents_jsonl:
        extra_documents.extend(load_extra_documents(path, allowed_directions))
    selected_documents = select_documents(load_corpus_documents(corpus_path), document_ids,
                                          arguments.maximum_documents, extra_documents)
    document_records = build_document_records(selected_documents, generation_prompt_files)
    condition_records = prompt_rendering.build_condition_records(persona_set, arguments.model_type, framing)

    timestamp = timestamp_now()
    is_resume = arguments.resume_folder is not None
    if is_resume:
        run_folder = Path(arguments.resume_folder)
        if not run_folder.is_dir():
            raise SystemExit(f"--resume-folder {run_folder} does not exist")
    else:
        run_folder = audit_parent_directory / (
            f"{RUN_FOLDER_NAME_PREFIX}_{persona_set.name}_{arguments.model_type}_{timestamp}")

    if arguments.dry_run:
        print_dry_run_summary(document_records, condition_records, arguments.model_type, model_name,
                              model_revision, persona_set, run_folder)
        return None
    if not is_resume and run_folder.exists():
        raise SystemExit(f"{run_folder} already exists (same persona set, model type and minute). Wait a minute "
                         f"or pass --resume-folder {run_folder} to resume it.")

    configuration = {
        "run_folder_name": run_folder.name, "created_timestamp": timestamp,
        "resumed_from_existing_folder": is_resume,
        "model_type": arguments.model_type, "model_name": model_name, "model_revision": model_revision,
        "model_selection_source": model_source,
        "persona_set": {
            "name": persona_set.name, "folder": utils.repo_relative(persona_set.folder),
            "configuration": persona_set.configuration,
            "positive_control_instructions_are_draft": persona_set.positive_control_instructions_are_draft,
            "neutral_control_instructions_are_draft": persona_set.neutral_control_instructions_are_draft},
        "pass_criteria": persona_set.configuration["pass_criteria"],
        "document_selection": {
            "corpus_path": utils.repo_relative(corpus_path), "document_ids_file": (utils.repo_relative(arguments.document_ids_file)
                                  if arguments.document_ids_file else None),
            "maximum_documents": arguments.maximum_documents,
            "extra_documents_files": [utils.repo_relative(path) for path in arguments.extra_documents_jsonl],
            "number_of_corpus_documents": len(selected_documents) - len(extra_documents),
            "number_of_extra_documents": len(extra_documents),
            "document_ids": [record["document_id"] for record in document_records]},
        "number_of_conditions": len(condition_records),
        "prompt_file_sha256": collect_prompt_file_hashes(persona_set, arguments.model_type, framing_directory,
                                                         generation_prompt_directory),
        "persona_set_file_sha256": persona_set.file_sha256,
        "input_file_sha256": collect_input_file_hashes(corpus_path, arguments.document_ids_file,
                                                       arguments.extra_documents_jsonl)}

    scoring_inputs_directory = run_folder / "scoring_inputs"
    documents_path = scoring_inputs_directory / "documents.jsonl"
    conditions_path = scoring_inputs_directory / "conditions.jsonl"

    if is_resume:
        check_resume_is_compatible(run_folder, configuration)
        if utils.load_jsonl(documents_path) != document_records or utils.load_jsonl(conditions_path) != condition_records:
            raise RuntimeError("Refusing to resume: rebuilt documents or conditions differ from the files "
                               f"already in {scoring_inputs_directory} (different selection arguments?)")
        print(f"Resuming {run_folder}: scoring inputs unchanged ({len(document_records)} documents, "
              f"{len(condition_records)} conditions)")
    else:
        scoring_inputs_directory.mkdir(parents=True)
        utils.save_jsonl(document_records, documents_path)
        utils.save_jsonl(condition_records, conditions_path)
        print(f"Wrote {len(document_records)} documents and {len(condition_records)} conditions to "
              f"{scoring_inputs_directory}")
    configuration["scoring_input_file_sha256"] = {"documents.jsonl": sha256_of_file(documents_path),
                                                  "conditions.jsonl": sha256_of_file(conditions_path)}
    while (run_folder / f"run_configuration_{timestamp}.json").exists():
        timestamp += "_again"  # never overwrite an earlier configuration, for example on a resume within a minute
    configuration_path = run_folder / f"run_configuration_{timestamp}.json"
    configuration_path.write_text(json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run configuration: {configuration_path}\nRun folder: {run_folder}")
    return run_folder


if __name__ == "__main__":
    main()
