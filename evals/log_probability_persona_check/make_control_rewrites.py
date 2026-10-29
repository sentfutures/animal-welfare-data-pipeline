"""Control rewrites for the log-probability persona check: rewrite a sample of corpus documents with Claude so
that the check can be tested against documents whose persona-relevant behaviour is known to have changed.

For one persona set the command makes up to three kinds of rewrite of each chosen source document. The persona
set's `persona_check_configuration.json` names the directions and its instruction files hold the prompts:

    toward    positive control: the document gains the persona's behaviour   (positive_control_rewrite_instructions.txt)
    away      positive control, optional: the document loses the behaviour   (positive_control_rewrite_instructions.txt)
    neutral   neutral control: an edit of similar size that changes register only (neutral_control_rewrite_instructions.txt)

Document ids are `<source document id>__<direction name>`, for example `matrix_000005__preachier`. The output file
`rewritten_documents_<timestamp>.jsonl` is read by
`build_scoring_inputs.py --extra-documents-jsonl` (contract: run_folder_layout_and_interfaces.md, "Rewrites").

Choosing the sources: `--number-of-documents` documents are sampled with `--seed`, stratified by the corpus
variable `--stratify-by` (default tone): each value gets a share of the sample proportional to its share of the
eligible corpus (largest remainder). Documents in which the AI barely appears cannot show a change in its
behaviour, so by default documents whose `centrality` is "a minor detail mentioned only in passing" or whose
`resolution` says the AI raises no welfare consideration are not eligible (`--exclude-where VARIABLE=TEXT` adds
rules, `--include-ineligible-documents` removes the defaults). `--source-document-ids-file` replaces the sampling.

Choose sources for the away direction that already show the behaviour: a document that does not show it has
nothing to take away, and the rewrite then tests nothing. If the corpus has too few such documents, drop the away
direction (`--directions toward neutral`) instead of rewriting documents that do not show it.

Claude calls go through `shared/api.py`, so the calls land in the repository's cost log and use the key from the
repository's own `.env`. The backend is the one in `config.yaml`; use `backend: api` for the runs that count.
Because `shared/api.py` runs every call with extended thinking off, the rewrites are made without thinking.

Before any paid call the persona set's instruction file for each requested kind must be free of the `# DRAFT`
marker (`PersonaSet.require_reviewed_instructions`). A draft blocks the paid run but not `--dry-run`.

`--dry-run` renders every request, writes them and a cost estimate to the run folder, and makes no call.
Resuming (`--resume-folder`) skips every (source, direction) pair already in the rewritten documents file and
retries the pairs in the failures file. Failures (refusal, hitting the output limit, empty output, transport
errors) go to `rewrite_failures_<timestamp>.jsonl`.

Usage (from the repository root):
    python evals/log_probability_persona_check/make_control_rewrites.py --persona-set preachiness \\
        --input outputs/sdf/latest --dry-run
    python evals/log_probability_persona_check/make_control_rewrites.py --persona-set preachiness \\
        --input outputs/sdf/latest --number-of-documents 10 --seed 0
    python evals/log_probability_persona_check/make_control_rewrites.py --persona-set preachiness \\
        --input outputs/sdf/latest --directions toward neutral --stratify-by tone
    python evals/log_probability_persona_check/make_control_rewrites.py --persona-set preachiness \\
        --input outputs/sdf/latest --resume-folder <existing rewrite run folder>
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.log_probability_persona_check import build_scoring_inputs, persona_sets  # noqa: E402
from evals.log_probability_persona_check.timed_progress_and_jsonl_helpers import timestamp_unused_in_folder  # noqa: E402
from shared import api, utils  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
DEFAULT_OUTPUT_PARENT_DIRECTORY = REPOSITORY_ROOT / "outputs" / "log_probability_persona_check_rewrites"
RUN_FOLDER_NAME_PREFIX = "control_rewrites"
REWRITTEN_DOCUMENTS_FILE_PREFIX = "rewritten_documents_"
COST_LOG_STAGE_NAME = "persona_check_control_rewrite"

DEFAULT_REWRITE_MODEL = "claude-opus-5"
# Sent without streaming, so it must stay under the Anthropic SDK's non-streaming limit: the SDK refuses a call whose
# expected time, 3600 * max_tokens / 128000 seconds, exceeds 600 seconds (max_tokens above 21333). The documents are
# about 2,000 tokens, so 16000 leaves ample room.
DEFAULT_MAXIMUM_OUTPUT_TOKENS = 16000
DEFAULT_NUMBER_OF_DOCUMENTS = 10
DEFAULT_STRATIFY_BY = "tone"
DEFAULT_WORKER_COUNT = 3

# Documents in which the AI barely appears have no behaviour to change. (variable, text it must not contain)
DEFAULT_EXCLUSIONS = [("centrality", "a minor detail mentioned only in passing"),
                      ("resolution", "correctly raises no welfare consideration at all")]
# When stratifying by tone, keep at least this many documents of the tone in which the behaviour is least
# likely to be present, so that the away direction has sources that show it.
DEFAULT_GUARANTEED_MINIMUMS = {"tone": {"skeptical or displeased": 3}}

DIRECTION_ROLES = ["toward", "away", "neutral"]
INSTRUCTION_KIND_BY_ROLE = {"toward": "positive", "away": "positive", "neutral": "neutral"}
MAXIMUM_LENGTH_RATIO_DEVIATION = 0.15
LANGUAGES_COUNTED_IN_CHARACTERS = {"Mandarin Chinese", "Japanese"}  # no spaces between words
LANGUAGES_IN_LATIN_SCRIPT = {"English", "Spanish", "German", "Portuguese", "French", "Polish", "Indonesian",
                             "Vietnamese", "Norwegian"}

# Dry-run estimate assumptions
LATIN_SCRIPT_CHARACTERS_PER_TOKEN = 4.0
OTHER_SCRIPT_CHARACTERS_PER_TOKEN = 1.5
ASSUMED_THINKING_TOKENS_PER_CALL = 8000  # the first run's value; shared/api.py disables thinking, so this is a margin


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


# ----------------------------------------------------------------------------
# Source selection
# ----------------------------------------------------------------------------

def parse_exclusion_rules(rule_texts):
    rules = []
    for rule_text in rule_texts:
        variable, separator, text = rule_text.partition("=")
        if not separator or not variable or not text:
            raise SystemExit(f"--exclude-where must look like VARIABLE=TEXT, got {rule_text!r}")
        rules.append((variable, text))
    return rules


def parse_guaranteed_minimums(rule_texts, stratify_by):
    """`VALUE=N` rules, or the defaults for the stratifying variable when none are given."""
    if not rule_texts:
        return dict(DEFAULT_GUARANTEED_MINIMUMS.get(stratify_by, {}))
    minimums = {}
    for rule_text in rule_texts:
        value, separator, number = rule_text.rpartition("=")
        if not separator or not value or not number.isdigit():
            raise SystemExit(f"--guarantee-stratum must look like VALUE=NUMBER, got {rule_text!r}")
        minimums[value] = int(number)
    return minimums


def is_eligible_for_rewriting(document, exclusion_rules):
    variables = document["variables"]
    return not any(text in str(variables.get(variable, "")) for variable, text in exclusion_rules)


def allocate_documents_per_stratum(number_of_documents, eligible_counts_by_stratum, guaranteed_minimums=None):
    """Proportional to each stratum's share of the eligible documents (largest remainder), never more than a
    stratum has, with each guaranteed stratum raised to its minimum (or to all it has) at the expense of the
    largest other stratum."""
    guaranteed_minimums = guaranteed_minimums or {}
    total_eligible = sum(eligible_counts_by_stratum.values())
    if number_of_documents > total_eligible:
        raise SystemExit(f"{number_of_documents} documents requested but only {total_eligible} are eligible")
    allocation = {stratum: 0 for stratum in eligible_counts_by_stratum}
    remaining_to_allocate = number_of_documents
    open_strata = set(allocation)
    while remaining_to_allocate > 0:
        open_total = sum(eligible_counts_by_stratum[stratum] for stratum in open_strata)
        exact = {stratum: remaining_to_allocate * eligible_counts_by_stratum[stratum] / open_total
                 for stratum in open_strata}
        shares = {stratum: int(math.floor(value)) for stratum, value in exact.items()}
        leftover = remaining_to_allocate - sum(shares.values())
        by_remainder = sorted(open_strata, key=lambda stratum: (exact[stratum] - shares[stratum], stratum),
                              reverse=True)
        for stratum in by_remainder[:leftover]:
            shares[stratum] += 1
        for stratum in sorted(open_strata):
            room = eligible_counts_by_stratum[stratum] - allocation[stratum]
            taken = min(shares[stratum], room)
            allocation[stratum] += taken
            remaining_to_allocate -= taken
            if allocation[stratum] == eligible_counts_by_stratum[stratum]:
                open_strata.discard(stratum)
    for stratum, minimum in guaranteed_minimums.items():
        if stratum not in allocation:
            continue
        minimum = min(minimum, eligible_counts_by_stratum[stratum], number_of_documents)
        while allocation[stratum] < minimum:
            donors = [other for other in allocation if other != stratum and allocation[other] > 0
                      and allocation[other] > guaranteed_minimums.get(other, 0)]
            if not donors:
                break
            donor = max(donors, key=lambda other: (allocation[other], other))
            allocation[donor] -= 1
            allocation[stratum] += 1
    return allocation


def select_source_documents(documents, number_of_documents, seed, stratify_by, exclusion_rules,
                            guaranteed_minimums=None):
    """Deterministic for (documents, number_of_documents, seed, stratify_by). Returns documents sorted by id.
    `documents` are corpus documents as loaded by build_scoring_inputs.load_corpus_documents."""
    eligible_by_stratum = {}
    for document in sorted(documents, key=lambda document: document["document_id"]):
        if stratify_by not in document["variables"]:
            raise SystemExit(f"Document {document['document_id']} has no variable {stratify_by!r} to stratify by; "
                             f"its variables are {sorted(document['variables'])}")
        if is_eligible_for_rewriting(document, exclusion_rules):
            eligible_by_stratum.setdefault(document["variables"][stratify_by], []).append(document)
    allocation = allocate_documents_per_stratum(
        number_of_documents, {stratum: len(members) for stratum, members in eligible_by_stratum.items()},
        guaranteed_minimums)
    random_generator = random.Random(seed)
    selected = []
    for stratum in sorted(eligible_by_stratum):
        selected.extend(random_generator.sample(eligible_by_stratum[stratum], allocation[stratum]))
    return sorted(selected, key=lambda document: document["document_id"])


def select_documents_from_ids_file(documents, ids_file):
    documents_by_id = {document["document_id"]: document for document in documents}
    wanted_ids = build_scoring_inputs.read_document_ids_file(ids_file)
    unknown = [document_id for document_id in wanted_ids if document_id not in documents_by_id]
    if unknown:
        raise SystemExit(f"Unknown document ids in {ids_file}: {unknown}")
    return [documents_by_id[document_id] for document_id in wanted_ids]


# ----------------------------------------------------------------------------
# Directions, prompts and lengths
# ----------------------------------------------------------------------------

def plan_rewrite_directions(persona_set, requested_roles):
    """[(role, direction name, instruction template, 'positive' or 'neutral')] for the requested roles, in
    toward, away, neutral order. Raises SystemExit for a role the persona set does not configure."""
    direction_names = persona_set.rewrite_direction_names()
    plan = []
    for role in DIRECTION_ROLES:
        if role not in requested_roles:
            continue
        if role not in direction_names:
            raise SystemExit(f"Persona set {persona_set.name} has no {role!r} rewrite direction; "
                             f"it configures {sorted(direction_names)}")
        kind = INSTRUCTION_KIND_BY_ROLE[role]
        template = (persona_set.positive_control_rewrite_instructions if kind == "positive"
                    else persona_set.neutral_control_rewrite_instructions)
        plan.append((role, direction_names[role], template, kind))
    if not plan:
        raise SystemExit("No rewrite directions requested")
    return plan


def render_rewrite_prompt(instruction_template, direction_name, document_text):
    """Fills {direction} (the direction name in capitals, as the instruction files name it) and then
    {document_text}; that order keeps braces inside the document from being read as slots."""
    return (instruction_template.replace("{direction}", direction_name.upper())
            .replace("{document_text}", document_text))


def length_in_unit(text, language):
    """(length, unit): non-whitespace characters for languages without spaces between words, else words."""
    if language in LANGUAGES_COUNTED_IN_CHARACTERS:
        return len("".join(text.split())), "characters"
    return len(text.split()), "words"


# ----------------------------------------------------------------------------
# One rewrite
# ----------------------------------------------------------------------------

def make_one_rewrite(source_document, direction_name, instruction_template, model, maximum_output_tokens):
    """Returns (record, failure_record). Exactly one is not None. Calls shared.api.call_claude."""
    source_document_id = source_document["document_id"]
    prompt_text = render_rewrite_prompt(instruction_template, direction_name, source_document["content"])
    failure_base = {"source_document_id": source_document_id, "rewrite_direction": direction_name}
    try:
        text, stop_reason = api.call_claude(
            user_message=prompt_text, model=model, max_tokens=maximum_output_tokens, return_stop_reason=True,
            stage=COST_LOG_STAGE_NAME, item_id=f"{source_document_id}__{direction_name}")
    except api.UsageLimitExceeded:
        raise
    except Exception as error:  # noqa: BLE001 - failure after the api module's own retries
        return None, {**failure_base, "failure_kind": "transport_failure", "description": repr(error)}
    text = (text or "").strip()
    if stop_reason == "refusal":
        return None, {**failure_base, "failure_kind": "refusal", "stop_reason": stop_reason}
    if stop_reason == "max_tokens":
        return None, {**failure_base, "failure_kind": "max_tokens", "stop_reason": stop_reason}
    if not text:
        return None, {**failure_base, "failure_kind": "empty_output", "stop_reason": stop_reason}
    language = source_document["language"]
    source_length, length_unit = length_in_unit(source_document["content"], language)
    rewrite_length, _ = length_in_unit(text, language)
    length_ratio = rewrite_length / source_length if source_length else None
    record = {
        "document_id": f"{source_document_id}__{direction_name}",
        "source_document_id": source_document_id,
        "rewrite_direction": direction_name,
        "content": text,
        "language": language,
        "variables": source_document["variables"],
        "rewrite_model_requested": model,
        "rewrite_model_used": model,  # shared/api.py does not return the model that served the call
        "source_word_count": len(source_document["content"].split()),
        "rewrite_word_count": len(text.split()),
        "length_unit": length_unit,
        "source_length": source_length,
        "rewrite_length": rewrite_length,
        "length_ratio": length_ratio,
        "length_within_fifteen_percent": (length_ratio is not None
                                          and abs(length_ratio - 1) <= MAXIMUM_LENGTH_RATIO_DEVIATION),
    }
    return record, None


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def find_rewritten_documents_file(run_folder):
    """The existing rewritten documents file of a run folder, or None. A folder holds at most one."""
    paths = sorted(Path(run_folder).glob(f"{REWRITTEN_DOCUMENTS_FILE_PREFIX}*.jsonl"))
    if len(paths) > 1:
        raise SystemExit(f"{run_folder} holds several rewritten documents files: {[path.name for path in paths]}")
    return paths[0] if paths else None


def find_completed_pairs(rewritten_documents_path):
    if rewritten_documents_path is None or not Path(rewritten_documents_path).exists():
        return set()
    return {(record["source_document_id"], record["rewrite_direction"])
            for record in utils.load_jsonl(rewritten_documents_path)}


def build_tasks(selected_documents, direction_plan):
    return [(document, direction_name, template)
            for document in selected_documents for _, direction_name, template, _ in direction_plan]


def run_rewrites(run_folder, session_timestamp, rewritten_documents_path, direction_plan, selected_documents,
                 model, maximum_output_tokens, worker_count):
    completed_pairs = find_completed_pairs(rewritten_documents_path)
    tasks = [task for task in build_tasks(selected_documents, direction_plan)
             if (task[0]["document_id"], task[1]) not in completed_pairs]
    print(f"{len(tasks)} rewrites to make ({len(completed_pairs)} already done).", flush=True)
    failures_path = Path(run_folder) / f"rewrite_failures_{session_timestamp}.jsonl"
    started = time.time()
    failure_count = 0

    def work(task):
        document, direction_name, template = task
        return make_one_rewrite(document, direction_name, template, model, maximum_output_tokens)

    # Results come back in task order and are written here, one at a time, so a stopped run resumes cleanly.
    for finished_count, (record, failure_record) in enumerate(utils.parallel_map(work, tasks, worker_count), 1):
        if record is not None:
            utils.append_jsonl(record, rewritten_documents_path)
        else:
            failure_count += 1
            utils.append_jsonl(failure_record, failures_path)
        print(f"  {finished_count}/{len(tasks)} done, {failure_count} failed, "
              f"{time.time() - started:.0f} seconds, cost so far ${api.get_total_cost():.2f} "
              f"(whole cost log)", flush=True)
    return failure_count


# ----------------------------------------------------------------------------
# Dry run
# ----------------------------------------------------------------------------

def estimate_tokens_from_characters(text, language):
    characters_per_token = (LATIN_SCRIPT_CHARACTERS_PER_TOKEN if language in LANGUAGES_IN_LATIN_SCRIPT
                            else OTHER_SCRIPT_CHARACTERS_PER_TOKEN)
    return len(text) / characters_per_token


def look_up_prices_per_million_tokens(model):
    prices = api._PRICING.get(model)
    if prices is None:
        raise SystemExit(f"{model!r} is not in shared/api.py _PRICING, so its cost cannot be estimated. "
                         f"Priced models: {sorted(api._PRICING)}")
    return prices


def estimate_cost(rendered_requests, model):
    """Input tokens from character counts; output tokens = the document's own token estimate plus a thinking
    margin. No cache hits, no refusals."""
    input_price, output_price = look_up_prices_per_million_tokens(model)
    total_input_tokens = total_output_tokens = 0.0
    for rendered in rendered_requests:
        total_input_tokens += estimate_tokens_from_characters(rendered["prompt_text"], rendered["language"])
        total_output_tokens += (estimate_tokens_from_characters(rendered["source_content"], rendered["language"])
                                + ASSUMED_THINKING_TOKENS_PER_CALL)
    cost = (total_input_tokens * input_price + total_output_tokens * output_price) / 1_000_000
    return {"model": model, "calls": len(rendered_requests),
            "estimated_input_tokens": round(total_input_tokens),
            "estimated_output_tokens": round(total_output_tokens), "estimated_cost_dollars": round(cost, 2),
            "assumptions": (f"input tokens = characters / {LATIN_SCRIPT_CHARACTERS_PER_TOKEN} for Latin-script "
                            f"languages and / {OTHER_SCRIPT_CHARACTERS_PER_TOKEN} for others; output tokens = the "
                            f"source document's token estimate plus {ASSUMED_THINKING_TOKENS_PER_CALL} thinking "
                            f"tokens per call; {model} prices ${input_price} input and ${output_price} output "
                            f"per million tokens (shared/api.py); no cache hits; no refusals")}


def run_dry_run(run_folder, session_timestamp, direction_plan, selected_documents, model, maximum_output_tokens):
    rendered_requests = []
    for document, direction_name, template in build_tasks(selected_documents, direction_plan):
        prompt_text = render_rewrite_prompt(template, direction_name, document["content"])
        rendered_requests.append({
            "source_document_id": document["document_id"], "rewrite_direction": direction_name,
            "language": document["language"], "source_content": document["content"], "prompt_text": prompt_text,
            "request": {"model": model, "max_tokens": maximum_output_tokens, "user_message": prompt_text}})
    utils.save_jsonl(rendered_requests, Path(run_folder) / f"rendered_requests_{session_timestamp}.jsonl")
    estimate = estimate_cost(rendered_requests, model)
    (Path(run_folder) / f"dry_run_cost_estimate_{session_timestamp}.json").write_text(
        json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps(estimate, indent=2))
    return estimate


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_arguments(argument_list=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--persona-set", required=True,
                        help="Persona set name (a folder under prompts/tools/persona_checks/) or a folder path")
    parser.add_argument("--input", default=build_scoring_inputs.DEFAULT_INPUT,
                        help="Pipeline run folder (reads final/sdf_corpus.jsonl) or a corpus .jsonl file")
    parser.add_argument("--directions", nargs="+", choices=DIRECTION_ROLES, default=None,
                        help="Which rewrites to make: toward, away, neutral. Default: every direction the "
                             "persona set configures. Choose sources for away that already show the behaviour, "
                             "or drop it")
    parser.add_argument("--number-of-documents", type=int, default=DEFAULT_NUMBER_OF_DOCUMENTS,
                        help="How many source documents to rewrite")
    parser.add_argument("--stratify-by", default=DEFAULT_STRATIFY_BY,
                        help="Corpus variable the sample is spread across, in proportion to the eligible corpus "
                             "(default: tone)")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the sampling")
    parser.add_argument("--source-document-ids-file", default=None,
                        help="Text file with one document id per line; replaces the sampling")
    parser.add_argument("--exclude-where", action="append", default=[], metavar="VARIABLE=TEXT",
                        help="Documents whose VARIABLE contains TEXT are not eligible; may be repeated")
    parser.add_argument("--include-ineligible-documents", action="store_true",
                        help="Remove the default exclusions (centrality a minor detail, resolution with no "
                             "welfare consideration)")
    parser.add_argument("--guarantee-stratum", action="append", default=[], metavar="VALUE=NUMBER",
                        help="At least NUMBER sources whose --stratify-by variable is VALUE; may be repeated. "
                             "Default with --stratify-by tone: 'skeptical or displeased=3'")
    parser.add_argument("--model", default=DEFAULT_REWRITE_MODEL, help="Claude model that makes the rewrites")
    parser.add_argument("--maximum-output-tokens", type=int, default=DEFAULT_MAXIMUM_OUTPUT_TOKENS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT, help="Concurrent Claude calls")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Pipeline config.yaml (backend, cost log)")
    parser.add_argument("--cost-log-path", default=None,
                        help="Cost log file; default: the one named in config.yaml")
    parser.add_argument("--output-parent-directory", default=str(DEFAULT_OUTPUT_PARENT_DIRECTORY),
                        help="The run folder is created inside this directory")
    parser.add_argument("--resume-folder", default=None, help="Existing rewrite run folder to resume")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render every request, write them and a cost estimate to the run folder, and print "
                             "the estimate; no Claude call and no key needed")
    return parser.parse_args(argument_list)


def check_resume_is_compatible(run_folder, configuration):
    first_configuration = load_first_configuration(run_folder)
    for key in ("persona_set_name", "model_requested", "corpus_sha256", "instruction_file_sha256",
                "direction_names"):
        if first_configuration[key] != configuration[key]:
            raise SystemExit(f"Refusing to resume {run_folder}: {key} differs from the first run "
                             f"({first_configuration[key]!r} then, {configuration[key]!r} now)")
    return first_configuration


def load_first_configuration(run_folder):
    paths = sorted(Path(run_folder).glob("rewrite_run_configuration_*.json"))
    if not paths:
        raise SystemExit(f"No rewrite_run_configuration_*.json in {run_folder}; not a rewrite run folder")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def main(argument_list=None):
    arguments = parse_arguments(argument_list)
    persona_set = persona_sets.load_persona_set(arguments.persona_set)
    requested_roles = arguments.directions or list(persona_set.rewrite_direction_names())
    direction_plan = plan_rewrite_directions(persona_set, requested_roles)
    corpus_path, _ = build_scoring_inputs.resolve_input(arguments.input)
    corpus_documents = build_scoring_inputs.load_corpus_documents(corpus_path)

    session_timestamp = timestamp_now()
    is_resume = arguments.resume_folder is not None
    if is_resume:
        run_folder = Path(arguments.resume_folder)
        if not run_folder.is_dir():
            raise SystemExit(f"--resume-folder {run_folder} does not exist")
    else:
        dry_run_label = "dry_run_" if arguments.dry_run else ""
        run_folder = (Path(arguments.output_parent_directory)
                      / f"{RUN_FOLDER_NAME_PREFIX}_{dry_run_label}{persona_set.name}_{session_timestamp}")
        if run_folder.exists():
            raise SystemExit(f"{run_folder} already exists (same persona set and minute). Wait a minute or pass "
                             f"--resume-folder {run_folder}.")

    configuration = {
        "persona_set_name": persona_set.name, "model_requested": arguments.model,
        "maximum_output_tokens": arguments.maximum_output_tokens,
        "direction_names": {role: name for role, name, _, _ in direction_plan},
        "instruction_file_sha256": {
            persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME:
                persona_set.file_sha256.get(persona_sets.POSITIVE_CONTROL_INSTRUCTIONS_FILE_NAME),
            persona_sets.NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME:
                persona_set.file_sha256.get(persona_sets.NEUTRAL_CONTROL_INSTRUCTIONS_FILE_NAME)},
        "corpus_path": utils.repo_relative(corpus_path),
        "corpus_sha256": build_scoring_inputs.sha256_of_file(corpus_path),
        "instructions_are_draft": {"positive": persona_set.positive_control_instructions_are_draft,
                                   "neutral": persona_set.neutral_control_instructions_are_draft},
        "dry_run": arguments.dry_run, "worker_count": arguments.workers, "resumed": is_resume}

    if is_resume:
        first_configuration = check_resume_is_compatible(run_folder, configuration)
        documents_by_id = {document["document_id"]: document for document in corpus_documents}
        selected_documents = [documents_by_id[document_id]
                              for document_id in first_configuration["selected_source_document_ids"]]
        selection = {key: first_configuration[key] for key in
                     ("selection_method", "seed", "stratify_by", "number_of_documents_requested")}
    else:
        if arguments.source_document_ids_file:
            selected_documents = select_documents_from_ids_file(corpus_documents, arguments.source_document_ids_file)
            selection_method = "source_document_ids_file"
        else:
            exclusion_rules = ([] if arguments.include_ineligible_documents else list(DEFAULT_EXCLUSIONS))
            exclusion_rules += parse_exclusion_rules(arguments.exclude_where)
            guaranteed_minimums = parse_guaranteed_minimums(arguments.guarantee_stratum, arguments.stratify_by)
            selected_documents = select_source_documents(
                corpus_documents, arguments.number_of_documents, arguments.seed, arguments.stratify_by,
                exclusion_rules, guaranteed_minimums)
            selection_method = f"stratified_by_{arguments.stratify_by}"
        selection = {"selection_method": selection_method, "seed": arguments.seed,
                     "stratify_by": arguments.stratify_by,
                     "number_of_documents_requested": arguments.number_of_documents}
    configuration.update(selection)
    configuration["selected_source_document_ids"] = [document["document_id"] for document in selected_documents]
    configuration["selected_source_values_of_stratify_by_variable"] = {
        document["document_id"]: document["variables"].get(selection["stratify_by"])
        for document in selected_documents}

    run_folder.mkdir(parents=True, exist_ok=True)
    if is_resume:  # a resume in the same minute as an earlier session must not overwrite or extend its files
        session_timestamp = timestamp_unused_in_folder(run_folder, [
            "rewrite_run_configuration_{timestamp}.json", "rewrite_failures_{timestamp}.jsonl",
            "rendered_requests_{timestamp}.jsonl", "dry_run_cost_estimate_{timestamp}.json"])
    (run_folder / f"rewrite_run_configuration_{session_timestamp}.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run folder: {run_folder}; {len(selected_documents)} source documents, directions "
          f"{[name for _, name, _, _ in direction_plan]}", flush=True)

    if arguments.dry_run:
        run_dry_run(run_folder, session_timestamp, direction_plan, selected_documents, arguments.model,
                    arguments.maximum_output_tokens)
        return run_folder

    for kind in sorted({kind for _, _, _, kind in direction_plan}):
        persona_set.require_reviewed_instructions(kind)
    try:
        api.init(arguments.config, cost_log_path=arguments.cost_log_path)
    except KeyError as error:
        raise SystemExit(f"{error} is not set: put it in this repository's .env (the backend in "
                         f"{arguments.config} needs it)") from error
    rewritten_documents_path = (find_rewritten_documents_file(run_folder)
                                or run_folder / f"{REWRITTEN_DOCUMENTS_FILE_PREFIX}{session_timestamp}.jsonl")
    failure_count = run_rewrites(run_folder, session_timestamp, rewritten_documents_path, direction_plan,
                                 selected_documents, arguments.model, arguments.maximum_output_tokens,
                                 arguments.workers)
    if failure_count:
        print(f"{failure_count} rewrites failed; see rewrite_failures_{session_timestamp}.jsonl. "
              f"Resume with --resume-folder {run_folder} to retry them.")
    return run_folder


if __name__ == "__main__":
    main()
