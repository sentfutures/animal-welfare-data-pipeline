"""Conditions, generation prompts, chat messages, and the location of a document's content tokens inside the
scored sequence, for the log-probability persona check.

Shared by build_scoring_inputs.py, the scorer and the tests. Nothing here calls a model or an API. Layout
contract: run_folder_layout_and_interfaces.md in this folder.

Which tokens are scored (instruct): the sequence is  <chat template: system turn (if any), user turn, then the
assistant header exactly as the model sees it when it is about to generate>  +  <document content>  +
<whatever the chat template writes after an assistant message, normally the end-of-turn token>.
Only the document content tokens are scored. The first content token is scored conditioned on the
assistant header; the end-of-turn token is never scored.

Gemma 4 note: with thinking off, the template's generation prompt ends with an empty thought channel
("<|turn>model\\n<|channel>thought\\n<channel|>") whereas the template renders a finished assistant message
without it. We score under the generation prompt (what the model conditions on when it writes a document)
and take only the text after the assistant content from the finished-message rendering.

Base model type: a pretrained model has no chat template, so the sequence is
<the tokenizer's bos token text, if it has one>  +  <preamble>  +  "\\n\\n"  +  <document content>
with nothing after the document (no end-of-sequence token is scored or even present). The preamble is
"The following text was written by <author>, in response to this request: <generation prompt>", or, for the
no_system_prompt condition, "The following text was written in response to this request: <generation
prompt>". The author is "an AI assistant" for the generic condition and "an AI assistant who <persona
description>" for every persona condition, using the same persona description lines as the instruct
conditions. Templates are in prompts/tools/persona_checks/model_type_framing/base_model_*.txt. As in the instruct
format, only the document content tokens are scored, the first one conditioned on the "\\n\\n" that ends the
preamble. Conditions, condition ids and groups are identical across model types; only the rendering differs.
Scores from the two model types are not comparable.
"""

import string
from dataclasses import dataclass

from evals.log_probability_persona_check.persona_sets import (PersonaSet, ModelTypeFraming, GenerationPromptFiles,
                                                              load_generation_prompt_files, load_model_type_framing)
from evals.log_probability_persona_check.scoring_model_defaults import (MODEL_TYPE_BASE, MODEL_TYPE_INSTRUCT,
                                                                        MODEL_TYPES, check_model_type,
                                                                        chat_template_keyword_arguments_for_model)

BASE_MODEL_PREAMBLE_AND_DOCUMENT_SEPARATOR = "\n\n"

CLOSING_PUNCTUATION_AND_QUOTES = "\"'”’)]」』»"
TERMINAL_PUNCTUATION = ".!?。！？…।۔"


# ---------------------------------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------------------------------

def render_persona_system_prompt(persona_frame, persona_description):
    if "{persona_description}" not in persona_frame:
        raise ValueError("Persona frame has no {persona_description} slot")
    return persona_frame.replace("{persona_description}", persona_description)


def render_base_model_author_description(author_frame, persona_description):
    if "{persona_description}" not in author_frame:
        raise ValueError("Base-model persona author frame has no {persona_description} slot")
    return author_frame.replace("{persona_description}", persona_description)


def build_condition_records(persona_set: PersonaSet, model_type=MODEL_TYPE_INSTRUCT,
                            framing: ModelTypeFraming = None):
    """The conditions of a persona set, in a fixed order: no_system_prompt, generic, the target rewordings,
    the topic control personas (each in file order), then the same-values control personas (each in file order) if the set has them. Rewording
    indices are 1-based. For the preachiness set this is 72 conditions: the 62 of the first preachiness run plus two more same-values personas.

    For the base model type every record also has `author_description` (None for no_system_prompt, the generic
    author description for generic, the persona author frame filled with the same persona description for the
    others). Instruct records do not have the field."""
    check_model_type(model_type)
    framing = framing or load_model_type_framing()
    is_base_model = model_type == MODEL_TYPE_BASE
    groups = persona_set.configuration["condition_groups"]

    conditions = [
        {"condition_id": "no_system_prompt", "condition_group": "no_system_prompt",
         "persona_name": "no_system_prompt", "rewording_index": None, "system_prompt": None},
        {"condition_id": "generic", "condition_group": "generic", "persona_name": "generic",
         "rewording_index": None, "system_prompt": framing.instruct_generic_system_prompt},
    ]
    if is_base_model:
        conditions[0]["author_description"] = None
        conditions[1]["author_description"] = framing.base_generic_author_description

    def add_persona_conditions(condition_group, persona_name, descriptions):
        for rewording_index, description in enumerate(descriptions, start=1):
            condition = {
                "condition_id": f"{persona_name}__rewording_{rewording_index:02d}",
                "condition_group": condition_group, "persona_name": persona_name,
                "rewording_index": rewording_index,
                "system_prompt": render_persona_system_prompt(framing.instruct_persona_frame, description)}
            if is_base_model:
                condition["author_description"] = render_base_model_author_description(
                    framing.base_persona_author_frame, description)
            conditions.append(condition)

    add_persona_conditions(groups["target"]["condition_group"], persona_set.persona_name,
                           persona_set.target_rewordings)
    for persona_name, descriptions in persona_set.topic_control_descriptions_by_persona.items():
        add_persona_conditions(groups["topic_control"]["condition_group"], persona_name, descriptions)
    if persona_set.has_same_values_control:
        for persona_name, descriptions in persona_set.same_values_control_descriptions_by_persona.items():
            add_persona_conditions(groups["same_values_control"]["condition_group"], persona_name, descriptions)

    condition_ids = [condition["condition_id"] for condition in conditions]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("Duplicate condition ids")
    if len(conditions) != persona_set.expected_condition_count():
        raise ValueError(f"Expected {persona_set.expected_condition_count()} conditions, built {len(conditions)}")
    return conditions


# ---------------------------------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------------------------------

def add_final_period_if_missing(resolution_text):
    """Adds "." to `resolution` unless it already ends in terminal punctuation (possibly followed by
    closing quotes or brackets). Trailing whitespace is dropped."""
    text = resolution_text.rstrip()
    if not text:
        raise ValueError("resolution is empty")
    last_meaningful_character = text.rstrip(CLOSING_PUNCTUATION_AND_QUOTES)[-1:] or text[-1]
    if last_meaningful_character in TERMINAL_PUNCTUATION:
        return text
    return text + "."


def generation_prompt_variable_names(template_text):
    """The document variables a generation prompt template needs: every slot except {in_universe_facts}, in
    order of first appearance. For the shipped template these are the twelve axes of the SDF variables matrix
    (document_type, culture, domain, ...)."""
    names = []
    for _, field_name, _, _ in string.Formatter().parse(template_text):
        if field_name and field_name != "in_universe_facts" and field_name not in names:
            names.append(field_name)
    return names


def render_generation_prompt(variables, document_id, generation_prompt_files: GenerationPromptFiles = None):
    """Fills the generation prompt template for one document. Fails loudly on any missing, non-string or
    empty variable. Loads the shipped template and facts when `generation_prompt_files` is not given (pass the
    loaded files when rendering many documents)."""
    files = generation_prompt_files or load_generation_prompt_files()
    variable_names = generation_prompt_variable_names(files.template_text)
    missing = [name for name in variable_names if name not in variables]
    if missing:
        raise KeyError(f"Document {document_id} is missing variables: {missing}")
    slot_values = {"in_universe_facts": files.in_universe_facts}
    for name in variable_names:
        value = variables[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Document {document_id} has an empty or non-string value for '{name}': {value!r}")
        slot_values[name] = value.strip()
    if "resolution" in slot_values:
        slot_values["resolution"] = add_final_period_if_missing(slot_values["resolution"])
    return files.template_text.format(**slot_values)


def build_chat_messages(condition, generation_prompt):
    messages = []
    if condition["system_prompt"] is not None:
        messages.append({"role": "system", "content": condition["system_prompt"]})
    messages.append({"role": "user", "content": generation_prompt})
    return messages


# ---------------------------------------------------------------------------------------------------
# Chat template rendering and document token location
# ---------------------------------------------------------------------------------------------------

class SystemPromptNotInRenderedPromptError(RuntimeError):
    pass


class DocumentTokenLocationError(RuntimeError):
    pass


def model_label(tokenizer, model_name):
    return model_name or getattr(tokenizer, "name_or_path", "unknown model")


def chat_template_keyword_arguments(tokenizer, model_name):
    """The model's extra `apply_chat_template` keyword arguments (scoring_model_defaults.py), looked up by the
    model name, or by the tokenizer's own name when no model name is given."""
    return chat_template_keyword_arguments_for_model(model_name or getattr(tokenizer, "name_or_path", None))


def assert_rendered_prompt_contains_system_prompt(rendered_text, condition, model_name):
    """Raises if a condition with a system prompt does not have that text in the rendered chat prompt."""
    system_prompt = condition["system_prompt"]
    if system_prompt is None:
        return
    if system_prompt.strip() not in rendered_text:
        raise SystemPromptNotInRenderedPromptError(
            f"The chat template of {model_name} dropped the system prompt of condition "
            f"{condition['condition_id']}. Rendered text:\n{rendered_text}")


def render_prompt_text_with_generation_header(tokenizer, condition, generation_prompt, model_name=None):
    """Renders system (if any) + user turns with the template's assistant generation header. Raises if the
    template rejects the system role or silently drops the system prompt."""
    label = model_label(tokenizer, model_name)
    messages = build_chat_messages(condition, generation_prompt)
    try:
        rendered_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_keyword_arguments(tokenizer, model_name))
    except Exception as error:
        raise SystemPromptNotInRenderedPromptError(
            f"The chat template of {label} failed on condition {condition['condition_id']} "
            f"(roles: {[message['role'] for message in messages]}): {error}") from error
    assert_rendered_prompt_contains_system_prompt(rendered_text, condition, label)
    return rendered_text


def render_text_after_assistant_message(tokenizer, condition, generation_prompt, document_text, model_name=None):
    """Text the chat template writes after an assistant message whose content is `document_text`
    (normally the end-of-turn token). Also returns the text before the content in that rendering."""
    label = model_label(tokenizer, model_name)
    messages = build_chat_messages(condition, generation_prompt)
    messages.append({"role": "assistant", "content": document_text})
    reference_text = tokenizer.apply_chat_template(
        messages, tokenize=False, **chat_template_keyword_arguments(tokenizer, model_name))
    assert_rendered_prompt_contains_system_prompt(reference_text, condition, label)
    content_start = reference_text.rfind(document_text)
    if content_start == -1:
        raise DocumentTokenLocationError(
            f"The chat template of {label} changed the assistant content, so the text after it cannot be "
            f"located. Rendered text:\n{reference_text[:2000]}")
    return reference_text[content_start + len(document_text):], reference_text[:content_start]


def tokenize_text(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


@dataclass
class TokenizedConditionAndDocument:
    full_token_ids: list
    document_start_index: int  # first document content token
    document_end_index: int    # one past the last document content token
    prompt_text: str
    suffix_text: str
    # Instruct: whether the finished-message rendering has the same text before the content as the generation
    # prompt. Base: None, because there is no chat template and so no finished-message rendering to compare.
    assistant_header_matches_finished_message_rendering: object


def locate_document_tokens(tokenizer, condition, prompt_text, document_text, suffix_text, label):
    """Tokenizes prompt + document + suffix and returns (full_token_ids, document_start_index,
    document_end_index) after checking that the prompt tokens are an exact prefix, the suffix tokens an exact
    tail, and the document tokens decode back to the document text apart from whitespace at the edges."""
    prompt_token_ids = tokenize_text(tokenizer, prompt_text)
    suffix_token_ids = tokenize_text(tokenizer, suffix_text) if suffix_text else []
    full_token_ids = tokenize_text(tokenizer, prompt_text + document_text + suffix_text)

    if full_token_ids[:len(prompt_token_ids)] != prompt_token_ids:
        raise DocumentTokenLocationError(
            f"For {label}, the tokens of the prompt (condition {condition['condition_id']}) are not an exact "
            f"prefix of the tokens of prompt + document, so the document's first token would merge with the "
            f"prompt. Prompt ends with {prompt_text[-80:]!r}; document starts with {document_text[:40]!r}.")
    document_start_index = len(prompt_token_ids)
    document_end_index = len(full_token_ids) - len(suffix_token_ids)
    if suffix_token_ids and full_token_ids[document_end_index:] != suffix_token_ids:
        raise DocumentTokenLocationError(
            f"For {label}, the tokens after the document are not an exact tail of the sequence "
            f"(suffix text {suffix_text!r}); the document's last token would merge with them.")
    if document_end_index <= document_start_index:
        raise DocumentTokenLocationError(f"No document content tokens found for {label}")

    decoded_content = tokenizer.decode(full_token_ids[document_start_index:document_end_index],
                                       skip_special_tokens=False)
    if decoded_content.strip() != document_text:
        mismatch_position = next((position for position, (left, right)
                                  in enumerate(zip(decoded_content.strip(), document_text)) if left != right),
                                 min(len(decoded_content.strip()), len(document_text)))
        raise DocumentTokenLocationError(
            f"For {label}, the document content tokens do not decode back to the document text "
            f"(first difference at character {mismatch_position}: decoded "
            f"{decoded_content.strip()[mismatch_position:mismatch_position + 60]!r} vs original "
            f"{document_text[mismatch_position:mismatch_position + 60]!r}).")
    return full_token_ids, document_start_index, document_end_index


def tokenize_condition_and_document(tokenizer, condition, generation_prompt, document_content, model_name=None):
    """Instruct model type. Returns the full token ids for (condition, document) and the index range of the
    document content tokens. Position i of the full sequence is predicted by the hidden state at position i - 1.

    Verified here: the system prompt survives the template; the tokens of the prompt are an exact prefix of
    the full sequence (so the first document token is scored given the assistant header and nothing leaks
    in from the document); the tokens after the document are an exact tail; the content tokens decode back
    to the document text apart from whitespace at the edges.
    """
    label = model_label(tokenizer, model_name)
    document_text = document_content.strip()
    if not document_text:
        raise DocumentTokenLocationError("Document content is empty")
    prompt_text = render_prompt_text_with_generation_header(tokenizer, condition, generation_prompt, model_name)
    suffix_text, text_before_content = render_text_after_assistant_message(
        tokenizer, condition, generation_prompt, document_text, model_name)
    full_token_ids, document_start_index, document_end_index = locate_document_tokens(
        tokenizer, condition, prompt_text, document_text, suffix_text, label)
    return TokenizedConditionAndDocument(
        full_token_ids=full_token_ids, document_start_index=document_start_index,
        document_end_index=document_end_index, prompt_text=prompt_text, suffix_text=suffix_text,
        assistant_header_matches_finished_message_rendering=(text_before_content == prompt_text))


# ---------------------------------------------------------------------------------------------------
# Base model type: plain-text author preamble, no chat template
# ---------------------------------------------------------------------------------------------------

class AuthorDescriptionNotInRenderedPromptError(RuntimeError):
    pass


def render_base_model_preamble(condition, generation_prompt, framing: ModelTypeFraming = None):
    """The preamble text (no beginning-of-sequence token, no trailing separator) for one condition. Pass the
    loaded `framing` when rendering many times; it is read from disk when omitted."""
    if "author_description" not in condition:
        raise ValueError(f"Condition {condition['condition_id']} has no author_description field; build the "
                         f"conditions with model_type={MODEL_TYPE_BASE!r}")
    framing = framing or load_model_type_framing()
    author_description = condition["author_description"]
    if author_description is None:
        template = framing.base_preamble_template_without_author
    else:
        template = framing.base_preamble_template_with_author
        if "{author_description}" not in template:
            raise ValueError("Base-model preamble template with author has no {author_description} slot")
        template = template.replace("{author_description}", author_description)
    if "{generation_prompt}" not in template:
        raise ValueError("Base-model preamble template has no {generation_prompt} slot")
    return template.replace("{generation_prompt}", generation_prompt)


def render_base_model_prompt_text(tokenizer, condition, generation_prompt, model_name=None,
                                  framing: ModelTypeFraming = None):
    """Beginning-of-sequence token text (if the tokenizer has one) + preamble + blank line. Raises if the
    condition's author description is missing from the result."""
    preamble = render_base_model_preamble(condition, generation_prompt, framing)
    prompt_text = (tokenizer.bos_token or "") + preamble + BASE_MODEL_PREAMBLE_AND_DOCUMENT_SEPARATOR
    author_description = condition["author_description"]
    if author_description is not None and author_description not in prompt_text:
        raise AuthorDescriptionNotInRenderedPromptError(
            f"The author description of condition {condition['condition_id']} is not in the rendered "
            f"base-model prompt for {model_label(tokenizer, model_name)}. Rendered text:\n{prompt_text}")
    return prompt_text


def tokenize_condition_and_document_for_base_model(tokenizer, condition, generation_prompt, document_content,
                                                   model_name=None, framing: ModelTypeFraming = None):
    """Base model counterpart of tokenize_condition_and_document: the prompt is the plain-text author preamble,
    and nothing follows the document. Same checks except the chat-template-only ones (system prompt survives
    the template, finished-message suffix); instead the author description must be in the prompt."""
    label = model_label(tokenizer, model_name)
    document_text = document_content.strip()
    if not document_text:
        raise DocumentTokenLocationError("Document content is empty")
    prompt_text = render_base_model_prompt_text(tokenizer, condition, generation_prompt, model_name, framing)
    full_token_ids, document_start_index, document_end_index = locate_document_tokens(
        tokenizer, condition, prompt_text, document_text, "", label)
    return TokenizedConditionAndDocument(
        full_token_ids=full_token_ids, document_start_index=document_start_index,
        document_end_index=document_end_index, prompt_text=prompt_text, suffix_text="",
        assistant_header_matches_finished_message_rendering=None)


def tokenize_condition_and_document_for_model_type(tokenizer, condition, generation_prompt, document_content,
                                                   model_name=None, model_type=MODEL_TYPE_INSTRUCT,
                                                   framing: ModelTypeFraming = None):
    check_model_type(model_type)
    if model_type == MODEL_TYPE_BASE:
        return tokenize_condition_and_document_for_base_model(
            tokenizer, condition, generation_prompt, document_content, model_name, framing)
    return tokenize_condition_and_document(tokenizer, condition, generation_prompt, document_content, model_name)


def tokenize_all_conditions_for_document(tokenizer, document, conditions, model_name,
                                         model_type=MODEL_TYPE_INSTRUCT, framing: ModelTypeFraming = None):
    """Tokenizes the document under every condition and asserts the document token ids are identical.
    Returns (tokenized sequences in condition order, the document token ids). `document` needs `document_id`,
    `content` and `generation_prompt`."""
    if model_type == MODEL_TYPE_BASE and framing is None:
        framing = load_model_type_framing()
    tokenized_sequences = []
    reference_document_token_ids = None
    for condition in conditions:
        tokenized = tokenize_condition_and_document_for_model_type(
            tokenizer, condition, document["generation_prompt"], document["content"], model_name, model_type, framing)
        document_token_ids = tokenized.full_token_ids[tokenized.document_start_index:tokenized.document_end_index]
        if reference_document_token_ids is None:
            reference_document_token_ids = document_token_ids
        elif document_token_ids != reference_document_token_ids:
            raise AssertionError(
                f"Document {document['document_id']}: document token ids under condition "
                f"{condition['condition_id']} differ from those under {conditions[0]['condition_id']}")
        tokenized_sequences.append(tokenized)
    return tokenized_sequences, reference_document_token_ids


def decode_each_token(tokenizer, token_ids):
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


def count_tokens_for_scoring_inputs(tokenizer, documents, conditions, model_type, model_name=None,
                                    framing: ModelTypeFraming = None):
    """Token totals for a dry run, without tokenizing every (document, condition) pair in full. Returns
    {"document_tokens_scored": ..., "total_tokens_through_model": ...}: the first is the sum over documents of
    document tokens times the number of conditions; the second adds each prompt's tokens for every (document,
    condition) pair. The end-of-turn suffix (a token or two per pair) is not counted."""
    check_model_type(model_type)
    if model_type == MODEL_TYPE_BASE and framing is None:
        framing = load_model_type_framing()
    document_tokens_scored = 0
    prompt_tokens = 0
    for document in documents:
        document_tokens_scored += len(conditions) * len(tokenize_text(tokenizer, document["content"].strip()))
        for condition in conditions:
            if model_type == MODEL_TYPE_BASE:
                prompt_text = render_base_model_prompt_text(tokenizer, condition, document["generation_prompt"],
                                                            model_name, framing)
            else:
                prompt_text = render_prompt_text_with_generation_header(
                    tokenizer, condition, document["generation_prompt"], model_name)
            prompt_tokens += len(tokenize_text(tokenizer, prompt_text))
    return {"document_tokens_scored": document_tokens_scored,
            "total_tokens_through_model": document_tokens_scored + prompt_tokens}
