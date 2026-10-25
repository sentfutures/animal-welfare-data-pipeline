"""Offline tests for document token location and base-model prompt rendering, using small offline tokenizers with
different chat templates (no weights, no downloads). Skipped when `tokenizers` or `transformers` is missing.

Covers the bugs that would invalidate the experiment: off-by-one in which tokens are scored, the chat template
dropping the system prompt, a prompt merging with the first document token, and document tokens differing
across conditions.
"""

import shutil

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

import tiny_model_and_tokenizer_helpers_for_log_probability_persona_check as tiny  # noqa: E402
from evals.log_probability_persona_check import persona_sets, prompt_rendering  # noqa: E402

SHORT_DOCUMENT_TEXTS = [
    "The Gulf shrimp season opened early. Regulators weighed the cost of new nets against animal welfare.",
    "Farmers met on Tuesday to discuss rodent control; the AI suggested traps, then cautioned about cruelty.",
    "A short note on pigeons in the plaza, and the council's decision to leave the feeders alone."]
TRAINING_TEXTS = SHORT_DOCUMENT_TEXTS + [
    "You are an AI assistant who moralizes about animals. Write a document from a world where AI is common.",
    "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 .,;:!?'\"-()/ The tone is neutral."]
GENERATION_PROMPT = "Write a short document about shrimp."


@pytest.fixture(scope="module")
def persona_set():
    return persona_sets.load_persona_set("preachiness")


@pytest.fixture(scope="module")
def instruct_conditions(persona_set):
    return prompt_rendering.build_condition_records(persona_set)


@pytest.fixture(scope="module")
def base_conditions(persona_set):
    return prompt_rendering.build_condition_records(persona_set, prompt_rendering.MODEL_TYPE_BASE)


def make_tokenizer(chat_template=tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN):
    return tiny.build_small_offline_tokenizer(chat_template, TRAINING_TEXTS * 20)


def make_condition(system_prompt="You are an AI assistant who is a test persona.", condition_id="test_condition"):
    return {"condition_id": condition_id, "condition_group": "preachy", "persona_name": "preachy",
            "rewording_index": 1, "system_prompt": system_prompt}


# ---------------------------------------------------------------- which tokens are document tokens


@pytest.mark.parametrize("chat_template", [tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN,
                                           tiny.TEMPLATE_WITH_SYSTEM_TURN_AND_PLAIN_HEADER])
def test_document_token_range_has_no_off_by_one(chat_template):
    tokenizer = make_tokenizer(chat_template)
    condition = make_condition()
    document_text = SHORT_DOCUMENT_TEXTS[0]
    tokenized = prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                                 document_text, "tiny test tokenizer")
    ids, start, end = tokenized.full_token_ids, tokenized.document_start_index, tokenized.document_end_index
    assert tokenizer.decode(ids[start:end]) == document_text
    assert tokenizer.decode(ids[:start]) == tokenized.prompt_text
    assert tokenized.prompt_text.endswith("<|turn>model\n") or tokenized.prompt_text.endswith("<channel|>")
    assert tokenizer.convert_ids_to_tokens(ids[start - 1]) in ("<channel|>", "Ċ")
    assert tokenizer.convert_ids_to_tokens(ids[end]) == "<turn|>"
    assert "<turn|>" not in tokenizer.decode(ids[start:end])
    assert tokenizer.convert_ids_to_tokens(ids[start]) != "<turn|>"
    assert condition["system_prompt"] in tokenizer.decode(ids[:start])


def test_gemma_like_header_differs_from_finished_message_rendering_and_generation_header_is_used():
    tokenizer = make_tokenizer(tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN)
    tokenized = prompt_rendering.tokenize_condition_and_document(
        tokenizer, make_condition(), GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "tiny test tokenizer")
    assert not tokenized.assistant_header_matches_finished_message_rendering
    assert tokenized.prompt_text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
    assert tokenized.suffix_text == "<turn|>\n"


def test_no_system_prompt_condition_renders_without_system_turn():
    tokenizer = make_tokenizer()
    condition = make_condition(system_prompt=None, condition_id="no_system_prompt")
    tokenized = prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                                 SHORT_DOCUMENT_TEXTS[0], "tiny test tokenizer")
    assert "<|turn>system" not in tokenized.prompt_text


def test_document_edge_whitespace_is_ignored_consistently():
    tokenizer = make_tokenizer()
    condition = make_condition()
    plain = prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                             SHORT_DOCUMENT_TEXTS[0], "m")
    padded = prompt_rendering.tokenize_condition_and_document(
        tokenizer, condition, GENERATION_PROMPT, "\n\n  " + SHORT_DOCUMENT_TEXTS[0] + "  \n", "m")
    assert plain.full_token_ids == padded.full_token_ids


def test_empty_document_is_rejected():
    with pytest.raises(prompt_rendering.DocumentTokenLocationError, match="empty"):
        prompt_rendering.tokenize_condition_and_document(make_tokenizer(), make_condition(), GENERATION_PROMPT, " \n")


def test_header_that_merges_with_first_document_token_is_rejected():
    tokenizer = make_tokenizer(tiny.TEMPLATE_WHOSE_HEADER_MERGES_WITH_DOCUMENT_START)
    with pytest.raises(prompt_rendering.DocumentTokenLocationError, match="prefix"):
        prompt_rendering.tokenize_condition_and_document(
            tokenizer, make_condition(), GENERATION_PROMPT, "the shrimp season opened early.", "tiny merging tokenizer")


# ---------------------------------------------------------------- system prompt survives the chat template


def test_template_that_silently_drops_system_prompt_raises_with_model_name_and_rendered_text():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES)
    condition = make_condition()
    with pytest.raises(prompt_rendering.SystemPromptNotInRenderedPromptError) as error_info:
        prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                         SHORT_DOCUMENT_TEXTS[0], "example-org/dropping-model")
    message = str(error_info.value)
    assert "example-org/dropping-model" in message and "dropped the system prompt" in message
    assert GENERATION_PROMPT in message
    assert condition["system_prompt"] not in message.split("Rendered text:")[1]


def test_template_that_rejects_system_role_raises_clear_error():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_REJECTS_SYSTEM_ROLE)
    with pytest.raises(prompt_rendering.SystemPromptNotInRenderedPromptError, match="example-org/rejecting-model"):
        prompt_rendering.tokenize_condition_and_document(tokenizer, make_condition(), GENERATION_PROMPT,
                                                         SHORT_DOCUMENT_TEXTS[0], "example-org/rejecting-model")


def test_template_that_drops_system_prompt_is_fine_for_the_no_system_prompt_condition():
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES)
    condition = make_condition(system_prompt=None, condition_id="no_system_prompt")
    prompt_rendering.tokenize_condition_and_document(tokenizer, condition, GENERATION_PROMPT,
                                                     SHORT_DOCUMENT_TEXTS[0], "m")


def test_tokenizing_all_conditions_aborts_when_template_drops_system_prompts(instruct_conditions):
    tokenizer = make_tokenizer(tiny.TEMPLATE_THAT_SILENTLY_DROPS_SYSTEM_MESSAGES)
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[0], "generation_prompt": GENERATION_PROMPT}
    with pytest.raises(prompt_rendering.SystemPromptNotInRenderedPromptError):
        prompt_rendering.tokenize_all_conditions_for_document(tokenizer, document, instruct_conditions, "m")


# ---------------------------------------------------------------- document token ids identical across conditions


def test_document_token_ids_are_identical_across_all_72_conditions(instruct_conditions):
    tokenizer = make_tokenizer()
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[1], "generation_prompt": GENERATION_PROMPT}
    tokenized_sequences, document_token_ids = prompt_rendering.tokenize_all_conditions_for_document(
        tokenizer, document, instruct_conditions, "m")
    assert len(tokenized_sequences) == 72
    for sequence in tokenized_sequences:
        assert sequence.full_token_ids[sequence.document_start_index:sequence.document_end_index] == document_token_ids
    assert len({sequence.document_start_index for sequence in tokenized_sequences}) > 1  # prompts differ in length


def test_mismatched_document_token_ids_across_conditions_are_caught(monkeypatch, instruct_conditions):
    tokenizer = make_tokenizer()
    real_function = prompt_rendering.tokenize_condition_and_document

    def corrupt_one_condition(tokenizer_argument, condition, generation_prompt, content, model_name=None):
        tokenized = real_function(tokenizer_argument, condition, generation_prompt, content, model_name)
        if condition["condition_id"] == "generic":
            tokenized.full_token_ids[tokenized.document_start_index + 2] += 1
        return tokenized

    monkeypatch.setattr(prompt_rendering, "tokenize_condition_and_document", corrupt_one_condition)
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[1], "generation_prompt": GENERATION_PROMPT}
    with pytest.raises(AssertionError, match="differ"):
        prompt_rendering.tokenize_all_conditions_for_document(tokenizer, document, instruct_conditions[:3], "m")


# ---------------------------------------------------------------- base model type


def test_base_prompt_text_for_each_condition_type(base_conditions):
    tokenizer = make_tokenizer()
    by_id = {condition["condition_id"]: condition for condition in base_conditions}
    expected_request = f"in response to this request: {GENERATION_PROMPT}\n\n"
    prompt_texts = {condition_id: prompt_rendering.render_base_model_prompt_text(
        tokenizer, by_id[condition_id], GENERATION_PROMPT)
        for condition_id in ("no_system_prompt", "generic", "preachy__rewording_01",
                             "animal_husbandry_expertise__rewording_01", "proportionate_welfare__rewording_01")}
    assert prompt_texts["no_system_prompt"] == (
        f"<bos>The following text was written in response to this request: {GENERATION_PROMPT}\n\n")
    assert prompt_texts["generic"] == f"<bos>The following text was written by an AI assistant, {expected_request}"
    for condition_id in ("preachy__rewording_01", "animal_husbandry_expertise__rewording_01",
                         "proportionate_welfare__rewording_01"):
        author_description = by_id[condition_id]["author_description"]
        assert author_description.startswith("an AI assistant who ")
        assert prompt_texts[condition_id] == (
            f"<bos>The following text was written by {author_description}, {expected_request}")
        assert "<|turn>" not in prompt_texts[condition_id]


def test_base_format_works_with_tokenizer_without_chat_template_or_bos_token(base_conditions):
    tokenizer = make_tokenizer(chat_template=None)
    assert tokenizer.chat_template is None
    condition = base_conditions[2]
    tokenized = prompt_rendering.tokenize_condition_and_document_for_model_type(
        tokenizer, condition, GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m", prompt_rendering.MODEL_TYPE_BASE)
    assert tokenized.prompt_text.startswith("<bos>The following text was written by an AI assistant who ")
    tokenizer.bos_token = None
    without_bos = prompt_rendering.tokenize_condition_and_document_for_base_model(
        tokenizer, condition, GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m")
    assert without_bos.prompt_text.startswith("The following text was written by an AI assistant who ")
    assert tokenizer.decode(without_bos.full_token_ids) == without_bos.prompt_text + SHORT_DOCUMENT_TEXTS[0]


def test_base_format_prefix_decode_and_no_suffix_checks(base_conditions):
    tokenizer = make_tokenizer(chat_template=None)
    document_text = SHORT_DOCUMENT_TEXTS[1]
    tokenized = prompt_rendering.tokenize_condition_and_document_for_base_model(
        tokenizer, base_conditions[1], GENERATION_PROMPT, f"  {document_text}\n", "m")
    ids, start, end = tokenized.full_token_ids, tokenized.document_start_index, tokenized.document_end_index
    assert tokenized.prompt_text.endswith("\n\n") and tokenized.suffix_text == ""
    assert tokenizer.decode(ids[:start]) == tokenized.prompt_text
    assert tokenizer.decode(ids[start:end]) == document_text
    assert end == len(ids)  # nothing after the document, no end-of-sequence token
    assert tokenized.assistant_header_matches_finished_message_rendering is None


def copy_framing_with_replaced_file(tmp_path, file_name, new_text):
    framing_directory = tmp_path / "model_type_framing"
    shutil.copytree(persona_sets.PERSONA_CHECKS_PROMPT_DIRECTORY / persona_sets.MODEL_TYPE_FRAMING_DIRECTORY_NAME,
                    framing_directory)
    (framing_directory / file_name).write_text(new_text, encoding="utf-8")
    return persona_sets.read_framing_files(framing_directory, [])


def test_base_format_rejects_prompt_that_merges_with_document_start(tmp_path, monkeypatch, base_conditions):
    tokenizer = make_tokenizer(chat_template=None)
    framing = copy_framing_with_replaced_file(
        tmp_path, persona_sets.BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME,
        "Written by {author_description} for {generation_prompt}\n")
    monkeypatch.setattr(prompt_rendering, "BASE_MODEL_PREAMBLE_AND_DOCUMENT_SEPARATOR", " ")
    with pytest.raises(prompt_rendering.DocumentTokenLocationError, match="prefix"):
        prompt_rendering.tokenize_condition_and_document_for_base_model(
            tokenizer, base_conditions[1], GENERATION_PROMPT, "the shrimp season opened early.", "m", framing)


def test_base_format_requires_author_description_in_prompt(tmp_path, monkeypatch, base_conditions, instruct_conditions):
    tokenizer = make_tokenizer(chat_template=None)
    framing = copy_framing_with_replaced_file(
        tmp_path, persona_sets.BASE_MODEL_PREAMBLE_TEMPLATE_WITH_AUTHOR_FILE_NAME,
        "The following text was written in response to this request: {generation_prompt}")
    with pytest.raises(ValueError, match="no {author_description} slot"):
        prompt_rendering.tokenize_condition_and_document_for_base_model(
            tokenizer, base_conditions[1], GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m", framing)
    # Defensive check on the rendered text itself, in case a future change drops the author.
    monkeypatch.setattr(prompt_rendering, "render_base_model_preamble",
                        lambda condition, generation_prompt, framing=None: generation_prompt)
    with pytest.raises(prompt_rendering.AuthorDescriptionNotInRenderedPromptError):
        prompt_rendering.tokenize_condition_and_document_for_base_model(
            tokenizer, base_conditions[1], GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m")
    monkeypatch.undo()
    with pytest.raises(ValueError, match="no author_description"):  # instruct conditions in a base run
        prompt_rendering.tokenize_condition_and_document_for_base_model(
            tokenizer, instruct_conditions[2], GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m")


def test_base_format_document_token_ids_are_identical_across_all_72_conditions(base_conditions):
    tokenizer = make_tokenizer(chat_template=None)
    document = {"document_id": "doc", "content": SHORT_DOCUMENT_TEXTS[2], "generation_prompt": GENERATION_PROMPT}
    tokenized_sequences, document_token_ids = prompt_rendering.tokenize_all_conditions_for_document(
        tokenizer, document, base_conditions, "m", prompt_rendering.MODEL_TYPE_BASE)
    assert len(tokenized_sequences) == 72
    for sequence in tokenized_sequences:
        assert sequence.full_token_ids[sequence.document_start_index:sequence.document_end_index] == document_token_ids


def test_unknown_model_type_is_rejected(instruct_conditions):
    with pytest.raises(ValueError, match="Unknown model type"):
        prompt_rendering.tokenize_condition_and_document_for_model_type(
            make_tokenizer(), instruct_conditions[2], GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], "m", "chat")


# ---------------------------------------------------------------- dry-run token counts


@pytest.mark.parametrize("model_type, chat_template", [("instruct", tiny.GEMMA_LIKE_TEMPLATE_WITH_SYSTEM_TURN),
                                                        ("base", None)])
def test_token_count_matches_full_tokenization_apart_from_the_suffix(model_type, chat_template, persona_set):
    tokenizer = make_tokenizer(chat_template)
    conditions = prompt_rendering.build_condition_records(persona_set, model_type)[:6]
    documents = [{"document_id": f"d{i}", "content": text, "generation_prompt": GENERATION_PROMPT}
                 for i, text in enumerate(SHORT_DOCUMENT_TEXTS[:2])]
    counts = prompt_rendering.count_tokens_for_scoring_inputs(tokenizer, documents, conditions, model_type, "m")
    expected_total, expected_scored = 0, 0
    for document in documents:
        for condition in conditions:
            tokenized = prompt_rendering.tokenize_condition_and_document_for_model_type(
                tokenizer, condition, document["generation_prompt"], document["content"], "m", model_type)
            expected_total += tokenized.document_end_index
            expected_scored += tokenized.document_end_index - tokenized.document_start_index
    assert counts == {"document_tokens_scored": expected_scored, "total_tokens_through_model": expected_total}


# ---------------------------------------------------------------- real Gemma 4 tokenizer (opt in)


@pytest.mark.slow
def test_real_gemma_tokenizer_chat_template_keeps_system_prompt_and_locates_document_tokens(instruct_conditions):
    """Needs the tokenizer files in the local Hugging Face cache (never downloads); run with --run-slow."""
    from evals.log_probability_persona_check.build_scoring_inputs import load_tokenizer_from_local_cache
    from evals.log_probability_persona_check.scoring_model_defaults import DEFAULT_SCORING_MODELS
    default = DEFAULT_SCORING_MODELS["instruct"]
    tokenizer = load_tokenizer_from_local_cache(default["model_name"], default["model_revision"])
    if tokenizer is None:
        pytest.skip("Gemma 4 tokenizer files are not in the local Hugging Face cache")
    reference_ids = None
    for condition in (instruct_conditions[0], instruct_conditions[1], instruct_conditions[2], instruct_conditions[-1]):
        tokenized = prompt_rendering.tokenize_condition_and_document(
            tokenizer, condition, GENERATION_PROMPT, SHORT_DOCUMENT_TEXTS[0], default["model_name"])
        if condition["system_prompt"]:
            assert condition["system_prompt"] in tokenized.prompt_text
            assert tokenized.prompt_text.startswith("<bos><|turn>system\n")
        else:
            assert "<|turn>system" not in tokenized.prompt_text
        assert tokenized.prompt_text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
        assert tokenized.suffix_text == "<turn|>\n"
        document_ids = tokenized.full_token_ids[tokenized.document_start_index:tokenized.document_end_index]
        reference_ids = reference_ids or document_ids
        assert document_ids == reference_ids


# ---------------------------------------------------------------- per-model chat template keyword arguments

# Same logic as the Qwen3 template: the generation prompt ends with an empty think block only when the caller passes
# enable_thinking=False, and a finished final assistant message always renders the empty block before its content.
QWEN3_LIKE_TEMPLATE = (
    "{% for message in messages %}<|turn>{{ message['role'] }}\n{{ message['content'] | trim }}<turn|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|turn>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<|channel>\n\n<channel|>\n\n{% endif %}{% endif %}")


def test_chat_template_keyword_arguments_are_applied_for_qwen3_and_for_no_other_model():
    from evals.log_probability_persona_check.scoring_model_defaults import chat_template_keyword_arguments_for_model
    assert chat_template_keyword_arguments_for_model("Qwen/Qwen3-32B") == {"enable_thinking": False}
    assert chat_template_keyword_arguments_for_model("google/gemma-4-12b-it") == {}
    returned = chat_template_keyword_arguments_for_model("Qwen/Qwen3-32B")
    returned["enable_thinking"] = True  # a copy: the table is not changed
    assert chat_template_keyword_arguments_for_model("Qwen/Qwen3-32B") == {"enable_thinking": False}


def test_prompt_ends_with_empty_think_block_for_qwen3_and_is_unchanged_for_other_models():
    tokenizer = make_tokenizer(QWEN3_LIKE_TEMPLATE)
    condition = make_condition()
    qwen = prompt_rendering.render_prompt_text_with_generation_header(tokenizer, condition, GENERATION_PROMPT, "Qwen/Qwen3-32B")
    other = prompt_rendering.render_prompt_text_with_generation_header(tokenizer, condition, GENERATION_PROMPT, "other/model")
    assert qwen == other + "<|channel>\n\n<channel|>\n\n"
    assert condition["system_prompt"] in qwen


def test_document_tokens_are_located_after_the_think_block_for_qwen3_and_identical_across_conditions():
    tokenizer = make_tokenizer(QWEN3_LIKE_TEMPLATE)
    conditions = [make_condition(), make_condition(None, "no_system_prompt"),
                  make_condition("You are an AI assistant.", "generic")]
    sequences, document_token_ids = prompt_rendering.tokenize_all_conditions_for_document(
        tokenizer, {"document_id": "d", "content": SHORT_DOCUMENT_TEXTS[0], "generation_prompt": GENERATION_PROMPT},
        conditions, "Qwen/Qwen3-32B")
    assert tokenizer.decode(document_token_ids) == SHORT_DOCUMENT_TEXTS[0]
    for sequence in sequences:
        assert sequence.prompt_text.endswith("<channel|>\n\n")
        assert sequence.assistant_header_matches_finished_message_rendering in (True, False)
    assert "<|turn>system" not in sequences[1].prompt_text  # no default system prompt appears for the empty condition
