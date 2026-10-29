"""Default scoring models for the log-probability persona check, one per model type, each at a pinned revision.

The defaults live here, at package level, rather than in each persona set, because the model is a property of the
scoring setup and not of the persona. A persona set may still override them with an optional ``default_models``
block in its ``persona_check_configuration.json`` (same shape as ``DEFAULT_SCORING_MODELS``). Precedence, highest
first: ``--model-name`` / ``--model-revision`` on the command line, the persona set's ``default_models``, then
the values below.

A revision is always required, so that a run can be repeated against the same weights and tokenizer. Passing
``--model-name`` for any model other than the one the defaults name therefore needs ``--model-revision`` too.
"""

import re

MODEL_TYPE_INSTRUCT = "instruct"
MODEL_TYPE_BASE = "base"
MODEL_TYPES = [MODEL_TYPE_INSTRUCT, MODEL_TYPE_BASE]

# instruct: instruction-tuned model, scored through its chat template with personas as system prompts.
# base: pretrained model with no chat template, scored with a plain-text author preamble naming the persona.
DEFAULT_SCORING_MODELS = {
    MODEL_TYPE_INSTRUCT: {
        "model_name": "google/gemma-4-12b-it",
        "model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    },
    # The pretrained Gemma 4 12B (Hugging Face redirects to google/gemma-4-12B; same architecture as the
    # instruct model).
    MODEL_TYPE_BASE: {
        "model_name": "google/gemma-4-12b",
        "model_revision": "023679ed352de9bb66cc873c9009ce3482585c08",
    },
}

# Per-model keyword arguments for `tokenizer.apply_chat_template`, keyed on the model name (instruct runs only; a
# base run has no chat template). Qwen3 is a hybrid thinking model: with `enable_thinking=False` its generation
# prompt ends with an empty think block ("<think>\n\n</think>\n\n"), which is also what its template writes before
# the content of a finished final assistant message, so the document is scored exactly where the model would start
# writing a non-thinking reply. The run configuration records the arguments used.
CHAT_TEMPLATE_KEYWORD_ARGUMENTS_BY_MODEL_NAME = {
    "Qwen/Qwen3-32B": {"enable_thinking": False},
    "Qwen/Qwen3-0.6B": {"enable_thinking": False},
}

COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def check_model_type(model_type):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type {model_type!r}; expected one of {MODEL_TYPES}")


def chat_template_keyword_arguments_for_model(model_name):
    """A copy of the model's extra `apply_chat_template` keyword arguments ({} for a model that needs none)."""
    return dict(CHAT_TEMPLATE_KEYWORD_ARGUMENTS_BY_MODEL_NAME.get(model_name, {}))


def check_model_revision_is_pinned(model_revision, context="model revision"):
    """A pinned revision is a full 40-character commit hash; branch names such as 'main' move."""
    if not isinstance(model_revision, str) or not COMMIT_HASH_PATTERN.match(model_revision):
        raise ValueError(f"The {context} must be a full 40-character lowercase hexadecimal commit hash, "
                         f"got {model_revision!r}")


def resolve_scoring_model(model_type, persona_set_default_models=None, model_name_override=None,
                          model_revision_override=None):
    """Returns (model_name, model_revision, source) where source is 'command line', 'persona set' or 'package
    default'. Raises ValueError if no pinned revision can be determined."""
    check_model_type(model_type)
    persona_set_default = (persona_set_default_models or {}).get(model_type)
    if persona_set_default is not None:
        base_choice, base_source = persona_set_default, "persona set"
    else:
        base_choice, base_source = DEFAULT_SCORING_MODELS[model_type], "package default"

    if model_name_override is None and model_revision_override is None:
        model_name, model_revision, source = base_choice["model_name"], base_choice["model_revision"], base_source
    elif model_name_override is not None and model_name_override != base_choice["model_name"]:
        # A different model: the default revision belongs to the default model, so a new one must be given.
        if model_revision_override is None:
            raise ValueError(f"--model-name {model_name_override} differs from the default {model_type} model "
                             f"{base_choice['model_name']}; pass --model-revision with a pinned commit hash too")
        model_name, model_revision, source = model_name_override, model_revision_override, "command line"
    else:
        model_name = model_name_override or base_choice["model_name"]
        model_revision = model_revision_override or base_choice["model_revision"]
        source = "command line" if model_revision_override else base_source
    check_model_revision_is_pinned(model_revision)
    return model_name, model_revision, source
