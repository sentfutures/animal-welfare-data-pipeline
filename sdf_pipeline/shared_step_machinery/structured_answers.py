"""Small helpers shared by the layer-1 stages that ask a model for a schema-constrained JSON
answer (the implausibility screen, the description rewrite and the considerations selector).

Each stage keeps its own error class and its own validation of the answer's contents; what is
shared here is the mechanical part: reading a stage's template and answer schema from a prompts
directory, rejecting an answer that stopped before it finished, and turning a JSON parse failure
into the stage's own error.
"""

import json
from pathlib import Path

DEFAULT_UNEXPECTED_JSON_MESSAGE = "answer is not the expected JSON ({error_type}: {error})"


def read_template_if_present(prompts_directory, template_filename: str) -> str | None:
    """The template's text, or None for a run whose prompts snapshot predates the stage."""
    path = Path(prompts_directory) / template_filename
    return path.read_text(encoding="utf-8") if path.is_file() else None


def load_template_and_answer_schema(
    prompts_directory, template_filename: str, schema_filename: str
) -> tuple[str, dict] | None:
    """(template text, parsed answer schema), or None for a run whose prompts snapshot predates the stage."""
    template = read_template_if_present(prompts_directory, template_filename)
    if template is None:
        return None
    schema_path = Path(prompts_directory) / schema_filename
    return template, json.loads(schema_path.read_text(encoding="utf-8"))


def raise_if_answer_was_truncated(stop_reason, error_class: type[Exception]) -> None:
    """Raise error_class unless the model ended its turn normally."""
    if stop_reason != "end_turn":
        raise error_class(f"truncated answer (stop_reason={stop_reason})")


def parse_json_answer(
    raw,
    error_class: type[Exception],
    extract=None,
    caught_exceptions: tuple = (ValueError, KeyError, TypeError),
    failure_message: str = DEFAULT_UNEXPECTED_JSON_MESSAGE,
):
    """json.loads(raw or ""), passed through ``extract`` when given (which may index into the
    parsed answer). Any of ``caught_exceptions`` raised while parsing or extracting becomes
    ``error_class(failure_message)``, the message formatted with ``error_type`` and ``error``."""
    try:
        parsed = json.loads(raw or "")
        return extract(parsed) if extract is not None else parsed
    except caught_exceptions as error:
        raise error_class(
            failure_message.format(error_type=type(error).__name__, error=error)) from error
