"""Mechanical helpers shared by the stages that ask for a schema-constrained JSON answer: reading a template and schema,
rejecting an answer that stopped early, and turning a parse failure into the stage's own error. Each stage validates its own contents."""

import json
from pathlib import Path

DEFAULT_UNEXPECTED_JSON_MESSAGE = "answer is not the expected JSON ({error_type}: {error})"


def read_template_if_present(prompts_directory, template_filename: str) -> str | None:
    """The template's text, or None for a run whose prompts snapshot lacks the stage's files."""
    path = Path(prompts_directory) / template_filename
    return path.read_text(encoding="utf-8") if path.is_file() else None


def load_template_and_answer_schema(
    prompts_directory, template_filename: str, schema_filename: str
) -> tuple[str, dict] | None:
    """(template text, parsed answer schema), or None for a run whose prompts snapshot lacks the stage's files."""
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
    """``json.loads(raw or "")``, then ``extract`` if given; a caught exception becomes ``error_class(failure_message)``."""
    try:
        parsed = json.loads(raw or "")
        return extract(parsed) if extract is not None else parsed
    except caught_exceptions as error:
        raise error_class(
            failure_message.format(error_type=type(error).__name__, error=error)) from error
