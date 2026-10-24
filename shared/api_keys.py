"""One place that decides where an API key comes from.

Rules (the repo must be usable by other people, including labs on servers where keys
arrive as environment variables, so nothing outside the project is ever searched):

1. The only file consulted is ``<repo root>/.env``. The repo root is the directory that
   contains ``shared/``, found from this file's own location: never by searching parent
   folders and never from the current working directory.
2. If that file sets the variable to a non-empty value (whitespace stripped), it is used,
   it overrides the environment, and it is written to ``os.environ`` so code that reads the
   environment keeps working.
3. Otherwise the environment variable is used if non-empty (an empty value in the file
   never clobbers a good environment value).
4. If neither provides it, a required key raises ``MissingApiKeyError`` naming the path.
   An optional key returns None.

The source is printed once per variable per process. The key value is never printed.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent

_sources_already_announced: set[str] = set()


class MissingApiKeyError(RuntimeError, KeyError):
    """No key for a variable in the repo's .env or the environment.

    Subclasses both RuntimeError and KeyError because callers have historically raised
    either for a missing key."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def repo_dotenv_path() -> Path:
    return (REPO_ROOT / ".env").resolve()


def _announce_once(variable_name: str, message: str) -> None:
    if variable_name in _sources_already_announced:
        return
    _sources_already_announced.add(variable_name)
    print(message)


def load_optional_api_key(variable_name: str) -> str | None:
    """Return the key for ``variable_name`` following the module rules, or None if absent."""
    dotenv_path = repo_dotenv_path()
    dotenv_exists = dotenv_path.is_file()
    value_in_file = ""
    if dotenv_exists:
        value_in_file = (dotenv_values(dotenv_path).get(variable_name) or "").strip()
    if value_in_file:
        os.environ[variable_name] = value_in_file
        _announce_once(variable_name, f"{variable_name} from {dotenv_path} (value not printed)")
        return value_in_file

    value_in_environment = os.environ.get(variable_name, "").strip()
    if value_in_environment:
        if variable_name not in _sources_already_announced:
            if dotenv_exists:
                print(f"{dotenv_path} has no {variable_name}; using the key from the environment")
            print(f"{variable_name} from the environment")
            _sources_already_announced.add(variable_name)
        return value_in_environment
    return None


def load_api_key(variable_name: str) -> str:
    """Return the key for ``variable_name``; raise MissingApiKeyError if there is none."""
    key = load_optional_api_key(variable_name)
    if key is None:
        raise MissingApiKeyError(
            f"{variable_name} is not set. Fill in {variable_name} in {repo_dotenv_path()} "
            f"(see .env.example in the repo root) or set it in the environment.")
    return key
