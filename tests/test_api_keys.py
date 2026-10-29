"""shared/api_keys.py: where a key comes from. Temporary folders and fake keys only."""

import argparse
import os

import pytest

from shared import api_keys

REAL_SET_API_KEY_SOURCE_MODE = api_keys.set_api_key_source_mode   # conftest replaces the module's own with a no-op
REAL_NEAREST_DOTENV_SEARCH = api_keys._nearest_dotenv_path_searching_upward   # conftest confines it to the repo root

VARIABLE = "FAKE_PROJECT_API_KEY"
FILE_KEY = "fake-key-from-the-dotenv-file-0001"
ENVIRONMENT_KEY = "fake-key-from-the-environment-0002"
PARENT_FILE_KEY = "fake-key-from-a-parent-folder-dotenv-0003"


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    """A repo root in the default key-source mode, with no key in the environment."""
    root = tmp_path / "outer_folder" / "repo_root"
    root.mkdir(parents=True)
    monkeypatch.setattr(api_keys, "REPO_ROOT", root)
    monkeypatch.setattr(api_keys, "_sources_already_announced", set())
    monkeypatch.setattr(api_keys, "_api_key_source_mode", api_keys.REPO_ROOT_DOTENV_ONLY)
    monkeypatch.setattr(api_keys, "set_api_key_source_mode", REAL_SET_API_KEY_SOURCE_MODE)
    monkeypatch.setattr(api_keys, "_nearest_dotenv_path_searching_upward", REAL_NEAREST_DOTENV_SEARCH)
    monkeypatch.delenv(VARIABLE, raising=False)
    return root


@pytest.fixture
def search_upward_mode(repo_root):
    api_keys.set_api_key_source_mode(api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT)


# Default mode: only <repo root>/.env

def test_default_reads_the_key_in_the_file_overriding_the_environment_and_sets_it(repo_root, monkeypatch, capsys):
    (repo_root / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == FILE_KEY
    assert os.environ[VARIABLE] == FILE_KEY
    assert capsys.readouterr().out.strip() == (
        f"{VARIABLE} from {(repo_root / '.env').resolve()} (value not printed)")


def test_default_ignores_the_environment_when_the_file_is_missing(repo_root, monkeypatch):
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    with pytest.raises(api_keys.MissingApiKeyError) as error:
        api_keys.load_api_key(VARIABLE)
    message = str(error.value)
    assert str((repo_root / ".env").resolve()) in message and VARIABLE in message
    assert "does not exist" in message
    assert "--find-dotenv" in message


@pytest.mark.parametrize("file_text", [f"{VARIABLE}=\n", "SOME_OTHER_VARIABLE=1\n", f'{VARIABLE}="   "\n'])
def test_default_ignores_the_environment_when_the_file_lacks_a_non_empty_key(repo_root, monkeypatch, file_text):
    (repo_root / ".env").write_text(file_text)
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    with pytest.raises(api_keys.MissingApiKeyError) as error:
        api_keys.load_api_key(VARIABLE)
    message = str(error.value)
    assert str((repo_root / ".env").resolve()) in message and VARIABLE in message
    assert "no non-empty" in message
    assert "--find-dotenv" in message
    assert os.environ[VARIABLE] == ENVIRONMENT_KEY


def test_default_optional_key_is_none_not_an_error_even_with_the_environment_set(repo_root, monkeypatch):
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_optional_api_key(VARIABLE) is None


def test_default_ignores_a_dotenv_in_a_parent_of_the_repo_root(repo_root):
    (repo_root.parent / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    with pytest.raises(api_keys.MissingApiKeyError):
        api_keys.load_api_key(VARIABLE)


def test_default_ignores_a_dotenv_in_the_current_directory(repo_root, tmp_path, monkeypatch):
    working_directory = tmp_path / "somewhere_else"
    working_directory.mkdir()
    (working_directory / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    monkeypatch.chdir(working_directory)
    with pytest.raises(api_keys.MissingApiKeyError):
        api_keys.load_api_key(VARIABLE)


def test_key_value_never_appears_in_output(repo_root, monkeypatch, capsys):
    (repo_root / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    api_keys.load_api_key(VARIABLE)
    api_keys.set_api_key_source_mode(api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT)
    monkeypatch.setattr(api_keys, "_sources_already_announced", set())
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    api_keys.load_api_key(VARIABLE)
    captured = capsys.readouterr()
    for secret in (FILE_KEY, ENVIRONMENT_KEY):
        assert secret not in captured.out and secret not in captured.err


def test_source_is_printed_once_per_variable(search_upward_mode, monkeypatch, capsys):
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    api_keys.load_api_key(VARIABLE)
    api_keys.load_api_key(VARIABLE)
    assert capsys.readouterr().out.count(f"{VARIABLE} from the environment") == 1


# --find-dotenv mode: search upward, then the environment

def test_find_dotenv_finds_a_dotenv_in_a_parent_of_the_repo_root(search_upward_mode, repo_root, monkeypatch):
    (repo_root.parent / ".env").write_text(f"{VARIABLE}={PARENT_FILE_KEY}\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == PARENT_FILE_KEY


def test_find_dotenv_prefers_the_nearest_dotenv(search_upward_mode, repo_root):
    (repo_root.parent / ".env").write_text(f"{VARIABLE}={PARENT_FILE_KEY}\n")
    (repo_root / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    assert api_keys.load_api_key(VARIABLE) == FILE_KEY


def test_find_dotenv_falls_back_to_the_environment_when_the_file_lacks_the_key(
        search_upward_mode, repo_root, monkeypatch, capsys):
    (repo_root.parent / ".env").write_text("SOME_OTHER_VARIABLE=1\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == ENVIRONMENT_KEY
    output = capsys.readouterr().out
    assert f"has no {VARIABLE}; using the key from the environment" in output


def test_find_dotenv_falls_back_to_the_environment_when_there_is_no_file(search_upward_mode, monkeypatch):
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == ENVIRONMENT_KEY


def test_find_dotenv_errors_when_neither_source_has_the_key(search_upward_mode):
    with pytest.raises(api_keys.MissingApiKeyError) as error:
        api_keys.load_api_key(VARIABLE)
    assert VARIABLE in str(error.value)
    assert api_keys.load_optional_api_key(VARIABLE) is None


# Command-line flags

def _parser_with_key_source_flags() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    api_keys.add_api_key_source_flags(parser)
    return parser


@pytest.mark.parametrize("arguments, expected_mode, expected_subprocess_flags", [
    ([], api_keys.REPO_ROOT_DOTENV_ONLY, []),
    (["--find-dotenv"], api_keys.SEARCH_UPWARD_THEN_ENVIRONMENT, ["--find-dotenv"]),
])
def test_flags_select_the_mode_and_can_be_forwarded_to_a_subprocess(
        repo_root, arguments, expected_mode, expected_subprocess_flags):
    api_keys.apply_api_key_source_flags(_parser_with_key_source_flags().parse_args(arguments))
    assert api_keys._api_key_source_mode == expected_mode
    assert api_keys.api_key_source_flags_for_subprocess() == expected_subprocess_flags


def test_an_unknown_mode_is_refused(repo_root):
    with pytest.raises(ValueError):
        api_keys.set_api_key_source_mode("anywhere")


def test_the_suite_fixture_does_not_search_above_its_temporary_repo_root(tmp_path_factory, monkeypatch):
    """The autouse fixture's repo root is a temporary folder; a .env in any folder above it (a base
    temporary folder under the project or home folder) must never be read."""
    from shared import api_keys as keys_under_suite_fixture
    folder_above_repo_root = keys_under_suite_fixture.REPO_ROOT.parent
    (folder_above_repo_root / ".env").write_text(f"{VARIABLE}={PARENT_FILE_KEY}\n")
    monkeypatch.delenv(VARIABLE, raising=False)
    try:
        assert keys_under_suite_fixture.load_optional_api_key(VARIABLE) is None
    finally:
        (folder_above_repo_root / ".env").unlink()
