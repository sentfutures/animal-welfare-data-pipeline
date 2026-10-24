"""shared/api_keys.py: where a key comes from. Temporary folders and fake keys only."""

import os

import pytest

from shared import api_keys

VARIABLE = "FAKE_PROJECT_API_KEY"
FILE_KEY = "fake-key-from-the-dotenv-file-0001"
ENVIRONMENT_KEY = "fake-key-from-the-environment-0002"


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    root = tmp_path / "outer_folder" / "repo_root"
    root.mkdir(parents=True)
    monkeypatch.setattr(api_keys, "REPO_ROOT", root)
    monkeypatch.setattr(api_keys, "_sources_already_announced", set())
    monkeypatch.delenv(VARIABLE, raising=False)
    return root


def test_key_in_file_overrides_environment_and_sets_it(repo_root, monkeypatch, capsys):
    (repo_root / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == FILE_KEY
    assert os.environ[VARIABLE] == FILE_KEY
    assert capsys.readouterr().out.strip() == (
        f"{VARIABLE} from {(repo_root / '.env').resolve()} (value not printed)")


def test_empty_value_in_file_uses_environment_and_says_the_file_has_no_key(repo_root, monkeypatch, capsys):
    (repo_root / ".env").write_text(f"{VARIABLE}=\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == ENVIRONMENT_KEY
    output = capsys.readouterr().out
    assert f"{(repo_root / '.env').resolve()} has no {VARIABLE}; using the key from the environment" in output
    assert f"{VARIABLE} from the environment" in output


def test_file_without_the_variable_uses_environment_and_says_so(repo_root, monkeypatch, capsys):
    (repo_root / ".env").write_text("SOME_OTHER_VARIABLE=1\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == ENVIRONMENT_KEY
    output = capsys.readouterr().out
    assert "has no " + VARIABLE in output and f"{VARIABLE} from the environment" in output


def test_no_file_uses_environment(repo_root, monkeypatch, capsys):
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == ENVIRONMENT_KEY
    assert capsys.readouterr().out.strip() == f"{VARIABLE} from the environment"


def test_neither_source_raises_an_error_naming_the_path(repo_root):
    with pytest.raises(api_keys.MissingApiKeyError) as error:
        api_keys.load_api_key(VARIABLE)
    message = str(error.value)
    assert str((repo_root / ".env").resolve()) in message
    assert f"Fill in {VARIABLE}" in message and ".env.example" in message


def test_optional_key_returns_none_when_absent(repo_root):
    assert api_keys.load_optional_api_key(VARIABLE) is None


def test_whitespace_only_value_counts_as_empty(repo_root, monkeypatch):
    (repo_root / ".env").write_text(f'{VARIABLE}="   "\n')
    with pytest.raises(api_keys.MissingApiKeyError):
        api_keys.load_api_key(VARIABLE)
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    assert api_keys.load_api_key(VARIABLE) == ENVIRONMENT_KEY


def test_empty_value_in_file_does_not_clobber_a_good_environment_value(repo_root, monkeypatch):
    (repo_root / ".env").write_text(f"{VARIABLE}=\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    api_keys.load_api_key(VARIABLE)
    assert os.environ[VARIABLE] == ENVIRONMENT_KEY


def test_dotenv_in_a_parent_of_the_repo_root_is_ignored(repo_root):
    (repo_root.parent / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    with pytest.raises(api_keys.MissingApiKeyError):
        api_keys.load_api_key(VARIABLE)


def test_dotenv_in_the_current_directory_is_ignored(repo_root, tmp_path, monkeypatch):
    working_directory = tmp_path / "somewhere_else"
    working_directory.mkdir()
    (working_directory / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    monkeypatch.chdir(working_directory)
    with pytest.raises(api_keys.MissingApiKeyError):
        api_keys.load_api_key(VARIABLE)


def test_key_value_never_appears_in_output(repo_root, monkeypatch, capsys):
    (repo_root / ".env").write_text(f"{VARIABLE}={FILE_KEY}\n")
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    api_keys.load_api_key(VARIABLE)
    (repo_root / ".env").write_text("")
    monkeypatch.setattr(api_keys, "_sources_already_announced", set())
    api_keys.load_api_key(VARIABLE)
    captured = capsys.readouterr()
    for secret in (FILE_KEY, ENVIRONMENT_KEY):
        assert secret not in captured.out and secret not in captured.err


def test_source_is_printed_once_per_variable(repo_root, monkeypatch, capsys):
    monkeypatch.setenv(VARIABLE, ENVIRONMENT_KEY)
    api_keys.load_api_key(VARIABLE)
    api_keys.load_api_key(VARIABLE)
    assert capsys.readouterr().out.count(f"{VARIABLE} from the environment") == 1
