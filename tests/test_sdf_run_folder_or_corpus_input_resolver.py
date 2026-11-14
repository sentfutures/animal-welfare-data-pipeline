"""``resolve_sdf_input``: how an eval's ``--input`` (a run folder or a corpus file) maps to the corpus
file, the run folder and the folder an audit's ``audit/`` goes in, and the two exit messages."""

import pytest

from sdf_pipeline.document_state_final_corpus import resolve_sdf_input, run_directory_of_corpus_file


def test_a_run_folder_names_its_final_corpus_itself_and_its_audit_folder(tmp_path):
    resolved = resolve_sdf_input(tmp_path)
    assert resolved.is_run_directory and resolved.run_directory == tmp_path
    assert resolved.corpus_path == tmp_path / "final" / "sdf_corpus.jsonl"
    assert resolved.audit_parent_directory == tmp_path / "audit"


def test_a_corpus_file_in_a_final_folder_belongs_to_the_run_folder_above_it(tmp_path):
    corpus = tmp_path / "final" / "sdf_corpus.jsonl"
    resolved = resolve_sdf_input(str(corpus))
    assert not resolved.is_run_directory and resolved.corpus_path == corpus
    assert resolved.run_directory == tmp_path and run_directory_of_corpus_file(corpus) == tmp_path
    assert resolved.audit_parent_directory == corpus.parent / "audit"


def test_a_corpus_file_elsewhere_has_no_run_folder(tmp_path):
    resolved = resolve_sdf_input(tmp_path / "sample.jsonl")
    assert resolved.run_directory is None and run_directory_of_corpus_file(tmp_path / "sample.jsonl") is None


def test_the_exit_messages_for_a_run_folder_without_a_corpus_and_for_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="No final/sdf_corpus.jsonl under"):
        resolve_sdf_input(tmp_path).corpus_path_or_exit_with_input_not_found_message()
    with pytest.raises(SystemExit, match="Input not found"):
        resolve_sdf_input(tmp_path / "missing.jsonl").corpus_path_or_exit_with_input_not_found_message()
    corpus = tmp_path / "sample.jsonl"
    corpus.write_text("", encoding="utf-8")
    assert resolve_sdf_input(corpus).corpus_path_or_exit_with_input_not_found_message() == corpus
