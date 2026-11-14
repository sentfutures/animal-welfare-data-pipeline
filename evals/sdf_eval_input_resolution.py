"""How an SDF eval's ``--input`` (a run folder or a corpus file) maps to the corpus file, the run folder and the
folder an audit's ``audit/`` folder goes in."""

from dataclasses import dataclass
from pathlib import Path

FINAL_CORPUS_RELATIVE_PATH = Path("final") / "sdf_corpus.jsonl"


@dataclass(frozen=True)
class ResolvedSdfInput:
    """What an ``--input`` of an eval names: an SDF run folder or a corpus file.

    ``corpus_path`` is the corpus file (for a run folder, its ``final/sdf_corpus.jsonl``, which may not
    exist). ``run_directory`` is the run folder (None for a corpus file that is not in a run's
    ``final/`` folder). ``audit_parent_directory`` is the folder an audit's ``audit/`` folder goes in."""
    input_path: Path
    corpus_path: Path
    run_directory: Path | None
    audit_parent_directory: Path

    @property
    def is_run_directory(self) -> bool:
        return self.input_path.is_dir()

    def corpus_path_or_exit_with_input_not_found_message(self) -> Path:
        """``corpus_path``, or an exit with the message the audits give for a run folder without a
        final corpus ("No final/sdf_corpus.jsonl under ...") or for a path that does not exist."""
        if self.is_run_directory and not self.corpus_path.exists():
            raise SystemExit(f"No final/sdf_corpus.jsonl under {self.input_path}")
        if not self.is_run_directory and not self.input_path.exists():
            raise SystemExit(f"Input not found: {self.input_path}")
        return self.corpus_path


def run_directory_of_corpus_file(corpus_path) -> Path | None:
    """The run folder a corpus file sits in (``<run>/final/sdf_corpus.jsonl``), else None."""
    corpus_path = Path(corpus_path)
    return corpus_path.parent.parent if corpus_path.parent.name == "final" else None


def resolve_sdf_input(input_argument) -> ResolvedSdfInput:
    """Resolve an eval's ``--input``. Checks nothing exists."""
    path = Path(input_argument)
    if path.is_dir():
        return ResolvedSdfInput(path, path / FINAL_CORPUS_RELATIVE_PATH, path, path / "audit")
    return ResolvedSdfInput(path, path, run_directory_of_corpus_file(path), path.parent / "audit")
