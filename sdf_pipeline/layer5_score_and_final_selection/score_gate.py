"""Layer 5 step: the score gate (no model call).

Keeps documents with alignment AND realism >= ``sdf.min_score_threshold``; spec_conformance is recorded and reported
but does not gate (drift diagnostics belong to humans while the dimension is young). A score record that fails to
parse is 5/5/5, which falls below the default threshold of 7.
"""

from sdf_pipeline.shared_step_machinery.update_steps import Accepted, CorpusStep, Removed


class ScoreGateStep(CorpusStep):
    """Keep documents with alignment AND realism >= the threshold. The documents are score records
    (``doc_id``, ``scores``); spec_conformance is advisory and does not gate."""

    step_name = "score_gate"

    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, score_records: list) -> dict:
        outcomes = {}
        for record in score_records:
            scores = record["scores"]
            if scores["alignment"] >= self.threshold and scores["realism"] >= self.threshold:
                outcomes[record["doc_id"]] = Accepted(None)
            else:
                outcomes[record["doc_id"]] = Removed(
                    "below_score_threshold",
                    f"alignment {scores['alignment']}, realism {scores['realism']}, threshold {self.threshold}")
        return outcomes
