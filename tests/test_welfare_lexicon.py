"""Tests for evals/welfare_lexicon.py (fully offline; no dataset streaming).

The derivation is driven with hand-built background counters rather than a real
stream, so the two filters that matter — distinctiveness against the background
and the animal-lift that drops AI vocabulary — are tested on geometry with a
known answer.
"""

import collections

import pytest
import yaml

from evals import welfare_lexicon as wl


def _background(animal_docs: dict[str, int], other_docs: dict[str, int],
                n_animal: int, n_other: int):
    """(df, df_animal, n, n_animal) from per-term document counts."""
    df = collections.Counter()
    df_animal = collections.Counter()
    for term, c in animal_docs.items():
        df[term] += c
        df_animal[term] += c
    for term, c in other_docs.items():
        df[term] += c
    return df, df_animal, n_animal + n_other, n_animal


class TestGrams:
    def test_yields_unigrams_and_bigrams_without_stopword_edges(self):
        g = wl.grams("The humane slaughter of farmed animals")
        assert "humane slaughter" in g
        assert "farmed animals" in g
        assert "slaughter farmed" not in g  # "of" breaks the bigram
        assert "the" not in g

    def test_counts_a_document_once_however_often_a_term_repeats(self):
        assert wl.grams("shrimp shrimp shrimp") == {"shrimp", "shrimp shrimp"}

    def test_is_case_insensitive(self):
        assert wl.grams("Aquaculture") == wl.grams("aquaculture")


class TestAnimalDoc:
    def test_three_seed_hits_makes_a_document_animal_ish(self):
        assert wl.is_animal_doc("the pigs and the chickens and the cows")

    def test_two_hits_does_not(self):
        assert not wl.is_animal_doc("the pigs and the chickens")

    def test_repeats_of_one_word_still_count(self):
        # deliberate: three mentions of animals is a document about animals
        assert wl.is_animal_doc("animals animals animals")


class TestDerive:
    def test_keeps_a_distinctive_animal_term(self):
        df, dfa, n, na = _background({"aquaculture": 40}, {"aquaculture": 5}, 2000, 18000)
        terms = wl.derive(["aquaculture ponds"] * 20, df, dfa, n, na)
        assert [t["term"] for t in terms if t["term"] == "aquaculture"]

    def test_rejects_a_distinctive_term_that_is_not_about_animals(self):
        # "claude" is as distinctive as any welfare term, but in the background
        # it sits with Monet and Shannon, not with animals
        df, dfa, n, na = _background({"claude": 5}, {"claude": 95}, 2000, 18000)
        terms = wl.derive(["claude said"] * 20, df, dfa, n, na)
        assert "claude" not in [t["term"] for t in terms]

    def test_rejects_a_term_too_rare_in_the_background_to_judge(self):
        df, dfa, n, na = _background({"decapod": 3}, {}, 2000, 18000)
        terms = wl.derive(["decapod crustaceans"] * 20, df, dfa, n, na)
        assert "decapod" not in [t["term"] for t in terms]

    def test_rejects_a_term_that_is_common_in_the_background(self):
        # appears everywhere, so it is not distinctive of our corpus
        df, dfa, n, na = _background({"animals": 2000}, {"animals": 14000}, 2000, 18000)
        terms = wl.derive(["animals here"] * 20, df, dfa, n, na)
        assert "animals" not in [t["term"] for t in terms]

    def test_ranks_by_distinctiveness_and_honours_top_k(self):
        df, dfa, n, na = _background(
            {"aquaculture": 30, "husbandry": 40, "livestock": 60}, {}, 2000, 18000)
        texts = ["aquaculture husbandry livestock"] * 20
        terms = wl.derive(texts, df, dfa, n, na, top_k=2)
        assert len(terms) == 2
        d = [t["distinctiveness"] for t in terms]
        assert d == sorted(d, reverse=True)

    def test_an_all_background_with_no_animal_documents_raises(self):
        df, dfa, n, na = _background({}, {"x": 10}, 0, 20000)
        with pytest.raises(ValueError, match="no animal-ish documents"):
            wl.derive(["x y"], df, dfa, n, na)

    def test_every_kept_term_carries_the_statistics_behind_it(self):
        df, dfa, n, na = _background({"aquaculture": 40}, {}, 2000, 18000)
        terms = wl.derive(["aquaculture ponds"] * 20, df, dfa, n, na)
        assert set(terms[0]) == {"term", "distinctiveness", "animal_lift",
                                 "corpus_df", "background_df"}


class TestMatching:
    def test_counts_distinct_terms_not_total_hits(self):
        pats = wl.compile_matcher(["shrimp", "welfare"])
        assert wl.count_phrases("shrimp shrimp shrimp", pats) == 1
        assert wl.count_phrases("shrimp welfare", pats) == 2

    def test_respects_word_boundaries(self):
        pats = wl.compile_matcher(["cat"])
        assert wl.count_phrases("the cat sat", pats) == 1
        assert wl.count_phrases("concatenate", pats) == 0

    def test_is_case_insensitive(self):
        assert wl.count_phrases("Animal Welfare", wl.compile_matcher(["animal welfare"])) == 1


class TestLoadLexicon:
    def test_round_trips_a_written_file(self, tmp_path):
        p = tmp_path / "lex.yaml"
        p.write_text(yaml.safe_dump({"terms": [{"term": "shrimp"}, {"term": "husbandry"}]}),
                     encoding="utf-8")
        assert wl.load_lexicon(p) == ["shrimp", "husbandry"]

    def test_a_missing_file_says_how_to_make_one(self, tmp_path):
        with pytest.raises(SystemExit, match="welfare_lexicon.py"):
            wl.load_lexicon(tmp_path / "absent.yaml")

    def test_an_empty_lexicon_is_an_error_not_a_silent_no_op(self, tmp_path):
        p = tmp_path / "lex.yaml"
        p.write_text(yaml.safe_dump({"terms": []}), encoding="utf-8")
        with pytest.raises(SystemExit, match="no terms"):
            wl.load_lexicon(p)
