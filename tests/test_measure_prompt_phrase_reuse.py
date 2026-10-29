from evals import measure_prompt_phrase_reuse as measure


def test_phrase_shares_are_case_and_whitespace_insensitive_and_split_by_language():
    documents = [
        {"content": "A Realistic\n possibility exists.", "language": "English"},
        {"content": "Nothing here.", "language": "English"},
        {"content": "realistic possibility", "language": "Spanish"},
        {"content": "Otra cosa.", "language": "Spanish"},
    ]
    (row,) = measure.measure(documents, [("prompt_phrase", "realistic possibility")])
    assert row["documents_containing"] == 2 and row["share_of_all_documents"] == 0.5
    assert row["english_documents_containing"] == 1 and row["share_of_english_documents"] == 0.5


def test_phrase_file_skips_comments_and_the_real_file_loads():
    phrases = measure.load_phrase_file(measure.DEFAULT_PHRASE_FILE)
    assert ("prompt_phrase", "realistic possibility") in phrases
    assert all(not phrase.startswith("#") for _, phrase in phrases)


def test_a_phrase_inside_a_longer_word_is_not_counted():
    documents = [
        {"content": "It was illegible.", "language": "English"},
        {"content": "The text is legible, if small.", "language": "English"},
    ]
    (row,) = measure.measure(documents, [("prompt_phrase", "legible")])
    assert row["documents_containing"] == 1 and row["english_documents_containing"] == 1


def test_curly_and_straight_apostrophes_match_each_other():
    documents = [
        {"content": "She didn’t lecture anyone.", "language": "English"},
        {"content": "He didn't lecture either.", "language": "English"},
    ]
    rows = measure.measure(documents, [
        ("prompt_phrase", "didn't lecture"), ("prompt_phrase", "didn’t lecture")])
    assert [row["documents_containing"] for row in rows] == [2, 2]
