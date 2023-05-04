from imagined_handwriting.metrics.text import TextEvaluator, levenshtein_distance


def test_text_errors_updates_with_edit_distance():
    """Verifies a TextEvaluator objects updates with edit distance metrics."""
    sut = TextEvaluator()
    sut.update("hello", "jello")
    assert sut.n_char == 5
    assert sut.n_char_error == 1
    assert sut.n_word == 1
    assert sut.n_word_error == 1


def test_text_errors_update_handles_list_input():
    """Verifies a TextEvaluator object handles list input correctly."""
    sut = TextEvaluator()
    sut.update(["one", "two"], ["oneb", "two"])
    assert sut.n_char == 6
    assert sut.n_char_error == 1
    assert sut.n_word == 2
    assert sut.n_word_error == 1


def test_text_errors_handles_multiple_updates():
    """Verifies that updating multiple times is additive."""
    sut = TextEvaluator()
    sut.update("hello", "jello")
    sut.update("world", "wor")
    assert sut.n_char == 10
    assert sut.n_char_error == 3
    assert sut.n_word == 2
    assert sut.n_word_error == 2


def test_cer_is_computed_correctly():
    """Verifies the CER is computed correctly."""
    sut = TextEvaluator()
    sut.update("hello", "jello")
    assert sut.cer() == 1 / 5


def test_text_errors_are_summed_correctly():
    """Verifies two instances are summed correctly."""
    te1 = TextEvaluator()
    te1.update("hello", "jello")
    te2 = TextEvaluator()
    te2.update("world", "wor")
    te3 = te1 + te2
    assert te3.n_char == 10
    assert te3.n_char_error == 3
    assert te3.n_word == 2
    assert te3.n_word_error == 2


def test_wer_is_computed_with_split_token():
    """Verifies the WER is computed correctly if a character other
    than > is used for the split token."""
    sut = TextEvaluator(split_token="+")
    sent1 = "hello+world"
    sent2 = "hello+wrd"
    sut.update(sent1, sent2)
    assert sut.wer() == 0.5


def test_levenshtein_distance_is_zero_on_equal_words():
    """Verifies the distance is 0 when the words are the same."""
    w1 = "testing"
    w2 = "testing"
    assert levenshtein_distance(w1, w2) == 0


def test_levenshtein_distance_is_zero_on_equal_sentences():
    """Verifies the distance is 0 when the words are the same."""
    s1 = "testing this function".split(" ")
    s2 = "testing this function".split(" ")

    assert levenshtein_distance(s1, s2) == 0


def test_levenshtein_distance_is_one_when_off_by_one():
    """Verifies the distance is correct when words are off by 1."""
    w1 = "cat"
    w2 = "bat"

    assert levenshtein_distance(w1, w2) == 1


def test_levenshtein_distance_is_symmetric():
    """Verifies the distance is symmetric in its inputs."""
    w1 = "hello"
    w2 = "jello"

    assert levenshtein_distance(w1, w2) == levenshtein_distance(w2, w1)


def test_levenshtein_distance_on_empty_string():
    """Verifies the distance is correct when one string is empty"""
    w1 = ""
    w2 = "oops"
    assert levenshtein_distance(w1, w2) == 4
