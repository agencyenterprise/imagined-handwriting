import numpy as np
import pytest

from imagined_handwriting.datasets.utils import (
    HandwritingTextSampler,
    SnippetSampler,
    TextSampler,
    parse_session_id,
)
from imagined_handwriting.settings import SESSIONS


@pytest.fixture(autouse=True)
def no_pause(monkeypatch, request):
    """Patches the snippet sampler to always sample the first example of the
    character."""
    if "pause" in request.keywords:
        return

    def _pause(*args, **kwargs):
        return None

    monkeypatch.setattr(SnippetSampler, "synthetic_pause", _pause)


@pytest.fixture(autouse=True)
def patch_pause_percent(monkeypatch):
    """Patches the snippet sampler to always sample the first example of the
    character."""

    monkeypatch.setattr(SnippetSampler, "PAUSE_PERCENT", 0.9)


# Snippet Sampler
# ---------------
@pytest.fixture()
def fake_snippets():
    rng = np.random.default_rng(0)
    snippets = [rng.normal(size=(100, 5)) for _ in range(4)]
    character_index = {"a": np.array([0, 1]), "b": np.array([2, 3])}
    sentence_index = np.array([0, 1, 2, 3])

    class FakeSnippets:
        def __init__(self):
            self.snippets = snippets
            self.character_index = character_index
            self.sentence_index = sentence_index

        def __getitem__(self, i):
            return self.snippets[i]

        def get_indices(self, char):
            return self.character_index[char]

    return FakeSnippets()


def test_snippet_sampler_returns_data_with_correct_shape(fake_snippets):
    """Verifies the snippet sampler returns data with the correct shape."""
    sampler = SnippetSampler(fake_snippets)
    sampled, _ = sampler("ab")
    assert sampled[0].shape == (100, 5)
    assert sampled[1].shape == (100, 5)


def test_snippet_sampler_does_not_change_data(fake_snippets):
    """Verifies the snippet sampler does not make changes to the data."""

    def add_noise(data, rng):
        return data + rng.standard_normal(data.shape)

    sampler = SnippetSampler(fake_snippets, transform=add_noise, seed=0)
    data = fake_snippets.snippets.copy()
    _ = sampler("ab")

    # verify the data is unchanged
    np.testing.assert_array_equal(data, fake_snippets.snippets)


def test_snippet_sampler_applies_transform(fake_snippets, monkeypatch):
    """Verifies the snippet sampler applies the transformation to the snippets."""

    def transform(data, rng):
        return data + 1

    # always sample the first example of the character.
    data = fake_snippets.snippets.copy()
    sampler = SnippetSampler(fake_snippets, transform=transform)
    monkeypatch.setattr(sampler, "_sample_index", lambda x: x[0])

    sampled, _ = sampler("ab")

    # verify the data is transformed
    np.testing.assert_array_equal(sampled[0], data[0] + 1)
    np.testing.assert_array_equal(sampled[1], data[2] + 1)


def test_snippet_sampler_is_deterministic_with_reset(fake_snippets):
    """Verifies the snippet sampler is deterministic with reset."""

    sampler = SnippetSampler(fake_snippets, transform=None, seed=0)
    sampled, _ = sampler("ab")
    sampler.reset()
    sampled2, _ = sampler("ab")
    np.testing.assert_array_equal(sampled, sampled2)


@pytest.mark.pause
def test_snippet_sampler_returns_pauses_with_correct_shape(fake_snippets):
    """Verifies the snippet sampler returns pauses with the correct shape."""
    sampler = SnippetSampler(fake_snippets)
    pause = sampler.make_pause()
    assert pause.shape[1] == 5


@pytest.mark.pause
def test_snippet_sampler_creates_pauses_roughly_correct_percent_of_time(fake_snippets):
    """Verifies the snippet sampler creates pauses roughly 3% of the time."""
    sampler = SnippetSampler(fake_snippets, transform=None, seed=0)
    num_pauses = 0
    for _ in range(1000):
        _, pause = sampler("a")
        if pause > 0:
            num_pauses += 1
    assert 0.85 <= num_pauses / 1000 <= 0.95


# Text Sampler
# ------------


def test_random_words_returns_a_string():
    """Verifies that the random_word method returns a string."""
    corpus = ["test", "words"]
    sampler = TextSampler(corpus)
    sentence = sampler.random_word()
    assert isinstance(sentence, str)


def test_sampler_repeats_sentence_after_reset():
    """Verifies that the sampler repeats words after reset."""
    corpus = ["hello", "world"]
    sampler = TextSampler(corpus, seed=0)
    sentence = sampler()
    sampler.reset()
    sentence2 = sampler()
    assert sentence == sentence2


@pytest.mark.parametrize("num_words", [2, 10])
def test_sampler_samples_correct_number_of_words(num_words):
    """Verifies that the sampler samples the correct number of words."""
    corpus = ["hello", "world"]
    sampler = TextSampler(corpus, num_words=num_words)
    sentence = sampler()
    words = sentence.split(" ")
    assert len(words) == num_words


def test_sample_sentence_samples_from_corpus():
    """Verifies that the sample_sentence method samples from the corpus."""
    corpus = ["hello", "world"]
    sampler = TextSampler(corpus, num_words=20)
    sentence = sampler.sample_sentence()
    for word in sentence.split(" "):
        assert word in corpus


def test_handwriting_text_sampler_adds_space_char():
    """Verifies that the handwriting text sampler adds a space character."""
    valid_chars = set(list("hello") + list("world") + [">"])
    corpus = ["hello", "world"]
    sampler = HandwritingTextSampler(valid_chars, corpus, num_words=4)
    sentence = sampler()
    assert ">" in sentence


@pytest.mark.parametrize("special_char", ["~", "?", ",", "'"])
def test_handwriting_text_sampler_adds_punctuation(special_char):
    """Verifies that the handwriting text sampler adds punctuation."""
    valid_chars = set(list("hello") + list("world") + [special_char])
    corpus = ["hello", "world"]
    sampler = HandwritingTextSampler(valid_chars, corpus, num_words=1000, seed=0)
    sentence = sampler()
    assert special_char in sentence


def test_handwriting_sampler_does_not_fail_on_missing_punctuation():
    """Verifies the handwriting sampler does not fail on missing punctuation."""
    valid_chars = set(list("hello") + list("world"))
    corpus = ["hello", "world"]
    sampler = HandwritingTextSampler(valid_chars, corpus, num_words=1000, seed=0)
    _ = sampler()


@pytest.mark.parametrize("session", ["pretrain", "copy_typing", "free_typing"])
def test_parse_session_id_for_name_session(session):
    """Tests that parse_session_id returns the correct session for a name session"""
    actual = parse_session_id(session)
    expected = getattr(SESSIONS, session)
    assert actual == expected


@pytest.mark.parametrize("session", SESSIONS.all)
@pytest.mark.parametrize("comparison", ["<", "<=", ">", ">="])
def test_parse_session_id_comparison(session, comparison):
    """Tests that parse_session_id returns the correct session with comparison
    operator"""
    actual = parse_session_id(f"{comparison}{session}")
    session_index = SESSIONS.all.index(session)
    if comparison == "<":
        expected = SESSIONS.all[:session_index]
    elif comparison == "<=":
        expected = SESSIONS.all[: session_index + 1]
    elif comparison == ">":
        expected = SESSIONS.all[session_index + 1 :]
    elif comparison == ">=":
        expected = SESSIONS.all[session_index:]
    assert actual == expected


def test_parse_session_id_for_comma_separated_sessions():
    """Tests that parse_session_id returns the correct session for a
    comma-separated session"""
    actual = parse_session_id(",".join(SESSIONS.all[:5]))
    expected = SESSIONS.all[:5]
    assert actual == expected
