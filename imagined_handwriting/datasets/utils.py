from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
from numpy import ndarray

from imagined_handwriting.data import Snippets
from imagined_handwriting.settings import SESSIONS


class NumpyRNGMixin:
    def set_rng(self, seed: Optional[int]) -> None:
        """Sets the numpy generator.

        Args:
            seed: The seed for the numpy generator.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Resets the numpy generator based on the current seed.

        Raise:
            RuntimeError: If the seed is not set.
            ValueError: If the seed is None.
        """
        if not hasattr(self, "seed"):
            raise RuntimeError("Can not call reset without a seed. First call set_rng")
        if self.seed is None:
            raise ValueError(
                "Can not reset the rng with a seed of None. "
                "First call set_rng with an integer argument."
            )
        self.rng = np.random.default_rng(self.seed)


class TextSampler(NumpyRNGMixin):
    """A pytorch multiprocessing memory safe sampler for text data."""

    def __init__(
        self,
        corpus: List[str],
        num_words: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Initializes a TextSampler.

        Args:
            corpus: A list of strings to sample from.
            seed: A seed for the random number generator.
        """

        self.corpus = corpus
        self.corpus_bytes = text_to_bytes_array(corpus)
        self.num_words = num_words
        self.set_rng(seed)

    def __call__(self) -> str:
        """Returns a randomly sampled sentence."""
        return self.sample_sentence()

    def random_word(self) -> str:
        """Samples a word from the corpus."""
        rand_idx = self.rng.integers(0, len(self.corpus))
        return bytes_to_string(self.corpus_bytes[rand_idx])

    def sample_words(self) -> List[str]:
        """Samples a list of words from the corpus.

        Args:
            num_words: The number of words to sample. If None, samples a random
                integer between 2 and 50.

        Returns:
            A list of words.
        """
        if self.num_words is None:
            num_words = self.rng.integers(4, 30)
        else:
            num_words = self.num_words
        return [self.random_word() for _ in range(num_words)]

    def sample_sentence(self):
        """Samples a sentence by joining randomly sampled words."""
        return " ".join(self.sample_words())


class HandwritingTextSampler(TextSampler):
    """A pytorch multiprocessing memory safe sampler for handwriting text data.

    Uses specifics about the handwriting dataset (i.e. encoding spaces, and punctuation)
    when sampling a sentence.
    """

    def __init__(
        self,
        valid_chars: Set[str],
        corpus: List[str],
        num_words: int = 8,
        seed: int = None,
    ):
        corpus = [x for x in corpus if all(c in valid_chars for c in x)]
        super().__init__(corpus, num_words, seed)
        self.valid_chars = valid_chars

    def sample_sentence(self) -> str:
        """Samples a sentence by joining random words and punctuation.

        Note, we use the encoded symbols that are used in the handwriting dataset.
        Namely,
            `>` is used as space
            `~` is used as a period

        Args:
            num_words: The number of words to sample. If None, samples a random
                number of words between 2 and 50.

        Returns:
            A sentence made of randomly sampled words with random punctuation.
        """
        words = self.sample_words()
        characters = []
        for word in words:
            word_chars = list(word)
            u = self.rng.uniform()
            if u < 0.03 and self.is_valid("'"):
                word_chars.insert(len(word_chars) - 1, "'")
            elif u < 0.1 and self.is_valid(","):
                word_chars.append(",")
            elif u < 0.15 and self.is_valid("~"):
                word_chars.append("~")
            elif u < 0.2 and self.is_valid("?"):
                word_chars.append("?")

            if self.is_valid(">"):
                word_chars.append(">")
            characters.extend(word_chars)

        # 50% of the time remove space at end of sentence.
        new_u = self.rng.uniform()
        if new_u < 0.5 and self.is_valid(">"):
            characters = characters[:-1]

        return "".join(characters)

    def is_valid(self, char: str):
        return char in self.valid_chars


class SnippetSampler(NumpyRNGMixin):
    """A pytorch multiprocessing memory safe sampler for snippets."""

    PAUSE_PERCENT = 0.03

    def __init__(
        self,
        snippets: Snippets,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        sample_rate: float = 100,
    ):
        """Initializes a SnippetSampler.

        Args:
            snippets: A Snippets object which contains the underlying data.
            transform: An optional function that will be applied independently
                to each snippet. It must accept a numpy array and an rng.
            seed: An optional integer to seed the random number generator.

        """
        self.snippets = snippets
        self.transform = transform
        self.sample_rate = sample_rate
        self.set_rng(seed)

    def __call__(self, sentence: str) -> Tuple[List[ndarray], int]:
        """Samples a snippet for a sentence.

        Args:
            sentence: The sentence to sample a snippet for.

        Returns:
            A list numpy arrays, one for each character in the sentence.
        """
        return self.sample_activity_for_sentence(sentence)

    def sample_activity_for_sentence(self, sentence: str) -> Tuple[List[ndarray], int]:
        """Samples a snippet for a sentence.

        Note that the snippets are sampled and transformed independently.
        Also, following the original paper we add a synthetic pause
        between snippets with probability 3%.  See the supplemental
        information pg. 23.

        Reference:
            https://doi.org/10.1038/s41586-021-03506-2

        Args:
            sentence: The sentence to sample a snippet for.

        Returns:
            A list numpy arrays, one for each character in the sentence.
        """
        data = []
        n_pause = 0
        for char in sentence:
            pause = self.synthetic_pause()
            if pause is not None:
                data.append(pause)
                n_pause += 1
            indices = self.snippets.get_indices(char)
            snippet = self.snippets[self._sample_index(indices)]
            if self.transform is not None:
                snippet = self.transform(snippet, rng=self.rng)
            data.append(snippet)
        return data, n_pause

    def synthetic_pause(self) -> Optional[ndarray]:
        """Samples a synthetic pause.

        The synthetic pause is a snippet of white noise whose
        length is drawn from an exponential distribution with
        mean 1 second.  See the supplemental information pg. 23.

        Returns:
            A numpy array representing a synthetic pause.
        """
        u = self.rng.uniform()
        if u < self.PAUSE_PERCENT:
            return self.make_pause()
        return None

    def make_pause(self) -> ndarray:
        """Makes a random pause of white noise."""
        pause_length = self.rng.exponential(scale=1.0)
        pause_bins = int(np.round((pause_length * self.sample_rate)))
        channels = self.snippets[0].shape[-1]
        return self.rng.normal(size=(pause_bins, channels)).astype(np.float32)

    def _sample_index(self, index: Union[List[int], ndarray]) -> int:
        return self.rng.choice(index)


def parse_session_id(session_id: Union[str, List[str]], sessions=SESSIONS) -> List[str]:
    """Parse the session id into a list of sessions

    Args:
        session_id: The session id to parse. This can be a
            single session, a comma separate list of sessions,
            a range of sessions starting with the comparison
            operators "<", ">", "<=", ">=" or a named session
            type: one of ['pretrain', 'copy_typing', 'free_typing'].

    Returns:
        A list of session ids.

    Examples:
    >>> parse_session_id("<t5.2019.12.11")
    ['t5.2019.05.08, 't5.2019.11.25', 't5.2019.12.09']
    """
    if isinstance(session_id, list):
        return session_id
    if session_id is None:
        raise ValueError("session_id cannot be None when parsing session id.")
    if session_id in ["pretrain", "copy_typing", "free_typing"]:
        return getattr(sessions, session_id)

    sessions = sessions.all

    if session_id.startswith("<"):
        if session_id.startswith("<="):
            session_index = sessions.index(session_id[2:]) + 1
        else:
            session_index = sessions.index(session_id[1:])
        return sessions[:session_index]

    if session_id.startswith(">"):
        if session_id.startswith(">="):
            session_index = sessions.index(session_id[2:])
        else:
            session_index = sessions.index(session_id[1:]) + 1
        return sessions[session_index:]

    if "," in session_id:
        return sorted(session_id.split(","))
    return [session_id]


"""
Utilities to avoid passing around list of strings or numpy arrays of
type object, which can lead to exploding memory with pytorch
multi-processing.

See:
* https://pytorch.org/docs/stable/data.html#multi-process-data-loading
* https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
"""


def text_to_bytes_array(text: List[str]) -> ndarray:
    return np.array(text).astype(np.string_)


def bytes_to_string(b: bytes) -> str:
    return str(b, encoding="utf-8")
