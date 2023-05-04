from typing import List, Sequence, Union

import numpy as np


class TextEvaluator:
    """Evaluates"""

    def __init__(self, split_token=">"):
        """Initializes the evaluator.

        Args:
            split_token: The token to split the strings on.  Default
            is ">" which represents white space in the imagined handwriting
            dataset.

        """
        self.n_char = 0
        self.n_char_error = 0
        self.n_word = 0
        self.n_word_error = 0
        self.split_token = split_token

    def __add__(self, other):
        if not isinstance(other, TextEvaluator):
            raise ValueError("Can only add TextErrors to TextErrors")
        new = TextEvaluator()
        new.n_char = self.n_char + other.n_char
        new.n_char_error = self.n_char_error + other.n_char_error
        new.n_word = self.n_word + other.n_word
        new.n_word_error = self.n_word_error + other.n_word_error
        return new

    def reset(self):
        """Resets the counters to 0."""
        self.n_char = 0
        self.n_char_error = 0
        self.n_word = 0
        self.n_word_error = 0

    def update(self, y_true: Union[str, List[str]], y_pred: Union[str, List[str]]):
        """Updates the counters with errors and counts for the given input."""
        if isinstance(y_true, str) or isinstance(y_pred, str):
            if not isinstance(y_true, str):
                raise ValueError("y_pred is a string but y_true is not")
            if not isinstance(y_pred, str):
                raise ValueError("y_true is a string but y_pred is not")
            y_true = [y_true]
            y_pred = [y_pred]
        for yt, yp in zip(y_true, y_pred):
            self._update_from_sentence(yt, yp)

    def _update_from_sentence(self, y_true: str, y_pred: str):
        self.n_char += len(y_true)
        self.n_char_error += levenshtein_distance(y_true, y_pred)
        self.n_word += len(y_true.split(self.split_token))
        self.n_word_error += levenshtein_distance(
            y_true.split(self.split_token), y_pred.split(self.split_token)
        )

    def cer(self) -> float:
        """Computes the current character error rate."""
        if self.n_char == 0:
            raise ValueError("No examples seen, did you mean to call update first?")
        return self.n_char_error / self.n_char

    def wer(self) -> float:
        """Computes the current word error rate."""
        if self.n_word == 0:
            raise ValueError("No examples seen, did you mean to call update first?")
        return self.n_word_error / self.n_word


def cer(
    y_true: Union[str, List[str]], y_pred: Union[str, List[str]], split_token=">"
) -> float:
    """Computes the character error rate.

    Args:
        y_true: Either a string or a list of strings
        y_pred: Either a string or a list of strings
        split_token: The token to split the strings on.  Default is ">" which represents
            white space in the imagined handwriting dataset.

    Returns:
        The character error rate

    """
    evaluator = TextEvaluator(split_token=split_token)
    evaluator.update(y_true, y_pred)
    return evaluator.cer()


def wer(
    y_true: Union[str, List[str]], y_pred: Union[str, List[str]], split_token=">"
) -> float:
    """Computes the word error rate.

    Args:
        y_true: Either a string or a list of strings
        y_pred: Either a string or a list of strings
        split_token: The token to split the strings on.  Default is ">" which represents
            white space in the imagined handwriting dataset.

    Returns:
        The word error rate

    """
    evaluator = TextEvaluator(split_token=split_token)
    evaluator.update(y_true, y_pred)
    return evaluator.wer()


def levenshtein_distance(token1: Sequence, token2: Sequence) -> int:
    """Computes the levenshtein distance between words or sentences.

    Args:
        token1: Either a string word or a list of words
            (i.e. a sentence split on whitespace)
        token2: Either a string word or a list of words
            (i.e. a sentences split on whitespace)

    Returns:
        The levenshtein distance between the tokens

    Implemenatation adapted from:
    https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/ # noqa

    """
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                distances[t1][t2] = min(a, b, c) + 1

    return distances[len(token1)][len(token2)]
