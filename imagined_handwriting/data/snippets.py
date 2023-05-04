from collections import defaultdict
from copy import deepcopy
from typing import List, Optional

import numpy as np
from numpy import ndarray

from imagined_handwriting.data.handwriting import HandwritingSessionData


class Snippets:
    """A container for holding snippets of single letter neural activity."""

    def __init__(
        self,
        snippets: List[ndarray],
        character_index: dict,
        sentence_index: Optional[List[int]],
    ):
        """Initialize a Snippets object.

        Args:
            snippets: A list of snippets.
            character_index: A dictionary mapping characters to the indices of
                the snippets that contain that character.
            sentence_index: A list of indices such that snippets[i] is a snippet
                for the sentence in which the character appears.

        """
        self.snippets = snippets
        self.character_index = character_index
        self.sentence_index = sentence_index

    def copy(self):
        index = None if self.sentence_index is None else deepcopy(self.sentence_index)
        return Snippets(
            snippets=deepcopy(self.snippets),
            character_index=deepcopy(self.character_index),
            sentence_index=index,
        )

    def __getitem__(self, i):
        return self.snippets[i]

    def get_snippets(self, char: str) -> List[ndarray]:
        """Returns the snippets for the the given character.

        Args:
            char: The character to get snippets for.

        Returns:
            A list of snippets for the given character.

        """
        return [self.snippets[i] for i in self.character_index[char]]

    def get_indices(self, char: str) -> ndarray:
        """Returns the indices of the snippets that contain the given character.

        Args:
            char: The character to get indices for.

        Returns:
            An array of indices such that snippets.snippets[i] is a snippet
            for the request character for each i in the array of indices.

        """
        return self.character_index[char]


def extract_snippets(session: HandwritingSessionData) -> Snippets:
    """Extracts snippets from the sentence trials of the given session.

    Args:
        Session: The session object to extract sentence snippets from.

    Returns:
        A Snippets object containing the snippets extracted from the session.
    """
    snippets = []
    snippet_characters = []  # type: List[str]
    sentence_index = []
    bad_trials = []

    for trial in range(session.x.shape[0]):
        extracted = extract_single_trial_snippet(trial, session)

        if extracted is not None:
            snips = extracted[0]
            chars = extracted[1]
            snippets.extend(snips)
            snippet_characters.extend(chars)
            sentence_index.extend([trial] * len(snips))
        else:
            bad_trials.append(trial)

    character_index = build_character_index(snippet_characters)
    n_indices = sum(len(v) for v in character_index.values())
    if len(snippets) != n_indices:
        raise ValueError(
            "Number of snippets and number of characters do not match. "
            f"{len(snippets)} != {n_indices}"
        )

    return Snippets(
        snippets=snippets,
        character_index=character_index,
        sentence_index=sentence_index,
    )


def extract_single_trial_snippet(trial: int, session: HandwritingSessionData):
    """Extracts a snippet from a single trial.

    A trial is invalid when the number of characters (as labeled by the HMM)
    is less than the number of characters in the text of the sentence.  In
    this case we can't figure out which time slices correspond to which
    characters so we return None.

    Args:
        trial: The trial index to extract snippets from.
        session: The processed session object.

    Returns:
        A tuple of (snippets, snippet_characters) or None if the trial is invalid.
    """
    eos = session.end_of_sentence_index[trial]
    neural_data = session.x[trial, :eos].astype(np.float32)
    text = session.text[trial]
    start_index = session.start_index[trial]
    splits = np.split(neural_data, start_index, axis=0)[1:]

    if len(splits) != len(text):
        return None

    return splits, list(text)


def build_character_index(characters: List[str]):
    """Builds a mapping from each character an array of indices.

    The indices are the indices of the character in the list of characters.

    Args:
        characters: A list of characters.

    Returns:
        A dictionary mapping each character to an array of indices.
    """

    char_map = defaultdict(list)
    for i, char in enumerate(characters):
        char_map[char].append(i)

    return {k: np.array(v) for k, v in char_map.items()}
