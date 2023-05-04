import numpy as np
from numpy import ndarray


def remove_pause_from_end_of_sentence(
    end_of_sentence: ndarray, y_start: ndarray, max_pause: int
):
    """Adjusts the end of sentence signal to account for max pause.

    In some trials the subject rested after the last character
    before turning his head to indicate a new trial.  To avoid
    including trailing rest periods at the end of the sentence we
    trim the sentence by waiting a maximum number of time steps
    after the last character starts.

    For more experimental details see the original paper:
        "High-performance brain-to-text communication via handwriting"
        https://doi.org/10.1038/s41586-021-03506-2

    Args:
        end_of_sentence: An array of shape (trials,) giving the
            index of the end of sentence indicated by the subject
            turning his head.
        y_start: An array of shape (trials, time_steps) where a
            1 indicates the start of a character.  Due to the way
            the start signal is created, 1s will be extended for
            some number of time steps from the start of the character
        max_pause: The maximum number of time steps that can occur
            after the last character before we declare the end of
            the sentence.

    Returns:
        The adjusted end of sentence signal of shape (trials, time_steps).
    """
    starts = get_last_character_start_index(y_start)
    for trial, start in enumerate(starts):
        end_of_sentence[trial] = min(end_of_sentence[trial], start + max_pause)
    return end_of_sentence


def get_last_character_start_index(y_start: ndarray) -> ndarray:
    """Gets index of the last character start index in a sentence.

    Args:
        y_start: An array of shape (trials, time_steps) where a
            1 indicates the start of a character.  Due to the way
            the start signal is created, 1s will be extended for
            some number of time steps from the start of the character

    Returns:
        The index of the last character start index in a sentence of
        shape (trials,).
    """
    y_start = np.concatenate([np.zeros((y_start.shape[0], 1)), y_start], axis=1)
    starts = []
    for i in range(y_start.shape[0]):
        nonzero = np.nonzero(np.diff(y_start[i]) > 0)[0]
        if len(nonzero) > 0:
            starts.append(nonzero[-1])
        else:
            starts.append(0)
    # starts = np.array([np.nonzero(np.diff(t) > 0)[0][-1] for t in y_start])
    return np.array(starts)
