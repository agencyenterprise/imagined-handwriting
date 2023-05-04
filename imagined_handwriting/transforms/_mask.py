from typing import List

import numpy as np
from numpy import ndarray


class MaskBeforeFirstCharacter:
    """A loss function that masks before the first full character

    Args:
        loss_fn: The loss function to use.
        blank_value: The value to use to determine if a window is blank.
            Defaults to 0.
    """

    def __call__(self, item: dict) -> dict:
        """Compute the masked loss.

        Args:
            y_hat: The predicted values.
            y: The true values.

        Returns:
            The masked loss.
        """
        loss_mask = item["loss_mask"]
        y_start = item["y_start"]
        loss_mask = mask_loss_before_first_character(loss_mask, y_start)
        item["loss_mask"] = loss_mask
        return item


def mask_loss_before_first_character(mask: ndarray, y_start: ndarray) -> ndarray:
    """Masks the loss before the first character in the sentence.

    We are assuming that the inputs are for a single sentence and both
    of shape (time_steps,).  I.e. this is meant to be used as a on-the-fly
    transform in a dataset.

    Args:
        mask (ndarray): The mask to apply to the loss.
        y_start (ndarray): The start targets.

    Returns:
        ndarray: The loss mask adjusted for the first character.

    """
    mask = mask.copy()
    start_times = np.nonzero(np.diff(y_start) > 0)[0] + 1
    if len(start_times) == 0:
        mask = np.zeros_like(mask)
    else:
        mask[: start_times[0]] = 0
    return mask


def mask_loss_on_blank_windows(
    mask: ndarray, blank_windows: List[List[ndarray]]
) -> ndarray:
    """Masks the loss across blank windows.

    The HMM labeling step labels windows of time as "blanks" if it believes
    no character is being written.  This function masks the loss at this step
    so the neural network does not learn the "wrong" character at these
    time steps.  This was not explicitly mentioned in the paper so this is
    an additional option that may or may not improve performance.

    Args:
        mask (ndarray): The mask to apply to the loss. Shape (sentences, time_steps)
        blank_window (List[List[ndarray]]): The blank windows for each
            session.  The out list indexes sentences and the inner list
            is a list arrays each of which contains the indices for the
            corresponding blank window. Length (sentences,).

    Returns:
        ndarray: The loss mask adjusted for blank windows.

    """
    windows = [[w for w in windows if len(w) > 0] for windows in blank_windows]
    if any(len(w) > 0 for w in windows):
        blank_windows = [np.concatenate(w) for w in windows if len(w) > 0]
        for i, w in enumerate(blank_windows):
            w = np.array([x for x in w if x < mask.shape[1]])  # type: ignore
            mask[i, w] = 0
        return mask
    return mask


def mask_loss_after_last_character(mask: ndarray, y_end: ndarray) -> ndarray:
    """Masks the loss after the end of the sentence.

    We are assuming that the inputs are for a single sentence and both
    of shape (time_steps,).  I.e. this is meant to be used as a on-the-fly
    transform in a dataset.

    Args:
        mask (ndarray): The mask to apply to the loss.
        y_end (ndarray): The end targets.

    Returns:
        ndarray: The loss mask adjusted for the end of the sentence.

    """
    mask = mask.copy()
    for i, end in enumerate(y_end):
        mask[i, end:] = 0
    return mask
