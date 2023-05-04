import numpy as np
from numpy import ndarray


def normalize(
    neural_activity: ndarray,
    *,
    sentence_blocks: ndarray,
    character_blocks: ndarray,
    block_means: ndarray,
    block_std: ndarray
) -> ndarray:
    """Normalizes the neural activity.

    The neural activity is normalized using the mean activity in the closest
    "experimental block" and the standard deviation across all blocks.

    Args:
        neural_activity: An array of shape (trials, timesteps, channels) of
            recorded neural activity.
        sent_blocks: An array of shape (trials,) which gives the block id
            of the trial.
        char_blocks: An array of shape (unique_char_trials, ) which gives the
            unique block ids of the single character trials.
        block_means: An array of shape (unique_char_trials,channels) of
            mean activity taken over the character blocks.
        std: An array of shape (channels,) of the standard deviation of activity
            taken over all blocks, for each channel.

    Returns:
        An array of normalized neural activity.
    """
    neural_activity = neural_activity.astype(np.float32)
    block_mean_index = closest_block_indices(sentence_blocks, character_blocks)
    mu = np.expand_dims(block_means[block_mean_index], 1)  # (trials, 1, channels)
    sigma = np.expand_dims(block_std, (0, 1))  # (1,1, channels)
    return z_score_normalization(neural_activity, mu, sigma)


def closest_block_indices(sent_blocks: ndarray, char_blocks: ndarray) -> ndarray:
    """Finds the closest character block to each sentence block.

    Args:
        sent_blocks: An array of shape (trials,) giving the id of the block
            to which the trial belongs.
        char_blocks: An array of shape (B,) where B is the number of character
            blocks, giving the id of the character block.
    """
    return np.array(
        [closest_block(b, char_blocks) for b in sent_blocks], dtype=np.int32
    )


def closest_block(trial_block: int, char_blocks: ndarray) -> int:
    """Finds the closets character block to the current trial block.

    Args:
        trial_block: The block index of a trial
        char_blocks: The block index of all character trials.

    Returns:
        The block index of the closest character block to the current trial block.
    """
    char_blocks = char_blocks.astype(np.int32)
    return int(np.argmin(np.abs(char_blocks - trial_block)))


def z_score_normalization(a, mean, std):
    return (a - mean) / std
