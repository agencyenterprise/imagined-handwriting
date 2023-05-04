from typing import List, Optional, Tuple

import numpy as np
from numpy import ndarray

from imagined_handwriting import transforms as T
from imagined_handwriting.data.handwriting import HandwritingSessionData
from imagined_handwriting.data.raw import Raw


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, raw: Raw) -> HandwritingSessionData:
        """Preprocess the raw data.

        Args:
            raw: The raw data to preprocess.
        """
        x = raw.neural_data
        y_start = raw.y_start
        y_char = raw.y_char
        loss_mask = raw.loss_mask
        text = raw.text
        end_of_sentence_index = raw.end_of_sentence_index
        blank_windows = raw.blank_windows
        start_index = raw.start_index
        config = self.config

        # fix up EOS and loss mask
        end_of_sentence_index = T.remove_pause_from_end_of_sentence(
            end_of_sentence_index, y_start, config["max_pause"]
        )
        loss_mask = self.process_loss_mask(
            loss_mask, end_of_sentence_index, blank_windows
        )

        if config["normalize"]:
            x = T.normalize(
                x,
                sentence_blocks=raw.sentence_blocks,
                character_blocks=raw.character_blocks,
                block_means=raw.character_block_means,
                block_std=raw.character_block_std,
            ).astype(np.float32)
        x, y_start, y_char, loss_mask, end_of_sentence_index, start_index = self.bin(
            x, y_start, y_char, loss_mask, end_of_sentence_index, start_index
        )

        if not config["one_hot"]:
            y_char = T.sparse(y_char, axis=2)

        if config["embed_channels"]:
            if config["normalize"]:
                raise ValueError(
                    "Only one of `normalize` and `embed_channels` can be true ."
                )
            else:
                x = T.channel_to_embedding_index(x, config["max_firing_rate"])

        if (
            not config["normalize"]
            and config["embed_channels"]
            and config["max_firing_rate"] is not None
        ):
            x = T.clip_to_max_firing_rate(x, config["max_firing_rate"])

        return HandwritingSessionData(
            session_id=raw.session_id,
            x=x,
            y_start=y_start,
            y_char=y_char,
            loss_mask=loss_mask,
            text=text,
            end_of_sentence_index=end_of_sentence_index,
            blank_windows=blank_windows,
            start_index=start_index,
            condition=raw.condition,
            train_index=raw.train_index,
            test_index=raw.test_index,
            bin_width=self.config["bin_width"],
            trial_index=raw.trial_index,
        )

    def bin(
        self, x, y_start, y_char, loss_mask, end_of_sentence_index, start_index
    ) -> Tuple:
        """Bins the data."""
        bin_width = self.config["bin_width"]
        mean_bin = T.Bin(bin_width, method="mean", axis=1)
        max_bin = T.Bin(bin_width, method="max", axis=1)
        x = mean_bin(x)
        if not self.config["normalize"]:
            x = np.round(x).astype(np.int32)
        y_start = max_bin(y_start)
        y_char = mean_bin(y_char)
        loss_mask = max_bin(loss_mask)
        if bin_width > 1:
            end_of_sentence_index = end_of_sentence_index // bin_width
            start_index = [s // bin_width for s in start_index]
        return x, y_start, y_char, loss_mask, end_of_sentence_index, start_index

    def process_loss_mask(
        self,
        mask: ndarray,
        end_of_sentence_index: ndarray,
        blank_windows: Optional[List[List[ndarray]]] = None,
    ) -> ndarray:
        """Process the loss mask.

        Args:
            mask: The loss mask.
            end_of_sentence_index: The end of sentence index.
            blank_windows: The blank windows. Default is None.

        """
        if self.config["mask_blank_windows"]:
            if blank_windows is None:
                raise ValueError("Blank windows must be provided to mask them.")
            mask = T.mask_loss_on_blank_windows(mask, blank_windows)
        mask = T.mask_loss_after_last_character(mask, end_of_sentence_index)
        return mask
