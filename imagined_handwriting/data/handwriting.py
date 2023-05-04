"""
Container for the preprocessed data used in the project.
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray


class HandwritingSessionData:
    """Container for a single session of data."""

    def __init__(
        self,
        *,
        session_id: str,
        x: ndarray,
        y_start: ndarray,
        y_char: ndarray,
        end_of_sentence_index: ndarray,
        loss_mask: ndarray,
        text: List[str],
        blank_windows: List[List[ndarray]],
        start_index: List[ndarray],
        condition: List[str],
        train_index: ndarray,
        test_index: ndarray,
        bin_width: int,
        trial_index: ndarray,
    ):
        self.x = x
        self.y_start = y_start
        self.y_char = y_char
        self.end_of_sentence_index = end_of_sentence_index
        self.loss_mask = loss_mask
        self.text = text
        self.blank_windows = blank_windows
        self.session_id = session_id
        self.start_index = start_index
        self.condition = condition
        self.train_index = train_index
        self.test_index = test_index
        self.bin_width = bin_width
        self.trial_index = trial_index

    def __len__(self):
        return len(self.x)

    def split(self, *args) -> Tuple["HandwritingSessionData", ...]:
        """Split the data into training and testing sets."""
        return tuple(
            HandwritingSessionData(
                x=self.x[index],
                y_start=self.y_start[index],
                y_char=self.y_char[index],
                end_of_sentence_index=self.end_of_sentence_index[index],
                loss_mask=self.loss_mask[index],
                text=[self.text[i] for i in index],
                blank_windows=[self.blank_windows[i] for i in index],
                session_id=self.session_id,
                start_index=[self.start_index[i] for i in index],
                condition=[self.condition[i] for i in index],
                train_index=self.train_index,
                test_index=self.test_index,
                bin_width=self.bin_width,
                trial_index=self.trial_index[index],
            )
            if len(index)
            else HandwritingSessionData.empty()
            for index in args
        )

    def dict(self):
        """Convert the data to a dictionary."""
        return {
            "x": self.x,
            "y_start": self.y_start,
            "y_char": self.y_char,
            "end_of_sentence_index": self.end_of_sentence_index,
            "loss_mask": self.loss_mask,
            "text": self.text,
            "blank_windows": self.blank_windows,
            "session_id": self.session_id,
            "start_index": self.start_index,
            "condition": self.condition,
            "train_index": self.train_index,
            "test_index": self.test_index,
            "bin_width": self.bin_width,
            "trial_index": self.trial_index,
        }

    @classmethod
    def empty(cls):
        return cls(
            x=np.array([]),
            y_start=np.array([]),
            y_char=np.array([]),
            end_of_sentence_index=np.array([]),
            loss_mask=np.array([]),
            text=[],
            blank_windows=[],
            session_id="",
            start_index=[],
            condition=[],
            train_index=np.array([]),
            test_index=np.array([]),
            bin_width=-1,
            trial_index=np.array([]),
        )
