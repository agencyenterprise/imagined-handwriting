from typing import List

import numpy as np
from numpy import ndarray


class Raw:
    """A container for raw data.

    The relevant data is stored in 4 .mat files for each session:
        1. sentence.mat
        2. {session_id}_timeSeriesLabels.mat
        3. singleLetter.mat
        4. trainTestPartitions_HeldOutBlocks.mat

    This class holds all three and provides retrieval with minor formatting
    and filtering.  For example, all data is stored in 2d arrays even when it
    is naturally 1d, we flatten those before returning them.  We also provide
    easy access to the train/test split data.

    By default no sentences are returned that have errors since we filter them
    out by only including sentences in either the train or test splits.  If
    access to the error sentences is needed then use the raw underlying data
    attributes, e.g. `sentences`.
    """

    def __init__(
        self,
        session_id: str,
        *,
        sentence: dict,
        label: dict,
        character: dict,
        split: dict,
    ):
        """Initializes the Raw class

        This class is specialized to the data format of the data released
        with the paper: "High-performance brain-to-text communication via handwriting"

        Reference:
            Willett, Francis et al. (2021), Data from: High-performance brain-to-text
            communication via handwriting, Dryad, Dataset,
            https://doi.org/10.5061/dryad.wh70rxwmv

        Args:
            session_id (str): The session id for this session
                of raw data.  E.g. "t5.2019.12.11"
            sentence (dict): The raw sentence data as loaded from
                disk via scipy.io.loadmat.
            label (dict): The raw label data as loaded from disk
                via scipy.io.loadmat.
            character (dict): The raw character data as loaded from
                disk via scipy.io.loadmat.
            split (dict): The raw split data as loaded from disk
                via scipy.io.loadmat.

        """
        self.session_id = session_id
        self.sentence = sentence
        self.label = label
        self.character = character
        self.split = split

    @property
    def neural_data(self) -> ndarray:
        return self.sentence["neuralActivityCube"].copy()

    @property
    def y_start(self) -> ndarray:
        return self.label["charStartTarget"].copy()

    @property
    def y_char(self) -> ndarray:
        return self.label["charProbTarget"].copy()

    @property
    def loss_mask(self) -> ndarray:
        return 1 - self.label["ignoreErrorHere"].copy()

    @property
    def end_of_sentence_index(self) -> ndarray:
        """Returns the index of the end of sentence token"""
        return self.sentence["numTimeBinsPerSentence"].flatten()

    @property
    def text(self) -> List[str]:
        """Returns the text for each sentence"""
        text = self.sentence["sentencePrompt"].flatten()
        return [x[0] for x in text]

    @property
    def sentence_blocks(self) -> ndarray:
        """Returns the block list for the sentences"""
        return self.sentence["sentenceBlockNums"].flatten()

    @property
    def character_blocks(self) -> ndarray:
        """Returns the block ids for the character data"""
        return np.sort(self.character["blockList"].flatten())

    @property
    def character_block_means(self) -> ndarray:
        """Returns the block means for the character data"""
        return self.character["meansPerBlock"]

    @property
    def character_block_std(self) -> ndarray:
        """Returns the block standard deviations for the character data"""
        return self.character["stdAcrossAllData"].flatten()

    @property
    def blank_windows(self) -> List[List[ndarray]]:
        """Returns the blank windows for the session"""
        return [
            [x.flatten() for x in y.flatten()]
            for y in self.label["blankWindows"].flatten()
        ]

    @property
    def condition(self) -> List[str]:
        """Returns the condition.

        This can be used to filter out sentences which are repeated in the test set,
        i.e. the sentences which are taken from Pandarinath et al., 2017 for direct
        comparison.
        """
        return [x[0] for x in self.sentence["sentenceCondition"].flatten()]

    @property
    def start_index(self) -> List[ndarray]:
        start_index = self.label["letterStarts"].astype(int)
        return [x[x != 0] for x in start_index]

    @property
    def train_index(self) -> ndarray:
        """Returns the train index"""
        return self.split[f"{self.session_id}_train"].flatten()

    @property
    def test_index(self) -> ndarray:
        """Returns the test index"""
        return self.split[f"{self.session_id}_test"].flatten()

    @property
    def trial_index(self) -> ndarray:
        """Returns the trial index"""
        return np.arange(self.sentence["neuralActivityCube"].shape[0])
