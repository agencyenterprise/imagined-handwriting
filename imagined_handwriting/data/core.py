"""Load the handwriting data."""
from typing import Optional, Tuple

import numpy as np

from imagined_handwriting.config import HandwritingDataConfig
from imagined_handwriting.data.download import HandwritingDownload
from imagined_handwriting.data.handwriting import HandwritingSessionData
from imagined_handwriting.data.io import (
    load_character,
    load_labels,
    load_sentence,
    load_splits,
)
from imagined_handwriting.data.preprocess import Preprocessor
from imagined_handwriting.data.raw import Raw


def load_raw(
    root: str, session_id: str, holdout="blocks", download: bool = True
) -> Raw:
    """Loads the raw data for a session from disk

    Args:
        root (str): The root directory of the data
        session_id (str): The session id for this session

    Returns:
        Raw: The raw data for the session
    """
    # download the data if it doesn't exist
    downloader = HandwritingDownload(root, download=download)
    if not downloader.check_exists():
        raise ValueError(
            "Data does not exist. Please download it first with download=True."
        )
    sentence = load_sentence(session_id, root)
    label = load_labels(session_id, root)
    character = load_character(session_id, root)
    split = load_splits(root, holdout=holdout)

    return Raw(
        session_id, sentence=sentence, label=label, character=character, split=split
    )


def preprocess(raw: Raw, **kwargs) -> HandwritingSessionData:
    """Preprocess the raw data.

    Args:
        raw (Raw): The raw data object.
        **kwargs: Keyword arguments to pass that will be passed to HandwritingDataConfig
            and will override any default values.

    Returns:
        HandwritingSessionData: The preprocessed data.
    """
    config = HandwritingDataConfig(**kwargs).dict()
    preprocessor = Preprocessor(config)
    return preprocessor(raw)


def load_preprocessed(
    root: str, session_id: str, download: bool = True, **kwargs
) -> HandwritingSessionData:
    """Loads the data from disk and preprocesses it.

    Args:
        root (str): The root directory of the data.
        **kwargs: Keyword arguments to pass that will be passed to
            HandwritingDataConfig and will override any default values.

    Returns:
       HandwritingSessionData: The preprocessed data.
    """
    downloader = HandwritingDownload(root, download=download)
    if not downloader.check_exists():
        raise ValueError(
            "Data does not exist. Please download it first with download=True."
        )
    config = HandwritingDataConfig(**kwargs).dict()
    raw_data = load_raw(root=root, session_id=session_id, holdout=config["holdout"])
    return preprocess(raw_data, **config)


def make_splits(
    data: HandwritingSessionData,
    n_train: Optional[int] = None,
    include_test_in_train: bool = False,
) -> Tuple[HandwritingSessionData, ...]:
    """Splits the data into train, validation and test sets.

    The data splitting is complicated by the fact that the in some cases we want to
    include the test data in the training set, for example when we are fine-tuning on
    session 4 and want to include the test data in session 1,2,3 so that we have more
    training data.

    A further complication is the fact that some of the test sentences are repeated
    across each session.  These are sentences taken from Pandarinath et al., 2017
    and these should never be included in the training set, so even when we want
    to include the test data in the train set, we still need to filter out these
    repeated sentences (so we don't overfit them).  In this case the repeated
    test sentences are placed in the validation set.

    Args:
        data (HandwritingSessionData): The data to split.
        n_train (Optional[int], optional): The number of training samples. If None,
            use allof the training data (i.e. no validation examples from this set).
            Defaults to None.
        include_test_in_train (bool, optional): Whether to include the test data in
            the training set.If true, the test sentences are included in the training
            data for this session, excluding the repeated test sentences which will
            be placed in the validation set.  See the docstring.Defaults to False.
    """
    train_index = data.train_index
    if n_train is not None:
        val_index = train_index[n_train:]
        train_index = train_index[:n_train]
    else:
        val_index = np.array([])

    test_index = data.test_index
    blacklist_from_train_index = np.array(
        [i for i, x in enumerate(data.condition) if x == "CL Pandarinath"]
    )

    if include_test_in_train:
        train_index = np.concatenate([train_index, test_index])
        train_index = np.setdiff1d(train_index, blacklist_from_train_index)
        if len(val_index) > 0:
            val_index = np.concatenate([val_index, blacklist_from_train_index])
        else:
            val_index = blacklist_from_train_index
        test_index = np.array([])

    return data.split(train_index, val_index, test_index)
