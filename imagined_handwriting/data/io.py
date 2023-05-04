from pathlib import Path
from typing import List

import scipy.io

from imagined_handwriting.settings import DOWNLOAD_DIR

DATASET = "handwritingBCIData/Datasets"
TRAINING_STEPS = "handwritingBCIData/RNNTrainingSteps"
LABELS = "handwritingBCIData/RNNTrainingSteps/Step2_HMMLabels/HeldOutBlocks"


def load_sentence(session_id: str, root: str) -> dict:
    """Load the sentence data from a file"""
    path = Path(root) / DOWNLOAD_DIR / DATASET / session_id / "sentences.mat"
    return scipy.io.loadmat(path)


def load_character(session_id: str, root: str) -> dict:
    """Load the character data from a file"""
    path = Path(root) / DOWNLOAD_DIR / DATASET / session_id / "singleLetters.mat"
    return scipy.io.loadmat(path)


def load_labels(session_id: str, root: str) -> dict:
    """Load the label data from a file"""
    path = Path(root) / DOWNLOAD_DIR / LABELS / f"{session_id}_timeSeriesLabels.mat"
    return scipy.io.loadmat(path)


def load_splits(root: str, holdout="blocks") -> dict:
    """Load the split data from a file"""
    holdout = "HeldOutBlocks" if holdout == "blocks" else "HeldOutTrials"
    path = (
        Path(root)
        / DOWNLOAD_DIR
        / TRAINING_STEPS
        / f"trainTestPartitions_{holdout}.mat"
    )
    return scipy.io.loadmat(path)


def load_corpus(root: str) -> List[str]:
    """Load the corpus data from a file"""
    path = Path(root) / DOWNLOAD_DIR / "google-10000-english-usa.txt"
    with open(path, "r") as f:
        return [x.strip() for x in f.readlines()]
