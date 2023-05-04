from pathlib import Path

import pytest
import scipy.io

DOWNLOAD_DIR = Path("ImaginedHandwriting/raw")


@pytest.fixture(scope="session")
def test_sessions():
    """Returns a list of test sessions"""
    return ["t5.2019.05.08", "t5.2019.12.11"]


@pytest.fixture
def CHANNELS():
    """Returns the number of channels"""
    return 192


@pytest.fixture
def SAMPLES():
    """Returns the number of time samples per sentence.

    The test data was generated with 500 samples per sentence.
    """
    return 500


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"


@pytest.fixture
def sentence(data_path, test_sessions):
    dataset_path = DOWNLOAD_DIR / "handwritingBCIData/Datasets"
    return [
        scipy.io.loadmat(data_path / dataset_path / s / "sentences.mat")
        for s in test_sessions
    ]


@pytest.fixture
def character(data_path, test_sessions):
    dataset_path = DOWNLOAD_DIR / "handwritingBCIData/Datasets"
    return [
        scipy.io.loadmat(data_path / dataset_path / s / "singleLetters.mat")
        for s in test_sessions
    ]


@pytest.fixture
def label(data_path, test_sessions):
    label_path = (
        DOWNLOAD_DIR
        / "handwritingBCIData/RNNTrainingSteps/Step2_HMMLabels/HeldOutBlocks"
    )
    return [
        scipy.io.loadmat(data_path / label_path / f"{s}_timeSeriesLabels.mat")
        for s in test_sessions
    ]


@pytest.fixture
def split(data_path):
    split_path = DOWNLOAD_DIR / "handwritingBCIData/RNNTrainingSteps"
    return scipy.io.loadmat(
        data_path / split_path / "trainTestPartitions_HeldOutBlocks.mat"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pause: disables the patch no_pause for snippet sampling.",
    )
