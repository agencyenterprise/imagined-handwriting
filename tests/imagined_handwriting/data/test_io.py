import pytest

from imagined_handwriting.data import io as data_io

TEST_SESSIONS = ["t5.2019.05.08", "t5.2019.12.11"]


@pytest.mark.parametrize("session_id", TEST_SESSIONS)
def test_load_character(session_id, data_path):
    """Test the load_character function"""
    actual = data_io.load_character(session_id, data_path)
    assert isinstance(actual, dict)
    assert "meansPerBlock" in actual
    assert "stdAcrossAllData" in actual


@pytest.mark.parametrize("session_id", TEST_SESSIONS)
def test_load_sentence(session_id, data_path):
    """Test the load_sentence function"""
    actual = data_io.load_sentence(session_id, data_path)
    assert "neuralActivityCube" in actual
    assert "sentenceBlockNums" in actual
    assert "sentencePrompt" in actual
    assert "numTimeBinsPerSentence" in actual


@pytest.mark.parametrize("session_id", TEST_SESSIONS)
def test_load_labels(session_id, data_path):
    """Test the load_labels function"""
    actual = data_io.load_labels(session_id, data_path)
    assert "charStartTarget" in actual
    assert "charProbTarget" in actual
    assert "ignoreErrorHere" in actual
    assert "blankWindows" in actual


def test_load_splits(data_path):
    """Test the load_splits function"""
    actual = data_io.load_splits(data_path)
    assert "t5.2019.05.08_train" in actual
    assert "t5.2019.05.08_test" in actual
    assert "t5.2019.12.11_train" in actual
    assert "t5.2019.12.11_test" in actual
