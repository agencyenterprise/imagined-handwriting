import pytest

from imagined_handwriting.data.raw import Raw


@pytest.fixture
def raw(test_sessions, sentence, label, character, split):
    objs = []
    for i in range(2):
        objs.append(
            Raw(
                test_sessions[i],
                sentence=sentence[i],
                label=label[i],
                character=character[i],
                split=split,
            )
        )
    return objs


@pytest.mark.parametrize("session", [0, 1])
def test_raw_train_index_is_1d(session, raw):
    """Tests that the train index is returned"""
    assert raw[session].train_index.ndim == 1


@pytest.mark.parametrize("session", [0, 1])
def test_raw_returns_test_index_is_1d(session, raw):
    """Tests that the test index is returned"""
    assert raw[session].test_index.ndim == 1


@pytest.mark.parametrize("session", [0, 1])
def test_neural_data_is_correct_dimensions(session, raw):
    """Tests that the neural data is returned"""
    assert raw[session].neural_data.ndim == 3


@pytest.mark.parametrize("session", [0, 1])
def test_neural_data_has_correct_num_channels(session, raw):
    """Tests that the neural data has the correct number of channels"""
    assert raw[session].neural_data.shape[2] == 192
