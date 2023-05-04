"""Script to make a small amount of test data.
"""
import argparse
import shutil
from pathlib import Path

import scipy.io

DATASET = Path("handwritingBCIData/Datasets")
TRAINING_STEPS = Path("handwritingBCIData/RNNTrainingSteps")
LABELS = Path("handwritingBCIData/RNNTrainingSteps/Step2_HMMLabels/HeldOutBlocks")


def name_to_filename(name, session_id):
    options = {
        "sentence": f"{session_id}/sentences.mat",
        "character": f"{session_id}/singleLetters.mat",
        "labels": f"{session_id}_timeSeriesLabels.mat",
        "splits": "trainTestPartitions_HeldOutBlocks.mat",
    }
    return options[name]


def name_to_path(name, session_id):
    options = {
        "sentence": DATASET / name_to_filename(name, session_id),
        "character": DATASET / name_to_filename(name, session_id),
        "labels": LABELS / name_to_filename(name, session_id),
        "splits": TRAINING_STEPS / name_to_filename(name, session_id),
    }
    return options[name]


def keep_keys(name):
    options = {
        "sentence": [
            "neuralActivityCube",
            "sentenceBlockNums",
            "sentencePrompt",
            "numTimeBinsPerSentence",
            "blockList",
            "sentenceCondition",
        ],
        "character": [
            "meansPerBlock",
            "stdAcrossAllData",
            "blockList",
        ],
        "labels": [
            "charStartTarget",
            "charProbTarget",
            "ignoreErrorHere",
            "blankWindows",
            "letterStarts",
        ],
        "splits": [
            "t5.2019.05.08_train",
            "t5.2019.05.08_test",
            "t5.2019.12.11_train",
            "t5.2019.12.11_test",
        ],
    }
    return options[name]


def main():
    help = (
        "Top level directory containing extracted dryad archive.\n"
        "You can obtain a copy from "
        "https://datadryad.org/stash/downloads/file_stream/662811"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help=help)
    args = parser.parse_args()
    root = Path(args.path)

    for session_id in ["t5.2019.05.08", "t5.2019.12.11"]:
        for name in ["sentence", "character", "labels", "splits"]:
            path = root / name_to_path(name, session_id)
            data = scipy.io.loadmat(path)
            data = {k: v for k, v in data.items() if k in keep_keys(name)}
            if "neuralActivityCube" in data:
                data["neuralActivityCube"] = data["neuralActivityCube"][:, :500, :]
            if "charStartTarget" in data:
                data["charStartTarget"] = data["charStartTarget"][:, :500]
            if "charProbTarget" in data:
                data["charProbTarget"] = data["charProbTarget"][:, :500]
            if "ignoreErrorHere" in data:
                data["ignoreErrorHere"] = data["ignoreErrorHere"][:, :500]
            save_path = Path(__file__).parent / "data"
            save_file = save_path / name_to_path(name, session_id)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            scipy.io.savemat(save_file, data)

    shutil.copy2(
        root / "google-10000-english-usa.txt",
        save_path / "google-10000-english-usa.txt",
    )


if __name__ == "__main__":
    main()
