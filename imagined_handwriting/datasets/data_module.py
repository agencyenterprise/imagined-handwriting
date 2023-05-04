from typing import Callable, List, Optional

import numpy as np
import pytorch_lightning as pl
from numpy.random import Generator
from torch.utils.data import DataLoader

from imagined_handwriting import transforms as T
from imagined_handwriting.config import HandwritingDataConfig
from imagined_handwriting.data import load_preprocessed, make_splits
from imagined_handwriting.data.download import HandwritingDownload
from imagined_handwriting.data.handwriting import HandwritingSessionData
from imagined_handwriting.data.io import load_corpus
from imagined_handwriting.datasets.handwriting import (
    HandwritingDatasetForTraining,
    SentenceDataset,
    SyntheticDataset,
)
from imagined_handwriting.datasets.utils import parse_session_id


class HandwritingDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for the Imagined Handwriting dataset.

    The imagined handwriting data is collected over multiple sessions, each
    session is uniquely identified by the date the session was recorded.  Each
    session contains a set of sentences which act as the target that the participant
    imagines writing, along with the binned thresholded spike counts of neural
    data while the participant imagines writing the sentence.  The process of
    aligning the neural data to the sentence is described in the paper and these
    labels are included in the downloaded dataset.

    This class will download the dataset if it doesn't exist to the folder
    specified by the root argument.

    The training dataset created by this data module combines data across all
    requested sessions. Each batch returned by the train dataloader will be
    homogenous with respect to session.  Each element in the batch will be
    a random crop of the sentence data of a fixed size, specified by the kwarg
    `crop_size`, default is 200.  This size should match the expected sequence
    length of the downstream model.  The training data is also augmented by
    default via random transformations applied to each batch.  This can be
    turned off by passing `transform=None` to the constructor.  The training
    data also includes synthetic data by default. The synthetic data is created
    by extracting character snippets from the sentences and then "gluing them
    together" to create new sentences.  The snippets are also augmented (before
    gluing) by default. This can be turned off by passing `synthetic_transform=None`
    to the constructor.  The number of synthetic examples in each batch is
    controlled by the `synthetic_per_batch`.  The default is a batch size of
    64 and a synthetic_per_batch of 24.  To turn off synthetic data pass
    `synthetic_per_batch=0`.  The training dataset is an infinite iterable
    so in particular when training you should set `max_steps` instead of `max_epochs`
    to limit the training iterations.

    Important:
        The validation and test set are not augmented or cropped and return
        batches of full sentences.  This means that the sequence length of these
        sets is much larger than the cropped training data.  Care must be taken
        when training a model with a fixed input size to ensure that the validation
        and test data handled correctly (see models.handwriting for an example).

    There are two modes for this dataset, a "pretrain" mode and a "fine tune" mode.
    In the pretrain mode the each session of data is sampled uniformly (when
    creating a batch of data we sample a session for that batch) and the
    test data is withheld from the training data.  In the fine tune mode the
    training data is sampled from the last session with proportion
    `fine_tune_weight` (default is 0.5) and the test data is included in the
    all but the last session.


    You must also provide a session or list of sessions to train against.
    There are several ways to specify which sessions to use.
        * comma separated string of ids: e.g "t5.2019.05.08,t5.2019.11.25"
        * a single id: e.g. "t5.2019.05.08"
        * a single id a comparison: e.g. "<=t5.2019.12.11" which will return
            all the sessions up to and including t5.2019.12.11
        * a named subset of data, one of
            ["pretrain", "copy_typing", "free_typing", "all"]

    """

    def __init__(
        self, root: str, session_ids: List[str], *, pretrain, download=True, **kwargs
    ):
        """Initialize the data module.

        Args:
            root (str): The root directory where the data will be downloaded
                or a root directory previously used (and so the data
                exists there).
            session_id (str): The session id to use.  See the class docstring
                for more details.
            pretrain (bool): If True, we load data in "pretrain mode" which
                means that all datasets are sampled uniformly and that the
                test data is withheld from the training data.  If False, we
                are in "fine tune mode" which means that the training data
                is sampled from the last session with proportion
                config['fine_tune_weight']
                and the test data is included in the training data for all
                the sessions except the last.
            download (bool): If True, download the data if it doesn't exist.
            kwargs: Additional keyword arguments that will override values
                in the config object. See `config.HandwritingDataConfig` for
                all options.
        """
        super().__init__()
        self.root = root
        self.download = download
        self.session_ids = session_ids
        self.pretrain = pretrain
        config = HandwritingDataConfig(**kwargs).dict()
        self.config = config
        hparams = dict(
            session_ids=self.session_ids,
            pretrain=pretrain,
            **config,
        )
        self.save_hyperparameters(hparams)

    def prepare_data(self) -> None:
        """Download the data if it doesn't exist."""
        downloader = HandwritingDownload(self.root, download=self.download)
        self.raw_folder = downloader.raw_folder
        self.corpus = load_corpus(self.root)

    def setup(self, stage: Optional[str] = None):
        """Setup the data module for training and testing."""
        preprocessed = self.load_preprocessed()
        if stage == "fit":
            self.train_dataset = self.prepare_dataset(preprocessed, "train")
            self.val_datasets = self.prepare_dataset(preprocessed, "val")
        elif stage == "test":
            self.test_datasets = self.prepare_dataset(preprocessed, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 0),
            pin_memory=self.config.get("pin_memory", False),
            shuffle=False,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                num_workers=self.config.get("num_workers", 0),
                pin_memory=self.config.get("pin_memory", False),
                shuffle=False,
            )
            for ds in self.val_datasets
            if len(ds) > 0
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                num_workers=self.config.get("num_workers", 0),
                pin_memory=self.config.get("pin_memory", False),
                shuffle=False,
            )
            for ds in self.test_datasets
        ]

    def load_preprocessed(self) -> List[HandwritingSessionData]:
        """Load the preprocessed data.

        Returns:
            data (dict): A dictionary containing the preprocessed data.
        """
        return [
            load_preprocessed(self.root, session_id=sess, **self.config)
            for sess in self.session_ids
        ]

    def prepare_dataset(self, preprocessed: List[HandwritingSessionData], split: str):
        """Prepare the dataset for training or testing.

        Args:
            split (str): The split to prepare.  This can be "train",
                "val", or "test".

        Returns:
            dataset (torch.utils.data.Dataset): The dataset for the
                given split.
        """
        config = self.config.copy()
        if split == "train":
            transform = self.get_transform()
            snippet_transform = self.get_snippet_transform()
        else:
            transform = None
            snippet_transform = None

        train = []
        val = []
        test = []
        n_train = config["n_train"]

        for i, d in enumerate(preprocessed):
            include_test_in_train = True
            # if we are fine-tuning then include the test data
            # for all but the last session.
            if not self.pretrain and i == len(self.session_ids) - 1:
                include_test_in_train = False
                if n_train is None:
                    n_train = int(0.8 * len(d.train_index))
            _train, _val, _test = make_splits(d, n_train, include_test_in_train)
            train.append(_train)
            val.append(_val)
            test.append(_test)

        if split == "train":
            data = train
        elif split == "val":
            data = val
        elif split == "test":
            data = test

        trim_end_of_sentence = split == "train"
        sentence_ds = [
            SentenceDataset(
                d,
                transform=transform,
                seed=config["seed"],
                trim_end_of_sentence=trim_end_of_sentence,
            )
            for d in data
        ]

        if split == "train":
            synth_ds = [
                SyntheticDataset(
                    d,
                    split,
                    transform=transform,
                    snippet_transform=snippet_transform,
                    seed=config["seed"],
                    corpus=self.corpus,
                    one_hot=config["one_hot"],
                )
                for d in data
            ]
            dataset = HandwritingDatasetForTraining(
                sentence_ds,
                synth_ds,
                synthetic_per_batch=self.config["synthetic_per_batch"],
                batch_size=self.config["batch_size"],
                sample_weights=self.get_sample_weights(len(preprocessed)),
                seed=config["seed"],
            )
            return dataset

        # if we are not training then return the sentence datasets
        # with no synthetic data.
        return sentence_ds

    def get_sample_weights(self, n_datasets):
        """Get the sample weights for the datasets."""
        if self.pretrain or self.config["fine_tune_weight"] is None:
            return None

        remaining_weight = 1 - self.config["fine_tune_weight"]
        if remaining_weight < 0 or remaining_weight > 1:
            raise ValueError("fine_tune_weight must be in [0,1]")
        distributed = remaining_weight / (n_datasets - 1)
        weights = [distributed for _ in range(n_datasets)]
        weights[-1] = self.config["fine_tune_weight"]
        assert 0.9999 < sum(weights) < 1.0001
        return weights

    def get_transform(self):
        """Get the transformation for the dataset.

        If self.transform == "default" then we return a "standard"
        transformation that includes padding, cropping, and adding
        noise.  Otherwise we return self.transform (which could be None)

        Returns:
            transform (Callable): The transformation to apply to the
                dataset with signature transform(x:dict, Optional[Generator]=None).
                where the optional generator is a numpy generator that can
                be used to control the randomness of the transformations.
        """
        config = self.config
        if config.get("transform") != "default":
            return config.get("transform")

        return default_transform(config)

    def get_snippet_transform(self):
        """Get the transformation for snippets.

        If self.snippet_transform == "default" then we return a "standard"
        transformation that includes time warping and scaling.
        Otherwise we return self.snippet_transform (which could be None)

        Returns:
            transform (Callable): The transformation to apply to the
                snippets independently for each sampled sentences.
                The signature is transform(x:ndarray, Optional[Generator]=None)
                where the optional generator is a numpy generator that can
                be used to control the randomness of the transformations.

        """
        config = self.config
        if config.get("snippet_transform") != "default":
            return config.get("snippet_transform")

        return default_snippet_transform(config)


class PretrainDataModule(HandwritingDataModule):
    """Data module for pretraining.

    In the original paper the authors pre-trained a model using the first
    3 pilot sessions.  They then fine-tuned this pre-trained model on
    the next session and used the fine-tune model for real-time
    copy typing decoding.  This class is used to get the pretraining
    data for a given session.

    By default this will return the first 3 pilot sessions.  If a
    session id is specified other than "pretrain" then it will
    return data for all session up to but *not* including the
    specified session.
    """

    def __init__(
        self,
        root: str,
        session_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize the PretrainData module.

        Args:
            root (str): The root directory where the data will be downloaded
                or a root directory previously used (and so the data
                exists there).
            session_id (str): The session ids to use.
            config (dict): The configuration object for the dataset.  See
                datasets.config for options. If None then the defaults
                will be provided by `config.HandwritingDataConfig`.
            kwargs: Additional keyword arguments that will override values
                in the config object.
        """
        if session_ids is None:
            session_ids = parse_session_id("pretrain")
        super().__init__(root, session_ids, pretrain=True, **kwargs)


class FineTuneDataModule(HandwritingDataModule):
    """Data module for fine-tuning.

    In the original paper the authors pre-trained a model using the first
    3 pilot sessions.  They then fine-tuned this pre-trained model on
    the next session and used the fine-tune model for real-time
    copy typing decoding.  This class is used to get the pretraining
    data for a given session.

    To include past data while fine-tuning make sure to include them
    with in the session_id.  For example, if you want to fine-tune on
    session "t5.2019.12.11" then pass in "<=t5.2019.12.11" as the
    session_id to include the past sessions during fine-tuning.  The
    weight for which the latest session is sampled can be set with
    config['fine_tune_weight'] or by passing in the keyword argument
    fine_tune_weight=<float>

    """

    def __init__(
        self,
        root: str,
        session_ids: List[str],
        **kwargs,
    ):
        """Initialize the PretrainData module.

        Args:
            root (str): The root directory where the data will be downloaded
                or a root directory previously used (and so the data
                exists there).
            session_id (str): The session id to use.  Assumes the last id
                is the one used for fine-tuning.
            config (dict): The configuration object for the dataset.  See
                datasets.config for options. If None then the defaults
                will be provided by `config.HandwritingDataConfig`.
            kwargs: Additional keyword arguments that will override values
                in the config object.
        """
        super().__init__(root, session_ids, pretrain=False, **kwargs)


class AllSessionsDataModule(HandwritingDataModule):
    """Data module for all sessions.

    This is used to get data for all sessions simultaneously.
    The training takes place as if we have all sessions available
    at the same time and the test set is split from each session.

    This is analogous to the training setup in the github repo
    associated with the paper.  The metrics reported in the repo
    come from this setup.

    See:
        https://github.com/fwillett/handwritingBCI
    """

    def __init__(
        self,
        root: str,
        **kwargs,
    ):
        """Initialize the PretrainData module.

        Args:
            root (str): The root directory where the data will be downloaded
                or a root directory previously used (and so the data
                exists there).
            config (dict): The configuration object for the dataset.  See
                datasets.config for options. If None then the defaults
                will be provided by `config.HandwritingDataConfig`.
            kwargs: Additional keyword arguments that will override values
                in the config object.
        """
        session_ids = parse_session_id("all")
        super().__init__(root, session_ids, pretrain=True, **kwargs)


def default_transform(config: dict) -> Callable:
    """Creates a default data augmentation transform.

    Args:
        config (dict): The configuration object for the default transformation.
    """

    def _transform(item, rng: Optional[Generator] = None):
        crop_keys = ["x", "y_start", "y_char", "loss_mask"]
        for k in crop_keys:
            item[k] = T.pad(item[k], pad_width=config["pad_width"], axis=0)
        item = T.random_crop(item, config["crop_size"], axis=0, keys=crop_keys, rng=rng)
        item["loss_mask"] = T.mask_loss_before_first_character(
            item["loss_mask"], item["y_start"]
        )
        item["x"] = T.white_noise(item["x"], config["white_noise_std"], rng=rng)
        item["x"] = T.offset_noise(
            item["x"], config["offset_noise_std"], axis=1, rng=rng
        )
        item["x"] = T.random_walk_noise(
            item["x"], config["random_walk_noise_std"], axis=0, rng=rng
        )
        if config["smooth"]:
            item["x"] = T.smooth(item["x"], kernel_std=4 / config["bin_width"])

        item["x"] = item["x"].astype(np.float32)

        return item

    return _transform


def default_snippet_transform(config: dict) -> Callable:
    """Creates a default snippet transformation.

    Args:
        config (dict): The configuration object for the default transformation.
    """

    def _snippet_transform(snippet, rng: Optional[Generator] = None):
        snippet = T.random_timewarp(
            snippet,
            low=config["snippet_timewarp_low"],
            high=config["snippet_timewarp_high"],
            rng=rng,
        )
        snippet = T.random_scale(
            snippet,
            low=config["snippet_scale_low"],
            high=config["snippet_scale_high"],
            rng=rng,
        )
        return snippet

    return _snippet_transform
