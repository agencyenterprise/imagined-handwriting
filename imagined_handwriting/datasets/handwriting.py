from typing import Callable, List, Optional

import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset, IterableDataset

from imagined_handwriting import transforms as T
from imagined_handwriting.data.handwriting import HandwritingSessionData
from imagined_handwriting.data.snippets import extract_snippets
from imagined_handwriting.datasets.utils import (
    HandwritingTextSampler,
    NumpyRNGMixin,
    SnippetSampler,
)
from imagined_handwriting.settings import SAMPLE_RATE


class HandwritingDatasetForTraining(IterableDataset, NumpyRNGMixin):
    """A pytorch dataset for supervised learning on the imagined handwriting task.

    We recommend using the pytorch-lightning handwriting data module found at
    `datasets.data_module.HandwritingDataModule` instead of using this class directly.
    The datamodule will handle instantiating this class correctly for training and
    testing configurations by correctly instantiating the sentence and synthetic
    datasets (e.g. providing default data augmentation functions during training etc).

    Note this uses a NumpyRNGMixin to provide a random number generator that can be
    used to apply random transformations to the data.
    """

    def __init__(
        self,
        sentence_datasets,  #: Union["SentenceDataset", List["SentenceDataset"]],
        synthetic_datasets,  #:
        synthetic_per_batch: int = 0,
        batch_size: int = 64,
        sample_weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the dataset.

        Args:
            sentence_datasets: A SentenceDataset or list of SentenceDatasets which
                return dictionaries of training examples.
            synthetic_datasets: A SyntheticDataset or list of SyntheticDatasets which
                return dictionaries of training examples from synthetic data.
            synthetic_per_batch: The number of synthetic examples to include in each
                batch.  If this is greater than 0 then the synthetic datasets must be
                provided.
            batch_size: The batch size of the dataset.  This is used to ensure that
                batches are homogenous with respect to the datasets they are drawn from.
            sample_weights: A list of weights for each dataset.  If provided the
                datasets will be sampled with replacement according to these weights.
            seed: The random seed to use to initialize the random number generator that
                is used for random transforms of the data.
        """
        self.sentence_datasets = ensure_list(sentence_datasets)
        self.synthetic_datasets = ensure_list_or_none(synthetic_datasets)
        self._check_synthetic_per_batch(synthetic_per_batch)
        self.synthetic_per_batch = synthetic_per_batch
        self.batch_size = batch_size
        self.sample_weights = sample_weights
        self.set_rng(seed)

    def __iter__(self):
        """Return an iterator over the dataset

        This returns sequences of length "batch_size" whose elements are drawn from a
        single session (i.e. the batches are homogenous with respect to session).  For
        example if batch_size is 32 then the first 32 items will be drawn from the
        same session, then next 32 from a possibly different session, etc.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.set_rng(worker_info.seed % 2**32)

        if self.synthetic_per_batch > 0:
            assert (
                self.synthetic_datasets is not None
            ), "synthetic_per_batch > 0 but no synthetic datasets provided"
            synthetic_iterators = [iter(d) for d in self.synthetic_datasets]

        real_per_batch = self.batch_size - self.synthetic_per_batch

        self.reset()
        self.shuffle()
        while True:
            # choose the next dataset to sample from
            p = self.sample_weights
            dataset_index = self.rng.choice(
                [i for i in range(len(self.sentence_datasets))], p=p
            )

            # yield `real_per_batch` real samples
            for _ in range(real_per_batch):
                index = self._seen[dataset_index]
                # if we've seen all the real samples in the dataset then
                # reset and shuffle
                if index >= len(self.sentence_datasets[dataset_index]):
                    self.reset(dataset_index)
                    self.shuffle(dataset_index)
                    i = self._indices[dataset_index][0]
                else:
                    i = self._indices[dataset_index][index]
                self._seen[dataset_index] += 1
                item = self.sentence_datasets[dataset_index][i]
                yield item

            # yield `synth_per_batch` synthetic samples
            for i in range(self.synthetic_per_batch):
                item = next(synthetic_iterators[dataset_index])

                yield item

    def shuffle(self, dataset_index=None):
        if dataset_index is None:
            for i in range(len(self.sentence_datasets)):
                self._shuffle(i)
        else:
            self._shuffle(dataset_index)

    def _shuffle(self, dataset_index):
        self._indices[dataset_index] = self.rng.permutation(
            len(self.sentence_datasets[dataset_index])
        )

    def reset(self, dataset_index=None):
        num_ds = len(self.sentence_datasets)
        if dataset_index is None:
            self._seen = [0 for _ in range(num_ds)]
            self._indices = [range(len(ds)) for ds in self.sentence_datasets]
        else:
            self._seen[dataset_index] = 0
            self._indices[dataset_index] = range(
                len(self.sentence_datasets[dataset_index])
            )

    def set_rng(self, seed: Optional[int] = None):
        super().set_rng(seed)
        for x in self.synthetic_datasets:
            x.set_rng(seed)
        self.synthetic_iters = [iter(d) for d in self.synthetic_datasets]
        for y in self.sentence_datasets:
            y.set_rng(seed)

    def _check_synthetic_per_batch(self, synthetic_per_batch: int):
        """Check that synthetic_per_batch is valid"""
        try:
            synthetic_per_batch = int(synthetic_per_batch)
        except ValueError:
            raise ValueError("synthetic_per_batch must be an integer")
        if synthetic_per_batch < 0:
            raise ValueError("synthetic_per_batch must be >= 0")
        if synthetic_per_batch > 0 and self.synthetic_datasets is None:
            raise ValueError(
                "synthetic_per_batch > 0 but no synthetic datasets provided"
            )


class SentenceDataset(Dataset, NumpyRNGMixin):
    def __init__(
        self,
        data: HandwritingSessionData,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        trim_end_of_sentence: bool = True,
    ):
        """Initialize a SentenceDataset object.

        Args:
            data (Processed): A Processed data object. See
                `imagined_handwriting.data.preprocess`
            split (str): One of ['train', 'val', 'test']. Defaults to 'train'.
            transform (Optional[Callable]): A function to apply each example.
                If provided it should have the signature
                `transform(x:dict, Optional[np.random.Generator]=None)->dict`
        """
        self.data = data.dict()
        self._len = len(data.x)
        self.transform = transform
        self.session_id = data.session_id
        self.trim_end_of_sentence = trim_end_of_sentence
        self.set_rng(seed)

    def __len__(self):
        return self._len

    def __getitem__(self, i: int) -> dict:
        if self.trim_end_of_sentence:
            eos = self.data["end_of_sentence_index"][i]
        else:
            # use all of the data so batches are the same size during validation
            eos = None
        item = {
            "x": self.data["x"][i, :eos],
            "y_start": self.data["y_start"][i, :eos],
            "y_char": self.data["y_char"][i, :eos],
            "loss_mask": self.data["loss_mask"][i, :eos],
            "end_of_sentence_index": self.data["end_of_sentence_index"][i],
            "text": self.data["text"][i],
            "session_id": self.session_id,
            "trial_index": self.data["trial_index"][i],
        }

        if self.transform is not None:
            item = self.transform(item, self.rng)
        return item


class SyntheticDataset(IterableDataset, NumpyRNGMixin):
    def __init__(
        self,
        data: HandwritingSessionData,
        split: str = "train",
        *,
        corpus: List[str],
        one_hot: bool,
        transform: Optional[Callable] = None,
        snippet_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.data = data
        self.split = split
        self.corpus = corpus
        self.one_hot = one_hot

        self.transform = transform
        self.snippet_transform = snippet_transform
        self.snippets = extract_snippets(data)
        self.characters = list(self.snippets.character_index.keys())

        self.text_sampler = HandwritingTextSampler(
            valid_chars=set(self.characters),
            corpus=self.corpus,
            num_words=5,
            seed=seed,
        )

        sample_rate = SAMPLE_RATE / self.data.bin_width
        self.snippet_sampler = SnippetSampler(
            self.snippets, self.snippet_transform, seed=seed, sample_rate=sample_rate
        )
        self.set_rng(seed)

    def set_rng(self, seed: Optional[int] = None) -> None:
        super().set_rng(seed)
        self.text_sampler.set_rng(seed)
        self.snippet_sampler.set_rng(seed)

    def __iter__(self):
        """Return an iterator over the dataset."""
        while True:
            item = self.generate_example()
            if self.transform is not None:
                item = self.transform(item, self.rng)
            yield item

    def generate_example(self):
        """Generates a single training example."""
        sentence = self.text_sampler()
        snippets, n_pause = self.snippet_sampler(sentence)
        # config = self.config
        example = make_example(
            sentence,
            snippets,
            self.characters,
            one_hot=self.one_hot,
            n_pause=n_pause,
            bin_width=self.data.bin_width,
        )
        example["session_id"] = self.data.session_id
        return example


def make_example(
    sentence: str,
    snippets: List[ndarray],
    characters: List[str],
    *,
    one_hot=False,
    bin_width=1,
    n_pause=0,
):
    """Makes a training example.

    Args:
        sentence: A string of characters.
        snippets: A list of arrays of snippets, one for each character in the sentence.
        characters: A list of characters that are in the full dataset. The index
            of each character in this list determines its label.
        one_hot: Whether to use one-hot encoding for the characters.
        bin_width: The width of the bins used during preprocessing.  This is used
        to determine the length of the start signal.
        n_pause: The number of pauses in the sentence, used as a sanity check
            the the number of sampled snippets - pauses is equal to the number of
            characters in the sentence

    Returns:
        A single training example.
    """
    if len(sentence) != len(snippets) - n_pause:
        raise ValueError("Sentence and snippets must be the same length.")

    x = np.concatenate(snippets, axis=0)
    y_start = make_y_start(snippets, bin_width)
    y_char = make_y_char(sentence, snippets, characters, one_hot)
    loss_mask = make_loss_mask(x)
    end_of_sentence_index = make_end_of_sentence_index(x)

    item = {
        "x": x,
        "y_start": y_start,
        "y_char": y_char,
        "loss_mask": loss_mask,
        "end_of_sentence_index": end_of_sentence_index,
        "text": sentence,
        "trial_index": np.nan,
    }

    return item


def make_y_start(snippets: List[ndarray], bin_width=1) -> ndarray:
    """Makes start labels for the sentence.


    Args:
        sentence: A string of characters.
        snippets: A list of arrays of snippets, one for each character in the sentence.
        bin_width: The width of the bins used during preprocessing. This will effect
            the length of the start signal.

    Returns:
        np.ndarray of shape (len(sentence),)
    """
    timebins = sum(s.shape[0] for s in snippets)
    y_start = np.zeros(timebins, dtype=np.float32)

    SIGNAL_LEN = max(1, 20 // bin_width)
    start_index = 0
    for i in range(len(snippets)):
        end_index = start_index + SIGNAL_LEN
        y_start[start_index:end_index] = 1
        start_index += snippets[i].shape[0]

    return y_start


def make_y_char(
    sentence: str, snippets: List[ndarray], characters: List[str], one_hot: bool
) -> ndarray:
    """Makes character labels for the sentence.

    Args:
        sentence: A string of characters.
        snippets: A list of arrays of snippets, one for each character in the sentence.
        characters: A list of characters that are in the full dataset. The index of
            each character in this list determines its label.
        one_hot: Whether to use one-hot encoding for the characters.

    Returns:
        An array of character labels of shape (len(sentence),) when one_hot is False and
        shape (len(sentence), len(characters)) when one_hot is True.  The data type of
        the return is based on one-hot vs sparse encoding to comply with pytorch
        expectations of these labels.
    """
    timebins = sum(s.shape[0] for s in snippets)
    y_char = np.zeros(timebins, dtype=np.int64)

    start_index = 0
    for i in range(len(sentence)):
        end_index = start_index + snippets[i].shape[0]
        character_index = characters.index(sentence[i])
        y_char[start_index:end_index] = character_index
        start_index += snippets[i].shape[0]

    if one_hot:
        y_char = T.one_hot(y_char, num_classes=len(characters)).astype(np.float32)

    return y_char


def make_loss_mask(x: ndarray) -> ndarray:
    """Makes a loss mask for the given input.

    Args:
        x: A numpy array of shape (timebins, ...)

    Returns:
        An array of ones of shape (timebins,)
    """
    return np.ones(x.shape[0], dtype=np.float32)


def make_end_of_sentence_index(x: ndarray) -> int:
    """Makes an index of the end of the sentence.

    Args:
        x: A numpy array of shape (timebins, ...)
    Returns:
        An int that represents the index of the end of the sentence.
    """
    return x.shape[0]


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def ensure_list_or_none(x):
    if x is None:
        return None
    else:
        return ensure_list(x)
