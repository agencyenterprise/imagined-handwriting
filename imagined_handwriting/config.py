import json
from typing import Callable, Optional, Union, get_args

from pydantic import BaseModel


class CropSize(BaseModel):
    crop_size = 200


class HandwritingBaseConfig(BaseModel):
    """Base config for all models."""

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        for arg in cls.__fields__.values():
            if not (
                any(x is Callable for x in get_args(arg.type_)) or arg.type_ is Callable
            ):
                if arg.type_ is dict:
                    parser.add_argument(
                        f"--{arg.name}", type=json.loads, default=arg.default
                    )
                else:
                    parser.add_argument(
                        f"--{arg.name}", type=arg.type_, default=arg.default
                    )

        return parent_parser


class HandwritingDataConfig(HandwritingBaseConfig):
    """Handwriting data configuration.

    Args:
        include_test (bool): Whether to include the test data in the training
            data. This is useful when we are fine-tuning on future sessions
            or pre-training and want to use all the data available. Default
            is False.
        n_train (Optional[int]): The number of training examples to use. If
            None all the training data will be used, otherwise a random
            split of train/val will be made with n_train examples in the
            training set (and the remainder in the validation set). Default
            is None.
        fine_tune (bool): Whether we are fine-tuning.  If True, then
            we don't include the test set in the training data for the most
            recent session. See `lightning.data_module.Handwriting`
            Default is False.
        fine_tune_weight (float): The weight to give to the most recent
            session when fine-tuning. Batches from the most recent session
            (the fine-tuning session) will be sampled with probability
            equal to this weight.  The other datasets will be sampled
            uniformly with probability (1 - fine_tune_weight).
            Default is 0.5.
        transform (Callable | str | None): The transform to apply to each
            data example before it is returned.  If a string, it must be
            "default".  If None, no transformation will be applied. Default
            is "default".
        snippet_transform (Callable | str | None): The transform to apply
            to each snippet before it is concatenated into a sentence. If
            a string, it must be "default".  If None, no transformation will
            be applied. Default is "default".
        batch_size (int): The batch size to use. Default is 64.
        synthetic_per_batch (int): The number of synthetic examples to
            include in each batch.  Default is 24.  To disable synthetic
            data set this to 0.  To use only synthetic data set it to
            batch_size.  Note if this is set to batch_size, then
            `infinite` must be True since the synthetic data is always
            an infinite iterable.
        shuffle (bool): Whether to shuffle the sentence data on
            each iteration through all the sentences.  Default is True.
        bin_width (int): The factor for rebinning the data. The raw data
            is threshold spike counts binned to 10ms.  The `bin_width`
            acts as a multiplier to increase the bin width. For example,
            if bin_width=2, then the data will be binned to 20ms.
            Default is 2.
        normalize (bool): Whether to normalize the data.  Default is True.
            This uses "block normalization" in which the data is normalized
            by the closest "character block" (i.e. the closest in time
            single character collection blocks to the block to be normalized).
            This is the normalization scheme used in the original paper.
            Default is True.
        one_hot (bool): Whether to one-hot encode the character targets.
            Default is False.
        embed_channels (bool): Whether to change firing rates to a unique
            embedding index for each channel.  That is firing rate r
            for channel index c will be converted to c*max_firing_rate + r.
            This can be used to embed the spike counts directly into a
            vector space while maximally retaining both firing rate and
            channel information.  Currently experimental.  Default is False.
        max_firing_rate (int): The maximum firing rate to use.  This is
            ignored if normalize is True, otherwise the maximum firing
            rate is clipped to this value. Default is 5.  If None the
            maximum firing rate is not clipped.
        max_pause (int): The maximum number of time steps to wait after
            the start of the last character in a sentence before declaring
            the sentence over.  This is applied before binning.
            This exists because some sentences the subject paused for
            a while before manually triggering (with a
            head movement) the end of the sentence. See the original
            paper for details. Default is 150.
        crop_size (int): The size of the crop to use for training data.
            A random crop of each sentence will be taken of this size
            to create a single example.  This determines the sequence
            length the model sees during training. Default is 200.
        pad_width (int): The size of padding applied to each example
            before cropping.  This allows the model to "start hot" in
            the middle of a sentence. Default is 100.
        white_noise_std (float): The standard deviation of the white noise
            data augmentation.  Default is 1.6.
        offset_noise_std (float): The standard deviation of the offset data
            augmentation.  Default is 0.6.
        random_walk_noise_std (float): The standard deviation of the random
            walk data augmentation.  Note that since our default crop size
            is much lower than the paper, the random walk has a smaller
            influence on average.  It may be prudent to raise this from
            the default used in the paper (same default here). Default is 0.02.
        timewarp_low (float): The lower bound of the percentage randomly warp
            snippets by.  Default is 0.7.
        timewarp_high (float): The upper bound of the percentage randomly warp
            snippets by.  Default is 1.3.
        scale_low (float): The lower bound of the percentage randomly scale
            snippets by.  Default is 0.7.
        scale_high (float): The upper bound of the percentage randomly scale
            snippets by.  Default is 1.3.
        num_workers (int): The number of workers to use for data loading.
            Default is 0.
        pin_memory (bool): Whether to pin memory for data loading. Default
            is False.
    """

    # train: bool = True
    n_train: Optional[int] = None
    fine_tune_weight: float = 0.25
    transform: Optional[Union[str, Callable]] = "default"
    snippet_transform: Optional[Union[str, Callable]] = "default"
    batch_size: int = 64
    synthetic_per_batch: int = 24
    shuffle: bool = True
    bin_width: int = 2
    normalize: bool = True
    one_hot: bool = False
    embed_channels: bool = False
    max_firing_rate: Optional[int] = 5
    max_pause: int = 150
    mask_blank_windows: bool = True
    seed: int = 0
    holdout: str = "blocks"
    # transform configs
    crop_size: int = CropSize().crop_size
    pad_width: int = 100
    white_noise_std: float = 1.2
    offset_noise_std: float = 0.6
    random_walk_noise_std: float = 0.02
    smooth: bool = True
    snippet_timewarp_low: float = 0.7
    snippet_timewarp_high: float = 1.3
    snippet_scale_low: float = 0.7
    snippet_scale_high: float = 1.3
    # environment
    num_workers: int = 0
    pin_memory: bool = False


class HandwritingTransformerConfig(HandwritingBaseConfig):
    input_channels: int = 192
    num_classes: int = 32  # num characters + 1 for start logits
    attention: str = "fft"
    d_model: int = 1024
    dim_feedforward: int = 4096
    num_layers: int = 2
    nhead: int = 1
    encoder_dropout: float = 0.1
    seq_dropout: float = 0.1
    channel_dropout: float = 0.1
    classifier_dropout: float = 0.1
    activation: str = "gelu"
    positional_embedding: Optional[str] = "sinusoid"
    linear_embedding: bool = True
    resample_factor: int = 5
    resample_method: str = "mean"
    layer_norm_eps: float = 1e-5
    norm_first: bool = True
    batch_first: bool = True
    orthogonal_init: bool = True


class HandwritingOptimizationConfig(HandwritingBaseConfig):
    lr: float = 5e-4
    optimizer: str = "AdamW"
    optimizer_config: dict = {"weight_decay": 1e-5}
    lr_scheduler: str = "LinearLR"
    lr_scheduler_config: dict = {
        "start_factor": 1.0,
        "end_factor": 1e-2,
        "total_iters": 100000,
    }
    mse_weight: float = 5.0
    freeze_encoder: bool = False


class LoggingConfig(HandwritingBaseConfig):
    log_figure_every_n_steps: int = 10000
    checkpoint_every_n_steps: int = 10000


class InferenceConfig(HandwritingBaseConfig):
    sliding_window_config: dict = {
        "window_size": CropSize().crop_size,
        "stride": 100,
        "prediction_start": 50,
    }


class HandwritingConfig(
    HandwritingDataConfig,
    HandwritingTransformerConfig,
    HandwritingOptimizationConfig,
    LoggingConfig,
    InferenceConfig,
):
    pass
