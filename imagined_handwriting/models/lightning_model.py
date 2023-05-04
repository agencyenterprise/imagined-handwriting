from typing import Optional

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

from imagined_handwriting import metrics
from imagined_handwriting.inference import SlidingWindowDecoder
from imagined_handwriting.transforms import one_hot

from .transformer import HandwritingTransformer


class HandwritingTransformerExperiment(LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = self.build_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def build_model(self):
        """Returns a handwriting model based on the config"""

        model = HandwritingTransformer(**self.config)
        if self.config["freeze_encoder"]:
            print("Freezing Encoder")
            model.freeze_encoder()
        return model

    def configure_optimizers(self):
        opt = self.get_opt()
        lr_scheduler_config = self.get_lr_scheduler(opt)
        if lr_scheduler_config is None:
            return opt
        return [opt], [lr_scheduler_config]

    def on_train_start(self):
        """Log the model architecture."""
        loggers = [
            logger for logger in self.loggers if isinstance(logger, MLFlowLogger)
        ]
        if len(loggers) > 0:
            mlflow_logger = loggers[0]
            mlflow_logger.experiment.log_text(
                mlflow_logger.run_id, str(self.model), "model.txt"
            )

    def get_opt(self):
        """Returns the optimizer"""
        opt_name = self.config["optimizer"]
        if not hasattr(torch.optim, opt_name):
            raise ValueError(
                f"{opt_name} is not a valid optimizer, must be one of "
                f"{list(torch.optim.__dict__.keys())}"
            )
        opt_cls = getattr(torch.optim, opt_name)
        opt = opt_cls(
            self.model.parameters(),
            lr=self.config["lr"],
            **self.config["optimizer_config"],
        )
        return opt

    def get_lr_scheduler(self, opt):
        """Returns the lr scheduler"""
        if self.config["lr_scheduler"] is None:
            return None
        lr_scheduler_name = self.config["lr_scheduler"]
        if not hasattr(torch.optim.lr_scheduler, lr_scheduler_name):
            raise ValueError(
                f"{lr_scheduler_name} is not a valid lr_scheduler, "
                f"must be one of {list(torch.optim.lr_scheduler.__dict__.keys())}"
            )
        lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_scheduler_name)
        lr_scheduler = lr_scheduler_cls(opt, **self.config["lr_scheduler_config"])
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return lr_scheduler_config

    def training_step(self, batch: dict, batch_index: int):
        return self.step(batch, batch_index, mode="train")

    def validation_step(
        self, batch: dict, batch_index: int, dataloader_index: Optional[int] = None
    ):
        return self.step(batch, batch_index, mode="val")

    def test_step(
        self, batch: dict, batch_index: int, dataloader_index: Optional[int] = None
    ):
        return self.step(batch, batch_index, mode="test")

    def step(self, batch: dict, batch_idx: int, mode: str):
        x = batch["x"]
        y_start = batch["y_start"]
        y_char = batch["y_char"]
        loss_mask = batch["loss_mask"]
        session_id = batch["session_id"]
        eos = batch["end_of_sentence_index"]
        if mode == "train":
            model = self.model
        else:
            model = SlidingWindowDecoder(
                self.model, **self.config["sliding_window_config"]
            )

        logits_start, logits_char = model(x, session_id=session_id)

        # use loss mask for mse during validation so time steps after the end of
        # don't count against mse
        mse = metrics.mse_loss(
            logits_start, y_start, loss_mask=(None if mode == "train" else loss_mask)
        )
        ce = metrics.cross_entropy_loss(logits_char, y_char, loss_mask)
        loss = self.config["mse_weight"] * mse + ce
        acc = metrics.frame_accuracy(logits_char, y_char, loss_mask)

        self.log(f"{mode}/loss", loss)
        self.log(f"{mode}/acc", acc)
        self.log(f"{mode}/mse", mse)
        self.log(f"{mode}/ce", ce)

        if self.global_step % self.config["log_figure_every_n_steps"] == 0:
            example = {
                "y_start": y_start[0],
                "y_char": y_char[0],
                "logits_start": logits_start[0],
                "logits_char": logits_char[0],
                "loss_mask": loss_mask[0],
            }
            if mode != "train":
                example = {k: v[: eos[0]] for k, v in example.items()}
            self.log_summary_figure(
                mode,
                **example,
            )

        return loss

    def log_summary_figure(
        self,
        mode,
        *,
        y_start,
        y_char,
        logits_start,
        logits_char,
        loss_mask,
    ):
        """Logs a summary figure of the labels and predictions"""
        fig, _ = self.summary_figure(
            y_start=y_start,
            y_char=y_char,
            logits_start=logits_start,
            logits_char=logits_char,
            loss_mask=loss_mask,
        )
        self.log_figure(fig, name=f"{mode}/summary")

    def log_figure(self, fig, name="summary", global_step=None):
        """Logs a figure to the supported loggers."""
        global_step = global_step or self.global_step
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure(name, fig, global_step=global_step)
            elif isinstance(logger, MLFlowLogger):
                run_id = logger.run_id
                logger.experiment.log_figure(run_id, fig, f"{name}_{global_step}.png")
            else:
                raise NotImplementedError(
                    f"Logging figure {logger} is not implemented."
                    "To log a figure with this type of logger, subclass "
                    "HandwritingExperiment and override the log_figure method."
                )

    def summary_figure(self, *, y_start, y_char, logits_start, logits_char, loss_mask):
        """Creates a summary figure to show labels and predictions."""
        global_step = self.global_step
        return summary_figure(
            y_start=y_start,
            y_char=y_char,
            logits_start=logits_start,
            logits_char=logits_char,
            loss_mask=loss_mask,
            global_step=global_step,
        )


def summary_figure(
    *, y_start, y_char, logits_start, logits_char, loss_mask, global_step=None
):
    """Creates a summary figure to show labels and predictions.

    Args:
        y_start: (timesteps,) array of character start labels which is 0 when the
            character has not started and 1 when it has started.
        y_char: (timesteps, n_chars) array of one-hot character labels.
        logits_start: (timesteps,) array of character start logits (needs sigmoid)
        logits_char: (timesteps, n_chars) array of character logits (needs softmax)
        loss_mask: (timesteps,) array of 0s and 1s which is 0 when the timestep
            should beignored in the loss and 1 when it should be included.
            This is used to mask padding steps etc.
        global_step: The global step to use in the title of the figure. Optional.
    """
    logits_start = _to_numpy(torch.sigmoid(logits_start))
    logits_char = _to_numpy(torch.softmax(logits_char, dim=-1))
    y_start = _to_numpy(y_start)
    y_char = _to_numpy(y_char)
    loss_mask = _to_numpy(loss_mask)

    timesteps = y_start.shape[0]
    n_chars = logits_char.shape[-1]
    y_char = _enforce_one_hot(y_char, n_chars)

    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    ax[0].plot(y_start, label="Character Start")
    ax[0].plot(logits_start, label="Predicted Start")
    ax[0].plot(loss_mask, color="black", label="Loss Mask")
    ax[0].legend()

    ax[1].imshow(y_char.T, clim=[0, 1])
    ax[2].imshow(logits_char.T, clim=[0, 1])
    for axis in ax:
        axis.set_aspect("auto")
        axis.set_xlim([0, timesteps])

    ax[1].set_title("True Characters")
    ax[2].set_title("Predicted Characters")
    if global_step is not None:
        fig.suptitle(f"Global Step {global_step}")
    fig.tight_layout()
    return fig, ax


def _enforce_one_hot(y_char, n_chars):
    if y_char.ndim == 1:
        return one_hot(y_char, num_classes=n_chars)
    else:
        return y_char


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()
