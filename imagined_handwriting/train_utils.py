from typing import List, Tuple

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


def parse_with_groups(parser):

    args = parser.parse_args()
    transformer_config = {}
    optimization_config = {}
    data_config = {}
    logging_config = {}
    inference_config = {}
    trainer_config = {}
    options = {}

    for group in parser._action_groups:
        if group.title == "HandwritingTransformerConfig":
            transformer_config = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
        elif group.title == "HandwritingOptimizationConfig":
            optimization_config = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
        elif group.title == "HandwritingDataConfig":
            data_config = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
        elif group.title == "LoggingConfig":
            logging_config = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
        elif group.title == "InferenceConfig":
            inference_config = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
        elif group.title == "pl.Trainer":
            trainer_config = {
                a.dest: getattr(args, a.dest, None) for a in group._group_actions
            }
        elif group.title in ["optional arguments", "options"]:
            options = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
                if a.dest != "help"
            }

    return {
        "transformer": transformer_config,
        "optimization": optimization_config,
        "data": data_config,
        "logging": logging_config,
        "inference": inference_config,
        "trainer": trainer_config,
        "options": options,
    }


def configure_trainer(
    trainer_config: dict, logging_config: dict, options: dict
) -> Trainer:
    """Configure the trainer."""
    use_logger = trainer_config.pop("logger")
    if use_logger:
        loggers = configure_loggers(options["name"])
        artifact_uri = get_mlflow_artifact_uri(loggers[0])
        dirpath = artifact_uri + "/checkpoints"
        callbacks = configure_callbacks(
            dirpath, logging_config["checkpoint_every_n_steps"]
        )
    else:
        raise ValueError("Logger is required for pretraining")
    return Trainer(**trainer_config, logger=loggers, callbacks=callbacks)


def configure_callbacks(dirpath: str, every_n_train_steps: int) -> List[Callback]:
    """Configure the callbacks."""
    ckpt = ModelCheckpoint(
        dirpath=dirpath,
        save_top_k=-1,
        every_n_train_steps=every_n_train_steps,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [ckpt, lr_monitor]


def configure_loggers(name) -> Tuple[MLFlowLogger, TensorBoardLogger]:
    """Create an MLFlow logger and a Tensorboard logger.

    The tensorboard logger will log to the artifact location of the
    MLFlow logger.  We use both loggers because we find that the
    tensorboard logger is better for visualizing the training process
    whereas the MLFlow logger is better for tracking the results.
    """
    mlflow_logger = MLFlowLogger(experiment_name=name)
    artifact_uri = get_mlflow_artifact_uri(mlflow_logger)
    tb_logger = TensorBoardLogger(save_dir=artifact_uri, name="", version="")
    return mlflow_logger, tb_logger


def get_mlflow_artifact_uri(mlflow: MLFlowLogger) -> str:
    """Gets the artifact uri from the mlflow logger.

    Used to set the Tensorboard logger directory to inside MLFlows artifact dir.
    """
    run_id = mlflow.run_id
    uri = mlflow.experiment.get_run(run_id).info.artifact_uri
    if uri.startswith("file:"):
        uri = uri[5:]
    return uri
