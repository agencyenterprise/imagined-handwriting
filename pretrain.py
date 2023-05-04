from argparse import ArgumentParser

import torch 
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from imagined_handwriting.models import HandwritingTransformerExperiment
from imagined_handwriting.datasets import PretrainDataModule
from imagined_handwriting.config import HandwritingOptimizationConfig, HandwritingDataConfig, HandwritingTransformerConfig, LoggingConfig, InferenceConfig
from imagined_handwriting.train_utils import parse_with_groups, configure_trainer



def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--name", type=str, default="pretrain")
    parser.add_argument("--session_ids", type=str, nargs="+", required=True)
    parser.add_argument("--id", type=str)
    parser = HandwritingDataConfig.add_model_specific_args(parser)
    parser = HandwritingOptimizationConfig.add_model_specific_args(parser)
    parser = HandwritingTransformerConfig.add_model_specific_args(parser)
    parser = LoggingConfig.add_model_specific_args(parser)
    parser = InferenceConfig.add_model_specific_args(parser)

    configs = parse_with_groups(parser)
    trainer_config = configs['trainer']
    optimization_config = configs['optimization']
    transformer_config = configs['transformer']
    logging_config = configs['logging']
    inference_config = configs['inference']
    data_config = configs['data']
    options = configs['options']

    # override trainer config for max_steps
    # we have an infinite dataloader so we need to set max_steps
    if trainer_config['max_steps'] is None or trainer_config['max_steps'] == -1:
        if 'lr_scheduler_config' in optimization_config:
            lr_scheduler_config = optimization_config['lr_scheduler_config']
            if 'total_iters' in lr_scheduler_config:
                trainer_config['max_steps'] = lr_scheduler_config['total_iters']
            elif 'total_steps' in lr_scheduler_config:
                trainer_config['max_steps'] = lr_scheduler_config['total_steps']
        else:
            trainer_config['max_steps'] = 100000


    if trainer_config['val_check_interval'] is None:
        trainer_config['val_check_interval'] = 1000
    


    # make full configs
    # we need to save the trainer config since we can set things like gradient clipping via the trainer
    model_config = {**transformer_config, **optimization_config, **logging_config, **inference_config, **options, **trainer_config}
    data_config = {**data_config, **options}

    # set seed
    torch.manual_seed(data_config['seed'])
    np.random.seed(data_config['seed'])


    ##########
    # Trainer 
    ##########
    trainer = configure_trainer(trainer_config, logging_config, options)

    #########################
    # Write execution metdata
    #   to be able to tie pretraining and finetuning together we 
    #   write the id of the pretraining run as a mlflow tag which 
    #   can be used later to tie runs together.
    #########################
    if options['id'] is not None:
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.set_tag(logger.run_id, "execution_id", options['id'])

    ##########
    # Model 
    ##########
    print('='*80)
    model = HandwritingTransformerExperiment(**model_config)
    print('Model configured')
    print(f'Training on sessions: {model.model.session_ids}')
    print('='*80)

    ############
    # Data 
    ############
    print('Configuring Data')
    dm = PretrainDataModule(**data_config)
    dm.prepare_data()
    dm.setup('fit')
    print('Data configured')
    print('='*80)

    ############
    # Train
    ############
    trainer.fit(model, datamodule=dm)





if __name__ == "__main__":
    main()



