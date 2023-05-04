from argparse import ArgumentParser
from typing import List

import torch 
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger 

from imagined_handwriting.datasets import FineTuneDataModule
from imagined_handwriting.models import HandwritingTransformerExperiment
from imagined_handwriting.config import HandwritingOptimizationConfig, HandwritingDataConfig, LoggingConfig, InferenceConfig
from imagined_handwriting.train_utils import parse_with_groups, configure_trainer


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--name', type=str, default='fine_tune')
    parser.add_argument('--from', type=str)
    parser.add_argument('--session_id', type=str)
    parser.add_argument('--id', type=str)
    parser = HandwritingDataConfig.add_model_specific_args(parser)
    parser = HandwritingOptimizationConfig.add_model_specific_args(parser)
    parser = LoggingConfig.add_model_specific_args(parser)
    parser = InferenceConfig.add_model_specific_args(parser)
    
    configs = parse_with_groups(parser)
    trainer_config = configs['trainer']
    optimization_config = configs['optimization']
    logging_config = configs['logging']
    inference_config = configs['inference']
    data_config = configs['data']
    options = configs['options']
    options['fine_tune_session_id'] = options['session_id']

    # set default max steps for fine-tuning if none was provided
    if trainer_config['max_steps'] is None or trainer_config['max_steps'] == -1:
        trainer_config['max_steps'] = 1000

    # make full configs
    model_config = {**optimization_config, **logging_config, **inference_config, **options, **trainer_config}
    data_config = {**data_config, **options}

    # set seed
    torch.manual_seed(data_config['seed'])
    np.random.seed(data_config['seed'])


    ###########
    # Trainer 
    ###########
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
   
    
    ############
    # Model 
    #  we need to load the pretrained model and add a new session to the 
    #  the session specific embedding layer, and initialize it with the 
    #  embedding of the last session in the pretrained model
    ############
    print('='*80)
    pretrain_session_ids = get_pretrain_session_ids(options['from'])
    new_session_id = options['session_id']
    session_ids = pretrain_session_ids + [new_session_id]
    model = HandwritingTransformerExperiment.load_from_checkpoint(
        options['from'], 
        strict=False, 
        session_ids=session_ids, 
        **model_config
        )
    model.model.initialize_embedding(new_session_id, pretrain_session_ids[-1])
    assert session_ids == model.model.session_ids
    print('Model configured')
    print(f'Fine tuning on session: {new_session_id}')
    print(f'Fine tuning from {options["from"]}')
    print('='*80)


    ############
    # Data
    ############
    print('Configuring Data')
    dm = FineTuneDataModule(session_ids=session_ids, **data_config)
    dm.prepare_data()
    dm.setup('fit')
    print('Data configured')
    n_val = len(dm.val_dataloader())
    print(f'Number of validation sessions: {n_val}')
    for val_dl in dm.val_dataloader():
        print(f'{val_dl.dataset.session_id} examples: {len(val_dl.dataset)}')
    print('='*80)


    ##########
    # Train 
    ##########
    trainer.fit(model, datamodule=dm)


def get_pretrain_session_ids(ckpt):
    return torch.load(ckpt)['hyper_parameters']['session_ids']



if __name__ == "__main__":
    main()



