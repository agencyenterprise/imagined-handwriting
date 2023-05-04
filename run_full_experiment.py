"""
Runs a full experiment.

The full experiment simulates the real-time handwriting training and testing setup
used in the original paper. We pretrain with the first 3 sessions and then we sequentially
fine-tune the model on each of the remaining sessions. At each fine-tuning step
we split the data so that the test set is composed of the "hold out blocks"
i.e. the real close-loop test data.

This script can be adapted (e.g. replacing the model) to run other experiments which
follow the same setup, thereby allowing a comparison of different methods on the same
data and in particular a comparison of how the model might generalize to a real-time
closed loop setup.  Note that the fine-tuning time should be limited since the fine
tuning must happen in-session.  The authors of the original paper report ~4minutes
for fine-tuning.
"""
from argparse import ArgumentParser
import subprocess
from pathlib import Path
from datetime import datetime
import inspect
import yaml
import os

import mlflow
from pytorch_lightning import Trainer

from imagined_handwriting.settings import SESSIONS


def parse_cli():
    """Parse the command line arguments."""
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--name', type=str, default='full_experiment')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--execution_id', type=str, default=None)
    parser.add_argument('--pretrain_steps', type=int, default=200000)
    parser.add_argument('--finetune_steps', type=int, default=1000)
    config = vars(parser.parse_args())

    name = config.pop('name')
    num_workers = config.pop('num_workers')
    execution_id = config.pop('execution_id')
    pretrain_steps = config.pop('pretrain_steps')
    finetune_steps = config.pop('finetune_steps')
    trainer_defaults = {k:v.default for k,v in inspect.signature(Trainer.__init__).parameters.items()}
    trainer_overrides = {}
    for k,v in config.items():
        if k not in trainer_defaults:
            raise ValueError(f"Invalid config key: {k}")
        if v!=trainer_defaults[k]:
            trainer_overrides[k] = v

    if 'max_steps' in trainer_overrides:
        raise ValueError("Do not set max_steps.  Use --pretrain-steps and --finetune-steps instead.")

    final_config = {
        "name":name,
        "execution_id":execution_id,
        **trainer_overrides
    }
    if num_workers is not None:
        final_config['num_workers'] = num_workers
    return final_config, pretrain_steps, finetune_steps


def main():
    """Runs a full experiment.

    Executes a full experiment including pretraining and sequential fine-tuning.
    This is essentially a wrapper/coordination script which calls the pretrain.py
    and finetune.py scripts.

    Logs written to mlruns via MLFlow and Tensorboard

    See pretrain.py and finetune.py.
    """
    execution_id = datetime.now().strftime('%Y%m%d%H%M%S')
    config, pretrain_steps, finetune_steps = parse_cli()
    prior_execution_id = config.pop('execution_id')

    # determine if we are relaunching an interrupted experiment
    # if we are, use the given execution id and skip any of the
    # training we have already accomplished.
    PRETRAIN = True
    FINETUNE_SESSIONS = SESSIONS.copy_typing
    if prior_execution_id is not None:
        # we are resuming an experiment, skip all the runs that have already been done.
        runs = get_runs_by_execution_id(prior_execution_id)
        max_id = ""
        for run in runs:
            session_ids = yaml.safe_load(run.data.params['session_ids'])
            max_id = max(max(session_ids), max_id)

        max_session = SESSIONS.all.index(max_id)
        FINETUNE_SESSIONS = SESSIONS.all[max_session+1:]
        PRETRAIN = False
        execution_id = prior_execution_id



    # pretrain
    if PRETRAIN:
        pretrain_config = config.copy()
        pretrain_config['max_steps'] = pretrain_steps
        cmd = make_pretrain_command(pretrain_config, execution_id)
        print("Executing pretrain command:")
        print(" ".join(cmd))
        exit = subprocess.run(cmd, errors=True)
        if exit.returncode !=0:
            print(exit.stderr)
            raise ValueError(f'Finetuning failed with exit code {exit.returncode}')

    # iterate through fine tune sessions
    for session in FINETUNE_SESSIONS:
        finetune_config = config.copy()
        finetune_config['max_steps'] = finetune_steps
        cmd = make_fine_tune_command(finetune_config, execution_id, session)
        print("Executing fine-tune command:")
        print(" ".join(cmd))
        exit = subprocess.run(cmd, errors=True)
        if exit.returncode !=0:
            print(exit.stderr)
            raise ValueError(f'Finetuning failed with exit code {exit.returncode}')


def make_pretrain_command(config, execution_id):
    """Make the pretrain command."""
    config = config.copy()
    config['session_ids'] = SESSIONS.pretrain
    return make_cmd('pretrain.py', config, execution_id)

def make_fine_tune_command(config, execution_id, finetune_session):
    """Make the fine-tune command."""
    config = config.copy()
    ckpt = get_checkpoint(execution_id, finetune_session)
    config['val_check_interval'] = 100
    config['log_figure_every_n_steps'] = 100
    config['checkpoint_every_n_steps'] = 1000
    config['session_id'] = finetune_session
    config['from'] = ckpt
    return make_cmd('finetune.py', config, execution_id)

def make_cmd(script, config, execution_id):
    """Make the command."""
    cmd = ['python', script]
    cmd += [f"--id", execution_id]
    for k,v in config.items():
        if k == 'session_ids':
            cmd += [f"--{k}"] + v
        else:
            cmd += [f"--{k}", str(v)]
    return cmd


def get_checkpoint(execution_id, finetune_session):
    """Get the correct checkpoint based on the execution id and fine tune session.

    The checkpoint is the checkpoint from the previous training run. The previous
    is either a pretrain run or a fine-tuning run.  We know which run checkpoint
    we want based on the execution id and the sessions that were used to train the
    model.  If max(sessions) = fine_tune_session -1 then this was the previous run.
    """
    runs = get_runs_by_execution_id(execution_id)
    for run in runs:
        session_ids = yaml.safe_load(run.data.params['session_ids'])
        last_session = max(session_ids)
        if SESSIONS.all.index(last_session) == SESSIONS.all.index(finetune_session)-1:
            checkpoint_dir = Path(run.info.artifact_uri + '/checkpoints').relative_to('file:./')
            if checkpoint_dir / 'last.ckpt' in list(checkpoint_dir.iterdir()):
                return str(checkpoint_dir / 'last.ckpt')
            else:
                raise ValueError(f"Could not find checkpoint in {checkpoint_dir}.  Make sure "
                    "the trainer is configured to save the last checkpoint as last.ckpt")


def get_runs_by_execution_id(execution_id):
    """Get the runs associated with a given execution id.

    Args:
        execution_id (str): The execution id to search for.

    Returns:
        list: A list of runs associated with the execution id.
    """
    client = mlflow.MlflowClient()
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    return client.search_runs(experiment_ids=all_experiments, filter_string=f"tags.execution_id = '{execution_id}'")




if __name__ == "__main__":
    main()