import warnings

from optuna.integration import PyTorchLightningPruningCallback

warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from model import LightningModel
from data import LightningData

from math import sqrt

import psutil
import torch
import optuna
import wandb

STUDY_NAME = 'crop-forecasting'

def main():
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(study_name=STUDY_NAME,
                                storage=f'sqlite:///{STUDY_NAME}.db',
                                load_if_exists=True,
                                direction='maximize',
                                pruner=pruner)

    study.optimize(objective, n_trials=100)


def init_optuna(trial, batch_size=16):
    data, first_batch = get_data(batch_size=batch_size, val_rate=0.2)

    s_hidden_size = trial.suggest_int('s_hidden_size', 64, 512)
    s_num_layers = trial.suggest_int('s_num_layers', 1, 3)
    m_hidden_size = trial.suggest_int('m_hidden_size', 64, 512)
    m_num_layers = trial.suggest_int('m_num_layers', 1, 4)
    learning_rate = trial.suggest_uniform('learning_rate', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout ', 0.5, 0.8)
    optimizer = trial.suggest_categorical('optimizer ', choices=['AdamW', 'RMSprop'])
    c_in_features = s_hidden_size - 2 + m_hidden_size - 2 + first_batch['g_input'].shape[0]
    c_out_in_features_1 = trial.suggest_int('c_out_in_features_1', int(sqrt(c_in_features)), 2 * c_in_features)
    c_out_in_features_2 = trial.suggest_int('c_out_in_features_2', int(sqrt(c_in_features)), 2 * c_in_features)

    config = {
        'batch_size': batch_size,
        's_hidden_size': s_hidden_size,
        's_num_layers': s_num_layers,
        'm_hidden_size': m_hidden_size,
        'm_num_layers': m_num_layers,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'epochs': 20,
        'optimizer': optimizer,
        'criterion': 'MSELoss',
        's_num_features': first_batch['s_input'].shape[1],
        'm_num_features': first_batch['m_input'].shape[1],
        'g_in_features': first_batch['g_input'].shape[0],
        'c_in_features': c_in_features,
        'c_out_in_features_1': c_out_in_features_1,
        'c_out_in_features_2': c_out_in_features_2,
        'trial.number': trial.number
    }

    wandb.init(
        project='winged-bull',
        config=config,
        group=STUDY_NAME,
        reinit=True
    )

    return trial, config, data


def objective(trial):
    trial, config, data = init_optuna(trial)
    model = LightningModel(config, trial)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        max_epochs=config['epochs'],
        accelerator='auto',
    )

    trainer.fit(model, data)

    return trainer.callback_metrics['best_score'].item()


def get_data(batch_size, val_rate):
    data = LightningData(batch_size, val_rate, num_workers=4)
    data.setup(stage='fit')
    first_batch = data.train_dataset[0]

    return data, first_batch


if __name__ == '__main__':
    main()
