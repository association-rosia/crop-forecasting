import warnings

warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from model import LightningModel
from data import get_dataloaders

from math import sqrt

import torch
import optuna
import wandb

STUDY_NAME = 'crop-forecasting'
BATCH_SIZE = 16
VAL_RATE = 0.2


def main():
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(study_name=STUDY_NAME,
                                storage=f'sqlite:///{STUDY_NAME}.db',
                                load_if_exists=True,
                                direction='maximize',
                                pruner=pruner)

    study.optimize(objective, n_trials=100)


def init_optuna(trial):
    train_dataloader, val_dataloader, first_batch = get_data()
    s_hidden_size = trial.suggest_int('s_hidden_size', 64, 256)
    s_num_layers = trial.suggest_int('s_num_layers', 1, 2)
    m_hidden_size = trial.suggest_int('m_hidden_size', 64, 256)
    m_num_layers = trial.suggest_int('m_num_layers', 1, 2)
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-5, 5e-3)
    dropout = trial.suggest_uniform('dropout', 0.5, 0.8)
    optimizer = trial.suggest_categorical('optimizer', choices=['AdamW', 'RMSprop'])
    c_in_features = s_hidden_size - 2 + m_hidden_size - 2 + first_batch['g_input'].shape[0]
    c_out_in_features_1 = trial.suggest_int('c_out_in_features_1', int(sqrt(c_in_features)), 2 * c_in_features)
    c_out_in_features_2 = trial.suggest_int('c_out_in_features_2', int(sqrt(c_in_features)), 2 * c_in_features)

    config = {
        'batch_size': BATCH_SIZE,
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
        'train_size': len(train_dataloader),
        'val_size': len(val_dataloader),
        'trial.number': trial.number
    }

    wandb.init(
        project='winged-bull',
        config=config,
        group=STUDY_NAME,
        reinit=True
    )

    return trial, config, train_dataloader, val_dataloader


def objective(trial):
    trial, config, train_dataloader, val_dataloader = init_optuna(trial)
    model = LightningModel(config, trial)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        max_epochs=config['epochs'],
        accelerator='auto',
        precision=16
    )

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    return trainer.callback_metrics['best_score'].detach()


def get_data():
    train_dataloader, val_dataloader = get_dataloaders(BATCH_SIZE, VAL_RATE)  # 4 * num_GPU
    first_batch = train_dataloader.dataset[0]
    print('GET DATA...')

    return train_dataloader, val_dataloader, first_batch


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
