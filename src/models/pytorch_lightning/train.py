import warnings
warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from model import LightningModel
from data import get_dataloaders

import torch
import wandb

GROUP_NAME = 'crop-forecasting'
VAL_RATE = 0.2


def train():
    torch.cuda.empty_cache()
    
    run = wandb.init(
        project='winged-bull',
        group=GROUP_NAME,
    )

    config, train_dataloader, val_dataloader = init_wandb()

    model = LightningModel(config, run)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        max_epochs=config['epochs'],
        accelerator='auto',
    )

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    wandb.finish()

    return trainer.callback_metrics['best_score'].detach()


def init_wandb():
    epochs = wandb.config.epochs # 20
    dropout = wandb.config.dropout # trial.suggest_uniform('dropout', 0.5, 0.8)
    criterion = wandb.config.criterion # MSELoss
    optimizer = wandb.config.optimizer # trial.suggest_categorical('optimizer', choices=['AdamW', 'RMSprop'])
    batch_size = wandb.config.batch_size # trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    learning_rate = wandb.config.learning_rate # trial.suggest_loguniform('learning_rate', 5e-5, 5e-3)
    c_out_in_features_1 = wandb.config.c_out_in_features_1 # trial.suggest_int('c_out_in_features_1', int(sqrt(c_in_features)), 2 * c_in_features)
    c_out_in_features_2 = wandb.config.c_out_in_features_2 # trial.suggest_int('c_out_in_features_2', int(sqrt(c_in_features)), 2 * c_in_features)
    m_num_layers = wandb.config.m_num_layers # trial.suggest_int('m_num_layers', 1, 2)
    s_num_layers = wandb.config.s_num_layers # trial.suggest_int('s_num_layers', 1, 2)
    s_hidden_size = wandb.config.s_hidden_size # trial.suggest_int('s_hidden_size', 64, 256)
    m_hidden_size = wandb.config.m_hidden_size # trial.suggest_int('m_hidden_size', 64, 256)
    train_dataloader, val_dataloader, first_batch = get_data(batch_size)
    c_in_features = s_hidden_size - 2 + m_hidden_size - 2 + first_batch['g_input'].shape[0]

    config = {
        'batch_size': batch_size,
        's_hidden_size': s_hidden_size,
        's_num_layers': s_num_layers,
        'm_hidden_size': m_hidden_size,
        'm_num_layers': m_num_layers,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'epochs': epochs,
        'optimizer': optimizer,
        'criterion': criterion,
        's_num_features': first_batch['s_input'].shape[1],
        'm_num_features': first_batch['m_input'].shape[1],
        'g_in_features': first_batch['g_input'].shape[0],
        'c_in_features': c_in_features,
        'c_out_in_features_1': c_out_in_features_1,
        'c_out_in_features_2': c_out_in_features_2,
        'train_size': len(train_dataloader),
        'val_size': len(val_dataloader),
    }

    return config, train_dataloader, val_dataloader


def get_data(batch_size):
    train_dataloader, val_dataloader = get_dataloaders(batch_size, VAL_RATE)  # 4 * num_GPU
    first_batch = train_dataloader.dataset[0]

    return train_dataloader, val_dataloader, first_batch


if __name__ == '__main__':
    train()
