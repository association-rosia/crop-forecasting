import warnings
warnings.filterwarnings('ignore')

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

import torch
import torch.nn as nn
from dataloader import get_data
from model import CustomModel
from trainer import Trainer
import wandb


def main():
    torch.cuda.empty_cache()
    device = get_device()

    config, train_dataloader, val_dataloader = init_wandb()

    model = CustomModel(config)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    train_config = {
        'model': model,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'optimizer': optimizer,
        'criterion': criterion,
        'epochs': config['epochs'],
        'device': device
    }

    trainer = Trainer(**train_config)
    trainer.train()


def init_wandb():
    wandb.init(
        project='winged-bull',
        group='crop-forecasting',
    )

    epochs = wandb.config.epochs
    dropout = wandb.config.dropout
    criterion = wandb.config.criterion
    optimizer = wandb.config.optimizer
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    c_out_in_features_1 = wandb.config.c_out_in_features_1
    c_out_in_features_2 = wandb.config.c_out_in_features_2
    m_num_layers = wandb.config.m_num_layers
    s_num_layers = wandb.config.s_num_layers
    s_hidden_size = wandb.config.s_hidden_size
    m_hidden_size = wandb.config.m_hidden_size
    train_dataloader, val_dataloader, first_batch = get_data(batch_size, val_rate=0.2)
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


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        raise Exception("None accelerator available")

    return device


if __name__ == '__main__':
    main()
