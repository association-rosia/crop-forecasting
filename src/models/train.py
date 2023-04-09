import warnings

warnings.filterwarnings('ignore')

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

import torch
import torch.nn as nn
import wandb
from dataloader import get_dataloaders
from model import CustomModel
from torch.utils.data import DataLoader
from trainer import Trainer


def main():
    # empty the GPU cache
    torch.cuda.empty_cache()

    # get the device
    device = get_device()

    # init W&B logger and get the model config from W&B sweep config yaml file
    # + get the training and validation dataloaders
    config, train_dataloader, val_dataloader = init_wandb()

    # init the model
    model = CustomModel(config)
    model.to(device)

    # init the loss, optimizer and learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=config['scheduler_patience'],
                                                           verbose=True)

    train_config = {
        'model': model,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'epochs': config['epochs'],
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }

    # init the trainer
    trainer = Trainer(**train_config)

    # train the model
    trainer.train()


def init_wandb() -> (dict, DataLoader, DataLoader):
    """ Init W&B logger and get the model config from W&B sweep config yaml file
        + get the training and validation dataloaders.

    :return: the model config and the training and validation dataloaders
    :rtype: (dict, DataLoader, DataLoader)
    """

    wandb.init(
        project='winged-bull',
        group='crop-forecasting',
    )

    epochs = wandb.config.epochs
    lstm_dropout = wandb.config.lstm_dropout
    cnn_dropout = wandb.config.cnn_dropout
    fc_dropout = wandb.config.fc_dropout
    criterion = wandb.config.criterion
    optimizer = wandb.config.optimizer
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    scheduler_patience = wandb.config.scheduler_patience
    c_out_in_features_1 = wandb.config.c_out_in_features_1
    c_out_in_features_2 = wandb.config.c_out_in_features_2
    m_num_layers = wandb.config.m_num_layers
    s_num_layers = wandb.config.s_num_layers
    s_hidden_size = wandb.config.s_hidden_size
    m_hidden_size = wandb.config.m_hidden_size
    train_dataloader, val_dataloader, _ = get_dataloaders(batch_size, 0.2, get_device())
    first_row = train_dataloader.dataset[0]
    
    c_in_features = s_hidden_size - 2 + m_hidden_size - 2 + first_row['g_input'].shape[0]

    config = {
        'batch_size': batch_size,
        's_hidden_size': s_hidden_size,
        's_num_layers': s_num_layers,
        'm_hidden_size': m_hidden_size,
        'm_num_layers': m_num_layers,
        'learning_rate': learning_rate,
        'scheduler_patience': scheduler_patience,
        'lstm_dropout': lstm_dropout,
        'cnn_dropout': cnn_dropout,
        'fc_dropout': fc_dropout,
        'epochs': epochs,
        'optimizer': optimizer,
        'criterion': criterion,
        's_num_features': first_row['s_input'].shape[1],
        'm_num_features': first_row['m_input'].shape[1],
        'g_in_features': first_row['g_input'].shape[0],
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
