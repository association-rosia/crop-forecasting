import warnings

warnings.filterwarnings('ignore')

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

import torch
import torch.nn as nn
import wandb
from src.models.dataloader import get_dataloaders
from src.models.model import LSTMModel
from torch.utils.data import DataLoader
from src.models.trainer import Trainer


def main():
    # empty the GPU cache
    torch.cuda.empty_cache()

    # get the device
    device = get_device()

    # init W&B logger and get the model config from W&B sweep config yaml file
    # + get the training and validation dataloaders
    config, train_dataloader, val_dataloader = init_wandb()

    # init the model
    model = LSTMModel(config, device)
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


def init_wandb() -> tuple[dict, DataLoader, DataLoader]:
    """ Init W&B logger and get the model config from W&B sweep config yaml file
        + get the training and validation dataloaders.

    :return: the model config and the training and validation dataloaders
    :rtype: (dict, DataLoader, DataLoader)
    """

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    
    learning_rate = wandb.config.learning_rate
    scheduler_patience = wandb.config.scheduler_patience
    
    s_hidden_size = wandb.config.s_hidden_size
    m_hidden_size = wandb.config.m_hidden_size
    s_num_layers = wandb.config.s_num_layers
    m_num_layers = wandb.config.m_num_layers
    c_out_in_features_1 = wandb.config.c_out_in_features_1
    c_out_in_features_2 = wandb.config.c_out_in_features_2
    dropout = wandb.config.dropout
    
    train_dataloader, val_dataloader, _ = get_dataloaders(batch_size, 0.2, get_device())
    first_row = train_dataloader.dataset[0]
    
    c_in_features = s_hidden_size - 2 + m_hidden_size - 2 + first_row['g_input'].shape[0]

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'train_size': len(train_dataloader),
        'val_size': len(val_dataloader),
        'learning_rate': learning_rate,
        'scheduler_patience': scheduler_patience,
        's_hidden_size': s_hidden_size,
        'm_hidden_size': m_hidden_size,
        's_num_features': first_row['s_input'].shape[1],
        's_num_layers': s_num_layers,
        'm_num_layers': m_num_layers,
        'm_num_features': first_row['m_input'].shape[1],
        'g_in_features': first_row['g_input'].shape[0],
        'c_in_features': c_in_features,
        'c_out_in_features_1': c_out_in_features_1,
        'c_out_in_features_2': c_out_in_features_2,
        'dropout': dropout,
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
    
    if sys.argv[1] == '--manual' or sys.argv[1] == '-m':
        wandb.init(
            project="crop-forecasting",
            entity="winged-bull",
            group="test",
            config=dict(
                epochs=20,
                batch_size=16,
                learning_rate=0.0001,
                scheduler_patience=6,
                s_hidden_size=256,
                m_hidden_size=256,
                s_num_layers=2,
                m_num_layers=2,
                c_out_in_features_1=256,
                c_out_in_features_2=128,
                dropout=.4,
            ),
        )
    else:
        wandb.init(project="crop-forecasting", entity="winged-bull", group='Deep Learning')
    
    main()
