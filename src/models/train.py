from math import sqrt
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

from constants import FOLDER, S_COLUMNS, M_COLUMNS, G_COLUMNS

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import torch

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader import DLDataset, get_loaders
from model import LSTMModel
from trainer import Trainer
import wandb


def main():
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wandb.init(
        project='winged-bull',
        config = {
            'batch_size': 16, # try 4, 8, 16, 32
            'hidden_size': 128, # try 128 to 512
            'num_layers': 2, # try 1 to 4
            'learning_rate': 1e-4,
            'dropout': 0.2,
            'epochs': 100,
            'optimizer': 'AdamW', 
            'criterion': 'MSELoss',
            'val_rate': 0.2
        }
    )

    train_loader, val_loader, test_loader = get_loaders(wandb.config, num_workers=4)
    first_batch = train_loader.dataset[0]
    
    wandb.config['train_size'] = len(train_loader.dataset)
    wandb.config['val_size'] = len(val_loader.dataset)
    wandb.config['s_num_features'] = first_batch['s_input'].shape[1]
    wandb.config['m_num_features'] = first_batch['m_input'].shape[1]
    wandb.config['g_in_features'] = first_batch['g_input'].shape[0]
    wandb.config['c_in_features'] = 2 * (wandb.config['hidden_size'] - 2) + wandb.config['g_in_features']

    model = LSTMModel(wandb.config, device)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, patience=10)

    train_config = {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'epochs': wandb.config['epochs'],
        'device': device
    }
    
    trainer = Trainer(**train_config)
    trainer.train()
    
    # make_submission(trainer.model, test_loader)
    
    
if __name__ == '__main__':
    main()