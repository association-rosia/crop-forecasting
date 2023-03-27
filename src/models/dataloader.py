from src.constants import S_COLUMNS, M_COLUMNS, G_COLUMNS, TARGET, FOLDER

import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)

from utils import ROOT_DIR
from os.path import join


class CustomDataset(Dataset):
    def __init__(self, data: xr.Dataset, test: bool = False, s_times: int = 24, m_times: int = 120):
        self.data: xr.Dataset = data
        self.test: bool = test
        self.m_times: int = m_times
        self.s_times: int = s_times

    def __len__(self):
        shape = self.data['ts_id'].shape
        return shape[0] * shape[1]

    def __getitem__(self, idx):
        xdf_id = self.data.where(self.data['ts_id'] == idx, drop=True)

        g_arr = xdf_id[G_COLUMNS].to_array().values
        g_arr = g_arr.reshape(-1)
        g_input = torch.tensor(g_arr)

        s_arr = xdf_id[S_COLUMNS].to_array().values
        s_arr = s_arr.reshape((len(S_COLUMNS), self.s_times)).T
        s_input = torch.tensor(s_arr)

        all_dates = pd.date_range(xdf_id['time'].min().values, xdf_id['time'].max().values, freq='D')
        all_dates = all_dates[-self.m_times:]
        g_arr = xdf_id.sel(datetime=all_dates, name=xdf_id['District'])[M_COLUMNS].to_array().values
        g_arr = g_arr.reshape((len(M_COLUMNS), self.m_times)).T
        m_input = torch.tensor(g_arr)

        if self.test:
            target = torch.tensor([0.])
        else:
            target = torch.tensor(xdf_id[TARGET].values)

        target = target.reshape(-1)
        observation = torch.tensor(xdf_id['ts_obs'].values)

        item = {
            'observation': observation.to(torch.float32),
            's_input': s_input.to(torch.float32),
            'm_input': m_input.to(torch.float32),
            'g_input': g_input.to(torch.float32),
            'target': target.to(torch.float32)
        }

        return item


def create_train_val_idx(xdf_train, val_rate):
    yields = xdf_train[TARGET][0, :].values
    yields_distribution = stats.norm(loc=yields.mean(), scale=yields.std())
    bounds = yields_distribution.cdf([0, 1])
    bins = np.linspace(*bounds, num=10)
    stratify = np.digitize(yields, bins)
    train_idx, val_idx = train_test_split(xdf_train.ts_obs,
                                          test_size=val_rate,
                                          random_state=42,
                                          stratify=stratify)

    return train_idx, val_idx


def get_dataloaders(batch_size, val_rate, num_workers=4):  # 4 * num_GPU
    dataset_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'train_enriched.nc')
    xdf_train = xr.open_dataset(dataset_path, engine='scipy')
    train_idx, val_idx = create_train_val_idx(xdf_train, val_rate)

    train_array = xdf_train.sel(ts_obs=train_idx)
    train_shape = train_array['ts_id'].shape
    train_array['ts_id'].values = np.arange(np.prod(train_shape)).reshape(train_shape)
    train_dataset = CustomDataset(train_array)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    val_array = xdf_train.sel(ts_obs=val_idx)
    val_shape = val_array['ts_id'].shape
    val_array['ts_id'].values = np.arange(np.prod(val_shape)).reshape(val_shape)
    val_dataset = CustomDataset(val_array)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True)

    dataset_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'test_enriched.nc')
    xdf_test = xr.open_dataset(dataset_path, engine='scipy')
    test_dataset = CustomDataset(xdf_test, test=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_data(batch_size, val_rate):
    train_dataloader, val_dataloader, _ = get_dataloaders(batch_size, val_rate)
    first_batch = train_dataloader.dataset[0]

    return train_dataloader, val_dataloader, first_batch
