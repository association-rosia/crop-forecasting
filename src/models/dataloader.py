import os
import sys

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.constants import FOLDER, G_COLUMNS, M_COLUMNS, S_COLUMNS, TARGET, TARGET_TEST

parent = os.path.abspath('.')
sys.path.insert(1, parent)

from os.path import join

from utils import ROOT_DIR



class JupyterDataset(Dataset):
    def __init__(self, s_inputs, g_inputs, m_inputs, obs_targets, augment, device):
        self.augment = augment
        self.device = device
        self.s_inputs = torch.tensor(s_inputs).to(device=self.device, dtype=torch.float32)
        self.g_inputs = torch.tensor(g_inputs).to(device=self.device, dtype=torch.float32)
        self.m_inputs = torch.tensor(m_inputs).to(device=self.device, dtype=torch.float32)
        self.observations = torch.tensor(obs_targets[:, 0]).to(device=self.device, dtype=torch.float32)
        self.targets = torch.tensor(obs_targets[:, 1]).to(device=self.device, dtype=torch.float32)
        
        
    def __len__(self):
        return self.s_inputs.shape[0]

    def __getitem__(self, idx):
        idx_obs = idx // self.augment
        item = {
            'observation': self.observations[[idx_obs]],
            's_input': self.s_inputs[idx],
            'm_input': self.m_inputs[idx_obs],
            'g_input': self.g_inputs[idx_obs],
            'target': self.targets[[idx_obs]]
        }

        return item


def create_train_val_idx(xdf_train, val_rate):
    yields = xdf_train[TARGET].values
    yields_distribution = stats.norm(loc=yields.mean(), scale=yields.std())
    bounds = yields_distribution.cdf([0, 1])
    bins = np.linspace(*bounds, num=10)
    stratify = np.digitize(yields, bins)
    train_idx, val_idx = train_test_split(xdf_train.ts_obs,
                                          test_size=val_rate,
                                          random_state=42,
                                          stratify=stratify)

    return train_idx, val_idx


def transform_data(xds: xr.Dataset, m_times: int = 120, test = False):
    items = {}
    xds = xds.sortby(['ts_obs', 'ts_aug'])
    g_arr = xds[G_COLUMNS].to_dataframe()
    items['g_inputs'] = g_arr.values

    s_arr = xds[S_COLUMNS].to_dataframe()[S_COLUMNS]
    s_arr = s_arr.to_numpy()
    s_arr = s_arr.reshape(s_arr.shape[0] // 24, 24, 8)
    items['s_inputs'] = s_arr

    df_time = xds[['time', 'District']].to_dataframe()
    df_time.reset_index(inplace=True)
    df_time = df_time[['ts_obs', 'state_dev', 'time', 'District']]
    df_time = df_time.groupby(['ts_obs', 'state_dev', 'District']).first()
    df_time.reset_index('state_dev', inplace=True)
    df_time = df_time[df_time['state_dev'].isin([0, 23])]
    df_time = df_time.pivot(columns='state_dev').droplevel(None, axis=1)
    df_time.reset_index('District', inplace=True)

    list_weather = []
    for _, series in df_time.iterrows():
        all_dates = pd.date_range(series[0], series[23], freq='D')
        all_dates = all_dates[-m_times:]
        m_arr = xds.sel(datetime=all_dates, name=series['District'])[M_COLUMNS].to_array().values
        list_weather.append(m_arr.T)

    items['m_inputs'] = np.asarray(list_weather)

    if test:
        df = xds[TARGET_TEST].to_dataframe().reset_index()
        df[TARGET_TEST] = 0
        items['obs_targets'] = df.to_numpy()
    else:
        items['obs_targets'] = xds[TARGET].to_dataframe().reset_index().to_numpy()

    items['augment'] = xds['ts_aug'].values.shape[0]

    return items


def get_dataloaders(batch_size, val_rate, device):  # 4 * num_GPU
    dataset_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'train_enriched.nc')
    xdf_train = xr.open_dataset(dataset_path, engine='scipy')

    train_idx, val_idx = create_train_val_idx(xdf_train, val_rate)
    train_array = xdf_train.sel(ts_obs=train_idx)
    items = transform_data(train_array)
    items['device'] = device
    train_dataset = JupyterDataset(**items)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True)

    val_array = xdf_train.sel(ts_obs=val_idx)
    items = transform_data(val_array)
    items['device'] = device
    val_dataset = JupyterDataset(**items)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                drop_last=True)

    dataset_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'test_enriched.nc')
    xdf_test = xr.open_dataset(dataset_path, engine='scipy')
    items = transform_data(xdf_test, test=True)
    items['device'] = device
    test_dataset = JupyterDataset(**items)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


def get_data(batch_size, val_rate, device):
    train_dataloader, val_dataloader, _ = get_dataloaders(batch_size, val_rate, device)
    first_batch = train_dataloader.dataset[0]

    return train_dataloader, val_dataloader, first_batch
