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
    def __init__(self, data: xr.Dataset, device: str, test: bool = False, m_times: int = 120):
        self.device = device
        self.test: bool = test
        self.m_times: int = m_times
        self.transform_data(data, device)
        self.augment = data['ts_aug'].shape[0]
        
        
    def __len__(self):
        return self.g_input[0] * self.augment

    def transform_data(self, xds: xr.Dataset)->pd.DataFrame:
        g_arr = xds[G_COLUMNS].to_array().values
        self.g_input = torch.tensor(g_arr.T).to(device=self.device, dtype=torch.float32)

        s_arr = xds[S_COLUMNS].to_array().values
        s_arr = s_arr.reshape((s_arr.shape[0], np.prod(s_arr.shape[1:3]), s_arr.shape[-1]))
        s_arr = s_arr.T.swapaxes(0, 1)
        self.s_input = torch.tensor(s_arr).to(device=self.device, dtype=torch.float32)

        df_time = xds[['time', 'District']].to_dataframe()
        df_time: pd.DataFrame = df_time.loc[:, [0, 23], 0]
        df_time = df_time.reset_index('ts_aug', drop=True).drop(columns='ts_id')
        df_time = df_time.reset_index('state_dev').set_index('District', append=True).pivot(columns='state_dev')
        df_time.columns = df_time.columns.get_level_values('state_dev')
        df_time.reset_index('District', inplace=True)

        list_weather = []
        for _, series in df_time.iterrows():
            all_dates = pd.date_range(series[0], series[23], freq='D')
            all_dates = all_dates[-self.m_times:]
            m_arr = xds.sel(datetime=all_dates, name=series['District'])[M_COLUMNS].to_array().values
            list_weather.append(m_arr.T)

        m_arr = np.asarray(list_weather)
        self.m_input = torch.tensor(m_arr).to(device=self.device, dtype=torch.float32)

        if self.test:
            self.target = torch.tensor([0.] * xds[TARGET_TEST].values.shape[0]).to(device=self.device, dtype=torch.float32)
        else:
            self.target = torch.tensor(xds[TARGET].values).to(device=self.device, dtype=torch.float32)

        self.observation = torch.tensor(xds['ts_obs'].values).to(device=self.device, dtype=torch.float32)


    def __getitem__(self, idx):
        idx_obs = idx // self.augment

        item = {
            'observation': self.observation[idx_obs],
            's_input': self.s_input[idx],
            'm_input': self.m_input[idx_obs],
            'g_input': self.g_input[idx_obs],
            'target': self.target[idx_obs]
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
