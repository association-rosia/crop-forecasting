from constants import S_COLUMNS, M_COLUMNS, G_COLUMNS

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch.utils.data import Dataset

class DLDataset(Dataset):
    def __init__(self, data: xr.Dataset, test: bool=False, s_times: int=24, m_times: int=120):
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
        g_arr = g_arr.reshape(-1).astype(np.float32)
        g_input = torch.tensor(g_arr)

        s_arr = xdf_id[S_COLUMNS].to_array().values
        s_arr = s_arr.reshape((len(S_COLUMNS), self.s_times)).T.astype(np.float32)
        s_input = torch.tensor(s_arr)
        
        all_dates = pd.date_range(xdf_id['time'].min().values, xdf_id['time'].max().values, freq='D')
        all_dates = all_dates[-self.m_times:]
        g_arr = xdf_id.sel(datetime=all_dates, name=xdf_id['District'])[M_COLUMNS].to_array().values
        g_arr = g_arr.reshape((len(M_COLUMNS), self.m_times)).T.astype(np.float32)
        m_input = torch.tensor(g_arr)
        
        if self.test:
            target = xdf_id['Predicted Rice Yield (kg/ha)'].values
        else:
            target = xdf_id['Rice Yield (kg/ha)'].values

        target = target.reshape(-1)


        item: dict = {
            'observation': xdf_id['ts_obs'].values,
            's_input': s_input,
            'm_input': m_input,
            'g_input': g_input,
            'target': target
        }

        return item