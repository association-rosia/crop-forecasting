from constants import FOLDER, S_COLUMNS, M_COLUMNS, G_COLUMNS

import pandas as pd
import xarray as xr

import torch
from torch.utils.data import Dataset

class DLDataset(Dataset):
    def __init__(self, data: xr.Dataset, test: bool=False, times: int=120):
        self.data: xr.Dataset = data
        self.test = test
        self.times = times

    def __len__(self):
        return self.data['ts_id'].shape[0]

    def __getitem__(self, idx):
        xdf_id = self.data.isel(ts_id=idx)
        
        g_input = torch.tensor(xdf_id[G_COLUMNS].to_array().values.astype('float64'), dtype=torch.float)

        s_input = torch.tensor(xdf_id[S_COLUMNS].to_array().values.T, dtype=torch.float)
        
        all_dates = pd.date_range(xdf_id['time'].min().values, xdf_id['time'].max().values, freq='d')
        all_dates = all_dates[-self.times:]
        m_input = torch.tensor(xdf_id.sel(datetime=all_dates, name=xdf_id['District'])[M_COLUMNS].to_array().values.T, dtype=torch.float)
        
        if self.test:
            label = xdf_id['Predicted Rice Yield (kg/ha)'].values
        else:
            label = xdf_id['Rice Yield (kg/ha)'].values
        
        item = {
            'district': xdf_id['District'].values, 
            'latitude': xdf_id['latitude'].values, 
            'longitude': xdf_id['longitude'].values, 
            'date_of_harvest': xdf_id['Date of Harvest'].values,
            's_input': s_input,
            'm_input': m_input,
            'g_input': g_input,
            'labels': label
        }

        return item