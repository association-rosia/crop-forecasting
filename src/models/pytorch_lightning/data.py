from sklearn.model_selection import train_test_split

from src.constants import TARGET, FOLDER

from torch.utils.data import DataLoader
from src.models.dataloader import CustomDataset

import xarray as xr
import numpy as np

from scipy import stats

import os
import sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)
from utils import ROOT_DIR
from os.path import join


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
                                  shuffle=True)

    val_array = xdf_train.sel(ts_obs=val_idx)
    val_shape = val_array['ts_id'].shape
    val_array['ts_id'].values = np.arange(np.prod(val_shape)).reshape(val_shape)
    val_dataset = CustomDataset(val_array)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers)

    return train_dataloader, val_dataloader
