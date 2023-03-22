import pytorch_lightning as pl
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


class LightningData(pl.LightningDataModule):
    def __init__(self, config, num_workers=0):
        super(LightningData, self).__init__()
        self.batch_size = config['batch_size']
        self.val_rate = config['val_rate']
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def create_train_val_idx(self, xdf_train):
        yields = xdf_train[TARGET][0, :].values
        yields_distribution = stats.norm(loc=yields.mean(), scale=yields.std())
        bounds = yields_distribution.cdf([0, 1])
        bins = np.linspace(*bounds, num=10)
        stratify = np.digitize(yields, bins)
        train_idx, val_idx = train_test_split(xdf_train.ts_obs,
                                              test_size=self.val_rate,
                                              random_state=42,
                                              stratify=stratify)

        return train_idx, val_idx

    def setup(self, stage):
        if stage == 'fit':
            dataset_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'train_enriched.nc')
            xdf_train = xr.open_dataset(dataset_path, engine='scipy')
            train_idx, val_idx = self.create_train_val_idx(xdf_train)

            train_array = xdf_train.sel(ts_obs=train_idx)
            train_shape = train_array['ts_id'].shape
            train_array['ts_id'].values = np.arange(np.prod(train_shape)).reshape(train_shape)
            self.train_dataset = CustomDataset(train_array)

            val_array = xdf_train.sel(ts_obs=val_idx)
            val_shape = val_array['ts_id'].shape
            val_array['ts_id'].values = np.arange(np.prod(val_shape)).reshape(val_shape)
            self.val_dataset = CustomDataset(val_array)

        if stage == 'test':
            dataset_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'test_enriched.nc')
            xdf_test = xr.open_dataset(dataset_path, engine='scipy')
            self.test_dataset = CustomDataset(xdf_test, test=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

