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

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from os.path import join

from utils import ROOT_DIR


class CustomDataset(Dataset):
    def __init__(
        self,
        s_inputs: np.ndarray,
        g_inputs: np.ndarray,
        m_inputs: np.ndarray,
        obs_targets: np.ndarray,
        augment: int,
        device: str,
    ):
        """Dataset used for the dataloader.

        :param s_inputs: Satellite data.
        :type s_inputs: np.ndarray
        :param g_inputs: Raw data.
        :type g_inputs: np.ndarray
        :param m_inputs: Meteorological data.
        :type m_inputs: np.ndarray
        :param obs_targets: Yield data.
        :type obs_targets: np.ndarray
        :param augment: Number of data augmentation.
        :type augment: int
        :param device: Training device.
        :type device: str
        """
        # Move data on the training device.
        self.augment = augment
        self.device = device
        self.s_inputs = torch.tensor(s_inputs).to(
            device=self.device, dtype=torch.float32
        )
        self.g_inputs = torch.tensor(g_inputs).to(
            device=self.device, dtype=torch.float32
        )
        self.m_inputs = torch.tensor(m_inputs).to(
            device=self.device, dtype=torch.float32
        )
        self.observations = torch.tensor(obs_targets[:, 0]).to(
            device=self.device, dtype=torch.float32
        )
        self.targets = torch.tensor(obs_targets[:, 1]).to(
            device=self.device, dtype=torch.float32
        )

    def __len__(self):
        return self.s_inputs.shape[0]

    def __getitem__(self, idx):
        # Return data for a particular indexe
        # The data depend only on the observation indexe
        # Only the satellite data depend on the augmentation indexe
        idx_obs = idx // self.augment
        item = {
            "observation": self.observations[[idx_obs]],
            "s_input": self.s_inputs[idx],
            "m_input": self.m_inputs[idx_obs],
            "g_input": self.g_inputs[idx_obs],
            "target": self.targets[[idx_obs]],
        }

        return item


def create_train_val_idx(xds: xr.Dataset, val_rate: float) -> tuple[list, list]:
    """Compute a stratifate Train/Val split.

    :param xds: Dataset used for the split.
    :type xds: xr.Dataset
    :param val_rate: Percentage of data in the validation set.
    :type val_rate: float
    :return: return list of train index & list of val index
    :rtype: tuple[list, list]
    """
    yields = xds[TARGET].values
    yields_distribution = stats.norm(loc=yields.mean(), scale=yields.std())
    bounds = yields_distribution.cdf([0, 1])
    bins = np.linspace(*bounds, num=10)
    stratify = np.digitize(yields, bins)
    train_idx, val_idx = train_test_split(
        xds.ts_obs, test_size=val_rate, random_state=42, stratify=stratify
    )

    return train_idx, val_idx


def transform_data(
    xds: xr.Dataset, m_times: int = 120, test=False
) -> dict[str, np.ndarray]:
    """Transform data from xr.Dataset to dict of np.ndarray
    sorted by observation and augmentation.

    :param xds: The Dataset to be transformed.
    :type xds: xr.Dataset
    :param m_times: Length of the time series for Weather data, defaults to 120.
    :type m_times: int, optional
    :param test: True if it is the Test dataset, defaults to False.
    :type test: bool, optional
    :return: Dictionnary of all data used to construct the torch Dataset.
    :rtype: dict[str, np.ndarray]
    """
    items = {}
    # Dataset sorting for compatibility with torch Dataset indexes
    xds = xds.sortby(["ts_obs", "ts_aug"])

    # Create raw data
    g_arr = xds[G_COLUMNS].to_dataframe()
    items["g_inputs"] = g_arr.values

    # Create satellite data
    # Keep only useful values and convert into numpy array
    s_arr = xds[S_COLUMNS].to_dataframe()[S_COLUMNS]
    s_arr = s_arr.to_numpy()
    # Reshape axis to match index, date, features
    # TODO: set as variable the number of state_dev and features.
    s_arr = s_arr.reshape(s_arr.shape[0] // 24, 24, 8)
    items["s_inputs"] = s_arr

    # Create Meteorological data
    # time and District are the keys to link observations and meteorological data
    df_time = xds[["time", "District"]].to_dataframe()
    # Keep only useful data
    df_time.reset_index(inplace=True)
    df_time = df_time[["ts_obs", "state_dev", "time", "District"]]
    # Meteorological data only dependend of the observation
    df_time = df_time.groupby(["ts_obs", "state_dev", "District"]).first()
    # Take the min and max datetime of satellite data to create a daily time series of meteorological data
    df_time.reset_index("state_dev", inplace=True)
    # TODO: set as variable the number of state_dev.
    df_time = df_time[df_time["state_dev"].isin([0, 23])]
    df_time = df_time.pivot(columns="state_dev").droplevel(None, axis=1)
    df_time.reset_index("District", inplace=True)

    # For each observation take m_times daily date before the
    # harverest date and get data with the corresponding location
    list_weather = []
    for _, series in df_time.iterrows():
        all_dates = pd.date_range(series[0], series[23], freq="D")
        all_dates = all_dates[-m_times:]
        m_arr = (
            xds.sel(datetime=all_dates, name=series["District"])[M_COLUMNS]
            .to_array()
            .values
        )
        list_weather.append(m_arr.T)

    items["m_inputs"] = np.asarray(list_weather)

    # If test create the target array with 0 instead of np.nan
    if test:
        df = xds[TARGET_TEST].to_dataframe().reset_index()
        df[TARGET_TEST] = 0
        items["obs_targets"] = df.to_numpy()
    else:
        items["obs_targets"] = xds[TARGET].to_dataframe().reset_index().to_numpy()

    items["augment"] = xds["ts_aug"].values.shape[0]

    return items


def get_dataloaders(
    batch_size: int, val_rate: float, device: str
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Generate Train / Validation / Test Torch Dataloader.

    :param batch_size: Batch size of Dataloader.
    :type batch_size: int
    :param val_rate: Percentage of data on the Validation Dataset.
    :type val_rate: float
    :param device: Device where to put the data.
    :type device: str
    :return: Train / Validation / Test Dataloader
    :rtype: tuple[DataLoader, DataLoader, DataLoader]
    """
    # Read the dataset processed
    dataset_path = join(ROOT_DIR, "data", "processed", FOLDER, "train_enriched.nc")
    xdf_train = xr.open_dataset(dataset_path, engine="scipy")

    # Create a Train / Validation split
    train_idx, val_idx = create_train_val_idx(xdf_train, val_rate)
    train_array = xdf_train.sel(ts_obs=train_idx)
    # Prepare data for th Torch Dataset
    items = transform_data(train_array)
    train_dataset = CustomDataset(**items, device=device)
    # Create the Dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
    )

    # ?: Make a function to create each dataloader
    val_array = xdf_train.sel(ts_obs=val_idx)
    items = transform_data(val_array)
    val_dataset = CustomDataset(**items, device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    dataset_path = join(ROOT_DIR, "data", "processed", FOLDER, "test_enriched.nc")
    xdf_test = xr.open_dataset(dataset_path, engine="scipy")
    items = transform_data(xdf_test, test=True)
    test_dataset = CustomDataset(**items, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader
