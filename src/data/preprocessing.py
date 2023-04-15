import warnings
from typing import Union

warnings.filterwarnings("ignore")

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from os.path import join

from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from src.constants import G_COLUMNS, M_COLUMNS, S_COLUMNS
from utils import ROOT_DIR


class Sorter(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Sort dataset to align dataset samples with labels samples."""

    def __init__(self) -> None:
        pass

    def fit(self, X=None, y=None) -> object:
        """Identity function

        :param X: Ignored
        :type X: None
        :param y: Ignored
        :type y: None
        :return: self
        :rtype: object
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder the indexes in an ascending way first by observation then by augmentation.

        :param X: Dataset that will be transformed.
        :type X: pd.DataFrame
        :return: Transformed Dataframe.
        :rtype: pd.DataFrame
        """
        return X.reorder_levels(["ts_obs", "ts_aug"]).sort_index()


# Convertor class used on ML exploration
class Convertor(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Used to transform the xarray.Dataset into pandas.DataFrame and reduce the dimention and/or tranform it.

    :param agg: If True then replace features with their aggregations along the state_dev axis (agg = min, mean, max), defaults to None
    :type agg: bool, optional
    :param weather: If False then remove weather data from the Dataset, defaults to True
    :type weather: bool, optional
    :param vi: If False then remove vegetable indices from the Dataset, defaults to True
    :type vi: bool, optional
    """

    def __init__(self, agg: bool = None, weather: bool = True, vi: bool = True) -> None:
        self.agg = agg
        self.weather = weather
        self.vi = vi

    def to_dataframe(self, X: xr.Dataset) -> pd.DataFrame:
        # Convert xarray.Dataset into usable pandas.DataFrame

        # Depend of aggregations was performed, change the columns name
        col = "agg" if self.agg else "state_dev"
        # Convert xarray.Dataset into pandas.DataFrame
        df = X.to_dataframe()
        # set G_COLUMNS as index to not be duplicate by the pivot operation
        df.set_index(G_COLUMNS, append=True, inplace=True)
        # reset the columns use to apply the pivot and convert its values into string
        df.reset_index(col, inplace=True)
        df[col] = df[col].astype(str)
        # Apply pivot to change state_dev or agg from samples to features
        df = df.pivot(columns=col)
        # Convert pandas.MultiIndex to a pandas.Index by merging names
        df.columns = df.columns.map("_".join).str.strip("_")
        # set G_COLUMNS as features
        df.reset_index(G_COLUMNS, inplace=True)
        # sort dataset for future compability
        df = df.reorder_levels(["ts_obs", "ts_aug"]).sort_index()
        return df

    def merge_dimensions(self, X: xr.Dataset) -> xr.Dataset:
        # Merge VI, Geographical and Meteorological data into the same dimension
        X = xr.merge(
            [
                X[G_COLUMNS],
                X[M_COLUMNS].sel(datetime=X["time"], name=X["District"]),
                X[S_COLUMNS],
            ]
        )
        # Drop useless columns
        X = X.drop(["name", "datetime", "time"])
        return X

    def compute_agg(self, X: xr.Dataset) -> xr.Dataset:
        # Compute aggregation on the Dataset and set the new dimension values
        # with the name of each aggregation performed
        X = xr.concat(
            [X.mean(dim="state_dev"), X.max(dim="state_dev"), X.min(dim="state_dev")],
            dim="agg",
        )
        X["agg"] = ["mean", "max", "min"]
        return X

    def fit(self, X=None, y=None) -> object:
        """Identity function.

        :param X: Ignored
        :type X: None
        :param y: Ignored
        :type y: None
        :return: Convertor.
        :rtype: object
        """
        return self

    def transform(self, X: xr.Dataset) -> pd.DataFrame:
        """Transform the xarray.Dataset to pandas.Dataframe depends on the argument of the class.

        :param X: Dataset that will be transformed.
        :type X: xr.Dataset
        :return: Transformed Dataset.
        :rtype: pd.DataFrame
        """
        # Transform data to depends of the sames dimentions
        X = self.merge_dimensions(X)
        # If True, compute aggregation to the data
        if self.agg:
            X = self.compute_agg(X)
        # If False, remove weather data
        if not self.weather:
            X = X.drop(M_COLUMNS)
        # If False, remove vi data
        if not self.vi:
            X = X.drop(S_COLUMNS)
        # Convert the Dataset into a DataFrame
        X = self.to_dataframe(X)
        return X


class Smoother(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Smooth Vegetable Indice Data.

    :param mode: methode used to smooth vi data, None to not perform smoothing during , defaults to "savgol"
    :type mode: str, optional
    """

    def __init__(self, mode: str = "savgol") -> None:
        self.mode = mode

    def smooth_savgol(self, ds: xr.Dataset) -> xr.Dataset:
        # apply savgol_filter to vegetable indice
        ds_s = xr.apply_ufunc(
            savgol_filter,
            ds[S_COLUMNS],
            kwargs={"axis": 2, "window_length": 12, "polyorder": 4, "mode": "mirror"},
        )
        # merge both dataset and override old vegetable indice and bands
        return xr.merge([ds_s, ds], compat="override")

    def fit(self, X: xr.Dataset = None, y=None) -> object:
        """Identity function.

        :param X: Ignored, defaults to None
        :type X: xr.Dataset, optional
        :param y: Ignored, defaults to None
        :type y: _type_, optional
        :return: Themself.
        :rtype: object
        """
        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Smooth Vegetable Indice Data according to the mode used.

        :param X: Dataset that will be transformed.
        :type X: xr.Dataset
        :return: Transformed Dataset.
        :rtype: xr.Dataset
        """
        # If mode not equal to savgol, transform correspond to identity function.
        if self.mode == "savgol":
            X = self.smooth_savgol(X)

        return X


def replaceinf(arr: np.ndarray) -> np.ndarray:
            if np.issubdtype(arr.dtype, np.number):
                arr[np.isinf(arr)] = np.nan
            return arr


class Filler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Fill dataset using the mean of each group of observation for a given date.
    For the reaining data use the mean of the dataset for a given developpment state.
    """

    def __init__(self) -> None:
        self.values = None

    def fit(self, X: xr.Dataset, y=None) -> object:
        """Compute mean by developpement state to be used for later filling.

        :param X: The data used to compute mean by developpement state used for later filling.
        :type X: xr.Dataset
        :param y: Ignored
        :type y: None
        :return: self
        :rtype: object
        """
        # replace infinite value by na
        xr.apply_ufunc(replaceinf, X[S_COLUMNS])
        # compute mean of all stage of developpement for each cluster obsevation
        self.values = (
            X[S_COLUMNS].mean(dim="ts_aug", skipna=True).mean(dim="ts_obs", skipna=True)
        )

        return self

    def transform(self, X: xr.Dataset) -> xr.Dataset:
        """Performs the filling of missing values

        :param X: The dataset used to fill.
        :type X: xr.Dataset
        :return: Transformed Dataset.
        :rtype: xr.Dataset
        """
        # replace infinite value by na
        xr.apply_ufunc(replaceinf, X[S_COLUMNS])
        # compute mean of all stage of developpement and all obsevation
        X[S_COLUMNS] = X[S_COLUMNS].fillna(X[S_COLUMNS].mean(dim="ts_aug", skipna=True))
        # fill na value with fited mean
        X[S_COLUMNS] = X[S_COLUMNS].fillna(self.values)

        return X
