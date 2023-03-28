import glob
from typing import Union
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import savgol_filter
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
)

import os, sys

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from src.constants import S_COLUMNS, G_COLUMNS, M_COLUMNS

from utils import ROOT_DIR
from os.path import join


from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator

# Scaler class used on ML exploration
class Scaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        scaler: Union[
            MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
        ] = StandardScaler(),
    ) -> None:
        """Scale an array. The method depend of the scaler given.

        :param scaler: Scaler used, defaults to StandardScaler()
        :type scaler: Union[MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer], optional
        """
        self.scaler = scaler

    def fit(self, X: pd.DataFrame, y=None) -> object:
        """Fit the scaler initialised at scaler.

        :param X: The data used to fit the scaler, used for later scaling along the features axis.
        :type X: pd.DataFrame
        :param y: Ignored, defaults to None
        :type y: None, optional
        :return: Fitted scaler.
        :rtype: object
        """
        self.scaler = self.scaler.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features of X according to feature_range.

        :param X: Dataframe that will be transformed.
        :type X: pd.DataFrame
        :return: Transformed data.
        :rtype: pd.DataFrame
        """
        return self.scaler.transform(X)


# Convertor class used on ML exploration
class Convertor(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, agg: bool = None, weather: bool = True, vi: bool = True) -> None:
        """Used to transform the xarray.Dataset into pandas.DataFrame and reduce the dimention and/or tranform it.

        :param agg: If True then replace features with their aggregations along the state_dev axis (agg = min, mean, max), defaults to None
        :type agg: bool, optional
        :param weather: If False then remove weather data from the Dataset, defaults to True
        :type weather: bool, optional
        :param vi: If False then remove vegetable indices from the Dataset, defaults to True
        :type vi: bool, optional
        """
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

    def fit(self, X: xr.Dataset = None, y=None) -> object:
        """Identity function.

        :param X: Ignored, defaults to None
        :type X: xr.Dataset, optional
        :param y: Ignored, defaults to None
        :type y: None, optional
        :return: Convertor.
        :rtype: object
        """
        return self

    def transform(self, X: xr.Dataset) -> pd.DataFrame:
        """Transform the xarray.Dataset to pandas.Dataframe depends on the argument of the class.

        :param X: Dataset that will be transformed.
        :type X: xr.Dataset
        :return: Dataset transformed.
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
    def __init__(self, mode: str = "savgol") -> None:
        """Smooth Vegetable Indice Data.

        :param mode: methode used to smooth vi data, None to not perform smoothing during , defaults to "savgol"
        :type mode: str, optional
        """
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
        :return: Dataset transformed.
        :rtype: xr.Dataset
        """
        # If mode not equal to savgol, transform correspond to identity function.
        if self.mode == "savgol":
            X = self.smooth_savgol(X)

        return X


class Concatenator:
    """Adding EY data and Weather data to the Dataset.
    Encode Categorical EY data.
    Modify Weather data.
    Compute Vegetable Indice and fill missing data.
    """

    def __init__(self) -> None:
        pass

    def add_observation(self, ds: xr.Dataset, test: bool) -> xr.Dataset:
        # Process and Merge EY data to Satellite Dataset

        def categorical_encoding(ds: xr.Dataset) -> xr.Dataset:
            # Encode Rice Crop Intensity feature D = 2 and T = 3
            ds["Rice Crop Intensity(D=Double, T=Triple)"] = (
                ds["Rice Crop Intensity(D=Double, T=Triple)"]
                .str.replace("D", "2")
                .str.replace("T", "3")
                .astype(np.int8)
            )
            return ds

        # Select the right file
        if test:
            file_name = "test.csv"
        else:
            file_name = "train.csv"

        path = join(ROOT_DIR, "data", "raw", file_name)
        # Read csv EY data
        df = pd.read_csv(path)
        # Set index name as ts_obs for linked both Dataset
        df.index.name = "ts_obs"
        # Convert pandas.DataFrame into xarray.Dataset and merge on ts_obs
        ds = xr.merge([ds, df.to_xarray()])
        # Encode categoricals data
        ds = categorical_encoding(ds)

        return ds

    def add_weather(self, ds: xr.Dataset) -> xr.Dataset:
        # Process and Merge Weather data to Satellite & EY Dataset

        def features_modification(ds: xr.Dataset) -> xr.Dataset:
            # Crreate new features named solarexposure
            # It is the difference between sunset and sunrise
            ds["sunrise"] = ds["sunrise"].astype(np.datetime64)
            ds["sunset"] = ds["sunset"].astype(np.datetime64)

            ds["solarexposure"] = (ds["sunset"] - ds["sunrise"]).dt.seconds
            return ds

        # Read all weather csv and create a pandas.DataFrame of its
        weather = []
        for path in glob.glob(join(ROOT_DIR, "data", "external", "weather", "*.csv")):
            weather.append(pd.read_csv(path))
        df_weather = pd.concat(weather, axis="index")

        # Convert timestamp into datetime for future purpose
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        # Format name to match District features
        df_weather["name"] = df_weather["name"].str.replace(" ", "_")
        # Set as index datetime and name to became dimensions with the
        # xarray.Dataset conversion
        df_weather.set_index(["datetime", "name"], inplace=True)
        ds_weather = df_weather.to_xarray().set_coords(["datetime", "name"])
        ds_weather["datetime"] = ds_weather["datetime"].dt.strftime("%Y-%m-%d")
        # Feature engineering on weather data
        ds_weather = features_modification(ds_weather)
        # Merge both Dataset
        ds = xr.merge([ds, ds_weather])

        return ds

    def compute_vi(self, ds: xr.Dataset) -> xr.Dataset:
        # Compute vegetable indices

        def compute_ndvi(ds: xr.Dataset) -> xr.Dataset:
            # Compute ndvi indice
            return (ds.nir - ds.red) / (ds.nir + ds.red)

        def compute_savi(ds, L=0.5) -> xr.Dataset:
            # Compute savi indice
            return 1 + L * (ds.nir - ds.red) / (ds.nir + ds.red + L)

        def compute_evi(ds, G=2.5, L=1, C1=6, C2=7.5) -> xr.Dataset:
            # Compute evi indice
            return G * (ds.nir - ds.red) / (ds.nir + C1 * ds.red - C2 * ds.blue + L)

        def compute_rep(ds: xr.Dataset) -> xr.Dataset:
            # Compute rep indice
            rededge = (ds.red + ds.rededge3) / 2
            return 704 + 35 * (rededge - ds.rededge1) / (ds.rededge2 - ds.rededge1)

        def compute_osavi(ds: xr.Dataset) -> xr.Dataset:
            # Compute osavi indice
            return (ds.nir - ds.red) / (ds.nir + ds.red + 0.16)

        def compute_rdvi(ds: xr.Dataset) -> xr.Dataset:
            # Compute rdvi indice
            return (ds.nir - ds.red) / np.sqrt(ds.nir + ds.red)

        def compute_mtvi1(ds: xr.Dataset) -> xr.Dataset:
            # Compute mtvi1 indice
            return 1.2 * (1.2 * (ds.nir - ds.green) - 2.5 * (ds.red - ds.green))

        def compute_lswi(ds: xr.Dataset) -> xr.Dataset:
            # Compute lswi indice
            return (ds.nir - ds.swir) / (ds.nir + ds.swir)

        ds["ndvi"] = compute_ndvi(ds)
        ds["savi"] = compute_savi(ds)
        ds["evi"] = compute_evi(ds)
        ds["rep"] = compute_rep(ds)
        ds["osavi"] = compute_osavi(ds)
        ds["rdvi"] = compute_rdvi(ds)
        ds["mtvi1"] = compute_mtvi1(ds)
        ds["lswi"] = compute_lswi(ds)

        return ds

    def statedev_fill(self, ds: xr.Dataset) -> xr.Dataset:
        # Fill missing vegetable indice and replace abnormal values

        def replaceinf(arr: np.ndarray) -> np.ndarray:
            if np.issubdtype(arr.dtype, np.number):
                arr[np.isinf(arr)] = np.nan
            return arr

        # replace ± infinite value by na
        xr.apply_ufunc(replaceinf, ds[S_COLUMNS])
        # compute mean of all stage of developpement and all obsevation
        ds_mean = ds.mean(dim="ts_aug", skipna=True)
        # fill na value with computed mean
        ds = ds.fillna(ds_mean)
        # compute mean of all stage of developpement of rice field to complete last na values
        ds_mean = ds_mean.mean(dim="ts_obs", skipna=True)
        # fill na value with computed mean
        ds = ds.fillna(ds_mean)

        return ds

    def transform(self, ds: xr.Dataset, test: bool) -> xr.Dataset:
        """_summary_

        :param ds: Dataset that will be transformed.
        :type ds: xr.Dataset
        :param test: True if it is the Test preprossessing.
        :type test: bool
        :return: Dataset trasformed.
        :rtype: xr.Dataset
        """
        # Process and Merge EY data to Satellite Dataset
        ds = self.add_observation(ds, test)
        # Process and Merge Weather data to Satellite & EY Dataset
        ds = self.add_weather(ds)
        # Compute vegetable indices
        ds = self.compute_vi(ds)
        # Fill missing vegetable indice and replace abnormal values
        ds = self.statedev_fill(ds)

        return ds
