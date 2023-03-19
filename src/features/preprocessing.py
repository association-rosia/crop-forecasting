import glob
import joblib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import savgol_filter

from datascaler import DatasetScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os, sys

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from tqdm import tqdm

from src.constants import FOLDER, S_COLUMNS, G_COLUMNS, M_COLUMNS, TARGET, TARGET_TEST

from utils import ROOT_DIR
from os.path import join


def features_modification(self, xdf: xr.Dataset, target: str) -> xr.Dataset:
        def find_columns(real_list: list[str], del_list: list[str]):
            list_col = []
            for real_col in real_list:
                for del_col in del_list:
                    if del_col in real_col:
                        list_col.append(real_col)
            return list_col



        xdf["time"] = xdf["time"].astype(np.datetime64)
        xdf["datetime"] = xdf["datetime"].astype(np.datetime64)
        xdf = xdf.reset_coords("time")

        # time and District are keys to link with weather data
        columns = S_COLUMNS + G_COLUMNS + M_COLUMNS + ["time", "District", target]
        xdf = xdf[columns]

        return xdf


class Convertor:
    def __init__(self, agg: bool = None, observation: bool = True, weather: bool = True, vi: bool = True,) -> None:
        pass


class Smoothor:
    def __init__(self, mode: str = 'savgol') -> None:
        self.mode = mode

    def smooth_savgol(self, xdf: xr.Dataset) -> xr.Dataset:
        # apply savgol_filter to vegetable indice
        xdf_s = xr.apply_ufunc(
            savgol_filter,
            xdf[S_COLUMNS],
            kwargs={"axis": 2, "window_length": 12, "polyorder": 4, "mode": "mirror"},
        )
        # merge both dataset and override old vegetable indice and bands
        return xr.merge([xdf_s, xdf], compat="override")
    
    def fit(self, X: xr.Dataset, y=None)->xr.Dataset:
        return X
    
    def transform(self, X: xr.Dataset, y=None)->xr.Dataset:
        if self.mode == 'savgol':
            X = self.smooth_savgol(X)
        
        return X
    
    def fit_transform(self, X: xr.Dataset, y=None)->xr.Dataset:
        X = self.fit(X)
        X = self.transform(X)


class Concatenator:
    def __init__(self) -> None:
        pass

    def add_observation(self, xdf: xr.Dataset, test: bool) -> xr.Dataset:
        def categorical_encoding(xdf: xr.Dataset) -> xr.Dataset:
            xdf["Rice Crop Intensity(D=Double, T=Triple)"] = (
                xdf["Rice Crop Intensity(D=Double, T=Triple)"]
                .str.replace("D", "2")
                .str.replace("T", "3")
                .astype(np.int8)
            )
            return xdf
        
        if test:
            file_name = 'test.csv'
        else:
            file_name = 'train.csv'
        
        path = join(ROOT_DIR, "data", "raw", file_name)

        df = pd.read_csv(path)
        df.index.name = "ts_obs"
        xdf = xr.merge([xdf, df.to_xarray()])
        xdf = categorical_encoding(xdf)

        return xdf


    def add_weather(self, xdf: xr.Dataset) -> xr.Dataset:
        def features_modification(xdf: xr.Dataset) -> xr.Dataset:
            xdf["sunrise"] = xdf["sunrise"].astype(np.datetime64)
            xdf["sunset"] = xdf["sunset"].astype(np.datetime64)

            xdf["solarexposure"] = (xdf["sunset"] - xdf["sunrise"]).dt.seconds
            return xdf

        weather = []
        for path in glob.glob(join(ROOT_DIR, "data", "external", 'weather', "*.csv")):
            weather.append(pd.read_csv(path))

        df_weather = pd.concat(weather, axis="index")
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        df_weather["name"] = df_weather["name"].str.replace(" ", "_")
        df_weather.set_index(["datetime", "name"], inplace=True)
        xdf_weather = df_weather.to_xarray().set_coords(["datetime", "name"])
        xdf_weather["datetime"] = xdf_weather["datetime"].dt.strftime("%Y-%m-%d")
        xdf_weather = features_modification(xdf_weather)
        xdf = xr.merge([xdf, xdf_weather])

        return xdf


    def compute_vi(self, xdf: xr.Dataset) -> xr.Dataset:
        def compute_ndvi(xdf: xr.Dataset) -> xr.Dataset:
            return (xdf.nir - xdf.red) / (xdf.nir + xdf.red)

        def compute_savi(xdf, L=0.5) -> xr.Dataset:
            return 1 + L * (xdf.nir - xdf.red) / (xdf.nir + xdf.red + L)

        def compute_evi(xdf, G=2.5, L=1, C1=6, C2=7.5) -> xr.Dataset:
            return G * (xdf.nir - xdf.red) / (xdf.nir + C1 * xdf.red - C2 * xdf.blue + L)

        def compute_rep(xdf: xr.Dataset) -> xr.Dataset:
            rededge = (xdf.red + xdf.rededge3) / 2
            return 704 + 35 * (rededge - xdf.rededge1) / (xdf.rededge2 - xdf.rededge1)

        def compute_osavi(xdf: xr.Dataset) -> xr.Dataset:
            return (xdf.nir - xdf.red) / (xdf.nir + xdf.red + 0.16)

        def compute_rdvi(xdf: xr.Dataset) -> xr.Dataset:
            return (xdf.nir - xdf.red) / np.sqrt(xdf.nir + xdf.red)

        def compute_mtvi1(xdf: xr.Dataset) -> xr.Dataset:
            return 1.2 * (1.2 * (xdf.nir - xdf.green) - 2.5 * (xdf.red - xdf.green))

        def compute_lswi(xdf: xr.Dataset) -> xr.Dataset:
            return (xdf.nir - xdf.swir) / (xdf.nir + xdf.swir)

        # compute all vegetable indice
        xdf["ndvi"] = compute_ndvi(xdf)
        xdf["savi"] = compute_savi(xdf)
        xdf["evi"] = compute_evi(xdf)
        xdf["rep"] = compute_rep(xdf)
        xdf["osavi"] = compute_osavi(xdf)
        xdf["rdvi"] = compute_rdvi(xdf)
        xdf["mtvi1"] = compute_mtvi1(xdf)
        xdf["lswi"] = compute_lswi(xdf)

        return xdf


    def statedev_fill(self, xdf: xr.Dataset) -> xr.Dataset:
        def replaceinf(arr: np.ndarray) -> np.ndarray:
            if np.issubdtype(arr.dtype, np.number):
                arr[np.isinf(arr)] = np.nan
            return arr

        # replace infinite value by na
        xr.apply_ufunc(replaceinf, xdf[S_COLUMNS])
        # compute mean of all stage of developpement and all obsevation
        xdf_mean = xdf.mean(dim="ts_aug", skipna=True)
        # fill na value with computed mean
        xdf = xdf.fillna(xdf_mean)
        # compute mean of all stage of developpement of rice field to complete last na values
        xdf_mean = xdf_mean.mean(dim="ts_obs", skipna=True)
        # fill na value with computed mean
        xdf = xdf.fillna(xdf_mean)

        return xdf


    def fit(self, X: xr.Dataset, test: bool = None, y = None)->xr.Dataset:
        return X
        

    def transform(self, X: xr.Dataset, test: bool, y = None)->xr.Dataset:
        X = self.add_observation(X, test)
        X = self.add_weather(X)
        X = self.compute_vi(X)
        X = self.statedev_fill(X)

        return X
    
    
    def fit_transform(self, X: xr.Dataset, test: bool, y = None)->xr.Dataset:
        X = self.fit(X)
        X = self.transform(X, test)
        return X
