import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import savgol_filter

import os, sys

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from src.constants import S_COLUMNS, G_COLUMNS, M_COLUMNS

from utils import ROOT_DIR
from os.path import join


from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator


class Convertor(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, agg: bool = None, weather: bool = True, vi: bool = True) -> None:
        self.agg = agg
        self.weather = weather
        self.vi = vi

    def to_dataframe(self, X:xr.Dataset)->pd.DataFrame:
        col = 'agg' if self.agg else 'state_dev'
        df = X.to_dataframe()
        df.set_index(G_COLUMNS, append=True, inplace=True)
        df.reset_index(col, inplace=True)
        df[col] = df[col].astype(str)
        df = df.pivot(columns=col)
        df.columns = df.columns.map('_'.join).str.strip('_')
        df.reset_index(G_COLUMNS, inplace=True)
        df = df.reorder_levels(['ts_obs', 'ts_aug']).sort_index()
        return df

    def merge_dimensions(self, X: xr.Dataset)->xr.Dataset:
        X = xr.merge([X[G_COLUMNS], X[M_COLUMNS].sel(datetime=X['time'], name=X['District']), X[S_COLUMNS]])
        X = X.drop(['name', 'datetime', 'time'])
        return X

    def compute_agg(self, X:xr.Dataset)->xr.Dataset:
        X = xr.concat([X.mean(dim='state_dev'), X.max(dim='state_dev'), X.min(dim='state_dev')], dim='agg')
        X['agg'] = ['mean', 'max', 'min'] 
        return X

    def fit(self, X: xr.Dataset, y=None)->xr.Dataset:
        return self
    
    def transform(self, X: xr.Dataset)->xr.Dataset:
        X = self.merge_dimensions(X)
        if self.agg:
            X = self.compute_agg(X)
        if not self.weather:
            X = X.drop(M_COLUMNS)
        if not self.vi:
            X = X.drop(S_COLUMNS)
        X = self.to_dataframe(X)
        return X
    

class Smoothor(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(self, mode: str = 'savgol') -> None:
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
    
    def fit(self, X: xr.Dataset, y=None)->xr.Dataset:
        return self
    
    def transform(self, X: xr.Dataset)->xr.Dataset:
        if self.mode == 'savgol':
            X = self.smooth_savgol(X)
        
        return X


class Concatenator:
    def __init__(self) -> None:
        pass

    def add_observation(self, ds: xr.Dataset, test: bool) -> xr.Dataset:
        def categorical_encoding(ds: xr.Dataset) -> xr.Dataset:
            ds["Rice Crop Intensity(D=Double, T=Triple)"] = (
                ds["Rice Crop Intensity(D=Double, T=Triple)"]
                .str.replace("D", "2")
                .str.replace("T", "3")
                .astype(np.int8)
            )
            return ds
        
        if test:
            file_name = 'test.csv'
        else:
            file_name = 'train.csv'
        
        path = join(ROOT_DIR, "data", "raw", file_name)

        df = pd.read_csv(path)
        df.index.name = "ts_obs"
        ds = xr.merge([ds, df.to_xarray()])
        ds = categorical_encoding(ds)

        return ds

    def add_weather(self, ds: xr.Dataset) -> xr.Dataset:
        def features_modification(ds: xr.Dataset) -> xr.Dataset:
            ds["sunrise"] = ds["sunrise"].astype(np.datetime64)
            ds["sunset"] = ds["sunset"].astype(np.datetime64)

            ds["solarexposure"] = (ds["sunset"] - ds["sunrise"]).dt.seconds
            return ds

        weather = []
        for path in glob.glob(join(ROOT_DIR, "data", "external", 'weather', "*.csv")):
            weather.append(pd.read_csv(path))

        df_weather = pd.concat(weather, axis="index")
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        df_weather["name"] = df_weather["name"].str.replace(" ", "_")
        df_weather.set_index(["datetime", "name"], inplace=True)
        ds_weather = df_weather.to_xarray().set_coords(["datetime", "name"])
        ds_weather["datetime"] = ds_weather["datetime"].dt.strftime("%Y-%m-%d")
        ds_weather = features_modification(ds_weather)
        ds = xr.merge([ds, ds_weather])

        return ds

    def compute_vi(self, ds: xr.Dataset) -> xr.Dataset:
        def compute_ndvi(ds: xr.Dataset) -> xr.Dataset:
            return (ds.nir - ds.red) / (ds.nir + ds.red)

        def compute_savi(ds, L=0.5) -> xr.Dataset:
            return 1 + L * (ds.nir - ds.red) / (ds.nir + ds.red + L)

        def compute_evi(ds, G=2.5, L=1, C1=6, C2=7.5) -> xr.Dataset:
            return G * (ds.nir - ds.red) / (ds.nir + C1 * ds.red - C2 * ds.blue + L)

        def compute_rep(ds: xr.Dataset) -> xr.Dataset:
            rededge = (ds.red + ds.rededge3) / 2
            return 704 + 35 * (rededge - ds.rededge1) / (ds.rededge2 - ds.rededge1)

        def compute_osavi(ds: xr.Dataset) -> xr.Dataset:
            return (ds.nir - ds.red) / (ds.nir + ds.red + 0.16)

        def compute_rdvi(ds: xr.Dataset) -> xr.Dataset:
            return (ds.nir - ds.red) / np.sqrt(ds.nir + ds.red)

        def compute_mtvi1(ds: xr.Dataset) -> xr.Dataset:
            return 1.2 * (1.2 * (ds.nir - ds.green) - 2.5 * (ds.red - ds.green))

        def compute_lswi(ds: xr.Dataset) -> xr.Dataset:
            return (ds.nir - ds.swir) / (ds.nir + ds.swir)

        # compute all vegetable indice
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
        def replaceinf(arr: np.ndarray) -> np.ndarray:
            if np.issubdtype(arr.dtype, np.number):
                arr[np.isinf(arr)] = np.nan
            return arr

        # replace infinite value by na
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
        
    def transform(self, ds: xr.Dataset, test: bool)->xr.Dataset:
        ds = self.add_observation(ds, test)
        ds = self.add_weather(ds)
        ds = self.compute_vi(ds)
        ds = self.statedev_fill(ds)

        return ds
