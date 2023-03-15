import glob
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import savgol_filter

from datascaler import DatasetScaler

from constants import FOLDER, S_COLUMNS, G_COLUMNS, M_COLUMNS, TARGET

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from utils import ROOT_DIR
from os.path import join


def add_observation(xdf: xr.Dataset, test: bool) -> xr.Dataset:
    if test:
        path = join(ROOT_DIR, 'data', 'raw', 'test.csv')
    else:
        path = join(ROOT_DIR, 'data', 'raw', 'train.csv')

    df = pd.read_csv(path)
    df.index.name = 'ts_obs'
    xdf = xr.merge([xdf, df.to_xarray()])
    return xdf


def add_weather(xdf: xr.Dataset) -> xr.Dataset:
    xdf = xdf

    weather = []
    for path in glob.glob('data/raw/weather/*.csv'):
        weather.append(pd.read_csv(path))

    df_weather = pd.concat(weather, axis='index')
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather['name'] = df_weather['name'].str.replace(' ', '_')
    df_weather.set_index(['datetime', 'name'], inplace=True)
    xdf_weather = df_weather.to_xarray().set_coords(['datetime', 'name'])
    xdf_weather['datetime'] = xdf_weather['datetime'].dt.strftime('%Y-%m-%d')

    xdf = xr.merge([xdf, xdf_weather])

    return xdf


def compute_vi(xdf: xr.Dataset) -> xr.Dataset:

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
    xdf['ndvi'] = compute_ndvi(xdf)
    xdf['savi'] = compute_savi(xdf)
    xdf['evi'] = compute_evi(xdf)
    xdf['rep'] = compute_rep(xdf)
    xdf['osavi'] = compute_osavi(xdf)
    xdf['rdvi'] = compute_rdvi(xdf)
    xdf['mtvi1'] = compute_mtvi1(xdf)
    xdf['lswi'] = compute_lswi(xdf)

    return xdf


def statedev_fill(xdf: xr.Dataset) -> xr.Dataset:
    
    def replaceinf(arr:  np.ndarray) -> np.ndarray:
        if np.issubdtype(arr.dtype, np.number):
            arr[np.isinf(arr)] = np.nan
        return arr
    
    # replace infinite value by na
    xr.apply_ufunc(replaceinf, xdf[S_COLUMNS])
    # compute mean of all stage of developpement and all obsevation
    xdf_mean = xdf.mean(dim='ts_aug', skipna=True)
    # fill na value with computed mean
    xdf = xdf.fillna(xdf_mean)
    # compute mean of all stage of developpement of rice field to complete last na values
    xdf_mean = xdf_mean.mean(dim='ts_obs', skipna=True)
    # fill na value with computed mean
    xdf = xdf.fillna(xdf_mean)

    return xdf


def smooth(xdf: xr.Dataset) -> xr.Dataset:
    # apply savgol_filter to vegetable indice
    xdf_s = xr.apply_ufunc(savgol_filter, xdf[S_COLUMNS], kwargs={'axis': 2, 'window_length': 12, 'polyorder': 4, 'mode': 'mirror'})
    # merge both dataset and override old vegetable indice and bands
    return xr.merge([xdf_s, xdf], compat='override')


def categorical_encoding(xdf: xr.Dataset) -> xr.Dataset:
    xdf['Rice Crop Intensity(D=Double, T=Triple)'] = xdf['Rice Crop Intensity(D=Double, T=Triple)'].str.replace("D", "2").str.replace("T", "3").astype(np.int8)
    return xdf


def features_modification(xdf: xr.Dataset, test: bool) -> xr.Dataset:
    xdf['sunrise'] = xdf['sunrise'].astype(np.datetime64)
    xdf['sunset'] = xdf['sunset'].astype(np.datetime64)

    xdf['solarexposure'] = (xdf['sunset'] - xdf['sunrise']).dt.seconds

    xdf['time'] = xdf['time'].astype(np.datetime64)
    xdf['datetime'] = xdf['datetime'].astype(np.datetime64)
    xdf = xdf.reset_coords('time')

    # time and District are keys to link with weather data 
    columns = S_COLUMNS + G_COLUMNS + M_COLUMNS + ['time', 'District']
    if not test:
        columns.append(TARGET)
    xdf = xdf[columns]

    return xdf


def scale_data(xdf: xr.Dataset, path: str, test: bool) -> xr.Dataset:
    # Path for saving scaler
    path = '/'.join(path.split('/')[:-1]) + "/scaler_dataset.joblib"
    
    if not test:
        scaler = DatasetScaler()
        xdf = scaler.fit_transform(xdf)
        joblib.dump(scaler, path)
    else:
        scaler: DatasetScaler = joblib.load(path)
        xdf = scaler.transform(xdf)
    
    return xdf


def create_id(xdf: xr.Dataset) -> xr.Dataset:
    ts_id = np.arange(xdf.dims['ts_obs'] * xdf.dims['ts_aug'])
    ts_id = ts_id.reshape((xdf.dims['ts_obs'], xdf.dims['ts_aug']))
    xdf = xdf.assign_coords({'ts_id': (('ts_obs', 'ts_aug'), ts_id)})
    return xdf


def process_data(path: str, test: bool=False):
    xdf = xr.open_dataset(path)
    # Add observation to the dataset
    xdf = add_observation(xdf, test)
    # Add weather to the dataset
    xdf = add_weather(xdf)
    # Compute vegetable indice
    xdf = compute_vi(xdf)
    # Fill na values
    xdf = statedev_fill(xdf)
    # Smooth variable
    xdf = smooth(xdf)
    # Create new features
    xdf = features_modification(xdf, test)
    # Encode categorical features
    xdf = categorical_encoding(xdf)
    # Scale data
    xdf = scale_data(xdf, path, test)
    # Add an id for each line
    xdf = create_id(xdf)
    # Save data
    path = '.'.join(path.split('.')[:-1]) + "_processed." + path.split('.')[-1]
    xdf.to_netcdf(path, engine='scipy')

if __name__ == '__main__':

    # Cloud filtered data
    train_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'train.nc')
    process_data(train_path)
    test_path = join(ROOT_DIR, 'data', 'processed', FOLDER, 'test.nc')
    process_data(test_path, test=True)