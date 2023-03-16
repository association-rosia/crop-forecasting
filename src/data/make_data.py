import os
import math
import time
from tqdm import tqdm
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import pystac_client
import planetary_computer as pc
from odc.stac import stac_load

import multiprocessing as mp
from random import uniform, random

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from utils import ROOT_DIR
from os.path import join


# Make data constants
SIZE = 'adaptative' # 'fixed'
FACTOR = 1 # for 'adaptative' 
NUM_AUGMENT = 50
MAX_AUGMENT = 5
DEGREE = 0.0014589825157734703 # = ha_to_degree(2.622685) # Field size (ha) mean = 2.622685 (train + test)

dict_band_name = {
    'B05': 'rededge1',
    'B06': 'rededge2',
    'B07': 'rededge3',
    'B11': 'swir'
}


def ha_to_degree(field_size): # Field_size (ha)
    ''' 
    1° ~= 111km
    1ha = 0.01km2
    then, side_size = sqrt(0.01 * field_size) (km)
    so, degree = side_size / 111 (°)
    '''
    side_size = math.sqrt(0.01 * field_size)
    degree = side_size / 111
    return degree


def create_folders() -> str:
    if NUM_AUGMENT > 1:
        save_folder = join(ROOT_DIR, 'data', 'processed', f'augment_{NUM_AUGMENT}_{MAX_AUGMENT}') 
    elif SIZE == 'fixed':
        degree = str(round(DEGREE, 5)).replace(".", "-")
        save_folder = join(ROOT_DIR, 'data', 'processed', f'fixed_{degree}') 
    elif SIZE == 'adaptative':
        save_folder = join(ROOT_DIR, 'data', 'processed', f'adaptative_factor_{FACTOR}') 
        
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def get_factors():
    factors = []
    for _ in range(4):
        factor = uniform(1, MAX_AUGMENT)
        if random() < 0.5: factor = 1 / factor
        factors.append(factor)

    return factors


def get_bbox(longitude, latitude, field_size):
    if SIZE == 'fixed':
        degree = DEGREE
    elif SIZE == 'adaptative':
        degree = ha_to_degree(field_size) * FACTOR
        
    length = degree / 2
    factors = get_factors()
    min_longitude = longitude - factors[0] * length
    min_latitude = latitude - factors[1] * length
    max_longitude = longitude + factors[2] * length
    max_latitude = latitude + factors[3] * length
    
    return (min_longitude, min_latitude, max_longitude, max_latitude)


def get_time_period(havest_date: str, history_days: int)->str:
    havest_datetime = datetime.strptime(havest_date, '%d-%m-%Y')
    sowing_datetime = havest_datetime - timedelta(days=history_days)
    return f'{sowing_datetime.strftime("%Y-%m-%d")}/{havest_datetime.strftime("%Y-%m-%d")}'


def get_data(bbox, time_period: str, bands: list[str], scale: float):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_period)
    items = search.item_collection()
    data = stac_load(items, bands=bands, crs="EPSG:4326", resolution=scale, bbox=bbox)
    return data


def process_data(xds: xr.Dataset, row: pd.Series, history_dates:int)->xr.Dataset:
    xds = xds.drop(['spatial_ref', 'SCL'])
    xds = xds.mean(dim=['latitude', 'longitude'], skipna=True)
    xds = xds.sortby('time', ascending=False)
    xds = xds.isel(time=slice(None, history_dates))
    xds['time'] = xds['time'].dt.strftime("%Y-%m-%d")
    xds['state_dev'] =  ('time', np.arange(history_dates)[::-1])
    xds = xds.swap_dims({'time': 'state_dev'})
    xds = xds.rename_vars(dict_band_name)
    return xds


def save_data(row, history_days, history_dates, resolution):
    scale = resolution / 111320.0
    bands = ['red', 'green', 'blue', 'B05', 'B06', 'B07', 'nir', 'B11', 'SCL']
    
    longitude = row['Longitude']
    latitude = row['Latitude']
    field_size = float(row['Field size (ha)'])
    bbox = get_bbox(longitude, latitude, field_size)

    havest_date = row['Date of Harvest']
    time_period = get_time_period(havest_date, history_days)
    
    xds = get_data(bbox, time_period, bands, scale)

    cloud_mask = ((xds.SCL != 0) & 
                  (xds.SCL != 1) & 
                  (xds.SCL != 3) & 
                  (xds.SCL != 6) & 
                  (xds.SCL != 8) & 
                  (xds.SCL != 9) & 
                  (xds.SCL != 10))

    xds = xds.where(cloud_mask)
    xds = process_data(xds, row, history_dates)
    
    return xds


def save_data_app(index_row, history_days=130, history_dates=24, resolution=10) -> xr.Dataset:
    list_xds = []
    
    for i in range(NUM_AUGMENT):
        xds = save_data(index_row[1], history_days, history_dates, resolution)
        xds = xds.expand_dims({'ts_aug': [i]})
        list_xds.append(xds)
    xds: xr.Dataset = xr.concat(list_xds, dim='ts_aug')
    xds = xds.expand_dims({'ts_obs': [index_row[0]]})
    
    return xds


def init_df(df: pd.DataFrame, path: str) -> tuple[pd.DataFrame, list]:
    list_data = []
    df.index.name = 'ts_obs'

    if os.path.exists(path=path):
        xdf = xr.open_dataset(path, engine='scipy')
        unique = np.unique(xdf['ts_obs'].values)
        list_data.append(xdf)
        df = df.loc[~df.index.isin(unique)]
    
    return df, list_data


def make_data(path, save_folder):
    checkpoint = False
    start = time.time()
    save_file = f'{save_folder}/{path.split("/")[-1].split(".")[0]}.nc'

    df: pd.DataFrame = pd.read_csv(path)
    df, list_data = init_df(df, save_file)

    print(f'\nRetrieve SAR data from {path.split("/")[-1]}...')
    try:
        with mp.Pool(4) as pool:
            for xds in tqdm(pool.imap(save_data_app, df.iterrows()), total=len(df)):
                list_data.append(xds)
                
                if time.time() - start > 3600:
                    checkpoint = True
                    raise Exception('Checkpoint.')
    except:
        "Error occurs during the data retrieval..."
    
    data = xr.concat(list_data, dim='ts_obs')
    print(f'\nSave SAR data from {path.split("/")[-1]}...')
    data.to_netcdf(save_file, engine='scipy')
    print(f'\nSAR data from {path.split("/")[-1]} saved!')
    
    if checkpoint:
        make_data(path, save_folder)
    

if __name__ == '__main__':
    save_folder = create_folders()

    train_path = join(ROOT_DIR, 'data', 'raw', 'train.csv')
    make_data(train_path, save_folder)

    test_path = join(ROOT_DIR, 'data', 'raw', 'test.csv')
    make_data(test_path, save_folder)