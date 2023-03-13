import os
import math
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


# Make data constants
SIZE = 'adaptative' # 'fixed'
FACTOR = 1 # for 'adaptative' 
NUM_AUGMENT = 10
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
        save_folder = f'data/processed/augment_{NUM_AUGMENT}_{MAX_AUGMENT}'
    elif SIZE == 'fixed':
        degree = str(round(DEGREE, 5)).replace(".", "-")
        save_folder = f'data/processed/fixed_{degree}'
    elif SIZE == 'adaptative':
        save_folder = f'data/processed/adaptative_factor_{FACTOR}'
        
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
    xds = xds.expand_dims({'ts_id': 1})
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
    
    data = get_data(bbox, time_period, bands, scale)

    cloud_mask = ((data.SCL != 0) & 
                  (data.SCL != 1) & 
                  (data.SCL != 3) & 
                  (data.SCL != 6) & 
                  (data.SCL != 8) & 
                  (data.SCL != 9) & 
                  (data.SCL != 10))

    data = data.where(cloud_mask)
    data = process_data(data, row, history_dates)
    
    return data


def save_data_app(index_row, history_days=130, history_dates=24, resolution=10):
    data = save_data(index_row[1], history_days, history_dates, resolution)
    return data

def init_df(df: pd.DataFrame, path: str)->pd.Index:
    list_data = []
    index_count =  pd.Series([NUM_AUGMENT] * df.shape[0], index=df.index)

    if os.path.exists(path=path):
        xdf = xr.open_dataset(path, engine='scipy')
        unique, counts = np.unique(xdf['ts_obs'].values, return_counts=True)
        index_count -= pd.Series(counts, index=unique).sort_index(ascending=True)
        list_data.append(xdf)
    
    index_count = index_count[index_count != 0]
    df = df.loc[index_count.index]
    list_obs = []
    for i in range(len(index_count)):
        list_obs += [df.loc[i]] * index_count[i]

    df = pd.concat(list_obs, axis='columns').T
    df.reset_index(inplace=True)
    df.index.name = 'ts_id'
    
    return df, list_data

def make_data(path, save_folder, augment):
    save_file = f'{save_folder}/{path.split("/")[-1].split(".")[0]}.nc'

    df: pd.DataFrame = pd.read_csv(path)
    df.index.name = 'ts_obs'
    # df.reset_index(inplace=True)
    # df.index.name = 'ts_id'

    df, list_data = init_df(df, save_file)

    print(f'\nRetrieve SAR data from {path.split("/")[-1]}...')
    try:
        with mp.Pool(8) as pool:
            for data in tqdm(pool.imap(save_data_app, df.iterrows()), total=len(df)):
                list_data.append(data)
    except:
        "Error occure during the data retrieval."
    finally:
        data = xr.concat(list_data, dim='ts_id')
        data = data.merge(df.to_xarray())

    print(f'\nSave SAR data from {path.split("/")[-1]}...')
    data.to_netcdf(save_file, engine='scipy')
    print(f'\nSAR data from {path.split("/")[-1]} saved!')

if __name__ == '__main__':
    save_folder = create_folders()

    train_path = 'data/raw/train.csv'
    make_data(train_path, save_folder, augment=NUM_AUGMENT)

    test_path = 'data/raw/test.csv'
    make_data(test_path, save_folder, augment=1)