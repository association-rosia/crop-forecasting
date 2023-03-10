import pystac_client
import planetary_computer as pc
import pandas as pd
from datetime import datetime, timedelta
from odc.stac import stac_load
import xarray as xr
import os
import multiprocessing as mp
import math
import numpy as np
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import tqdm

# from dotenv import load_dotenv
# load_dotenv()
# pc.settings.set_subscription_key(os.getenv('PC_SDK_SUBSCRIPTION_KEY'))

# Make data constants
SIZE = 'adaptative' # 'fixed'
FACTOR = 3
DEGREE = 0.0014589825157734703 # = ha_to_degree(2.622685) # Field size (ha) mean = 2.622685 (train + test)


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


def create_folders()->str:
    # os.makedirs('../../data/processed', exist_ok=True)
    if SIZE == 'fixed':
        degree = str(round(DEGREE, 5)).replace(".", "-")
        save_folder = f'../data/processed/fixed_{degree}'
    elif SIZE == 'adaptative':
        save_folder = f'../data/processed/adaptative_factor_{FACTOR}'
        
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


dict_band_name = {
    'B05': 'rededge1',
    'B06': 'rededge2',
    'B07': 'rededge3',
    'B11': 'swir'
}

def get_bbox(longitude, latitude, field_size):
    if SIZE == 'fixed':
        degree = DEGREE
    elif SIZE == 'adaptative':
        degree = ha_to_degree(field_size) * FACTOR
        
    min_longitude = longitude - degree / 2
    min_latitude = latitude - degree / 2
    max_longitude = longitude + degree / 2
    max_latitude = latitude + degree / 2 
    
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
    # data = data.mean(dim=['latitude', 'longitude'], skipna=True)
    # data = data.to_dataframe()
    # data = data.sort_index(ascending=False).iloc[:history_dates]
    # data.index = data.index.round('D')
    # data.rename(columns=dict_band_name, inplace=True)
    # data.drop(columns=['SCL', 'spatial_ref'], inplace=True)
    # df = pd.DataFrame([row]*history_dates, index=data.index)
    # data = pd.concat([df, data], axis='columns')
    # data.reset_index(inplace=True)

    # xdf = data.copy(deep=True) 
    xds = xds.drop(['spatial_ref', 'SCL'])
    xds = xds.mean(dim=['latitude', 'longitude'], skipna=True)
    xds = xds.sortby('time', ascending=False)
    xds = xds.isel(time=slice(None, history_dates))
    xds['time'] = xds['time'].dt.strftime("%Y-%m-%d")
    xds['state_dev'] =  ('time', np.arange(history_dates)[::-1])
    xds = xds.swap_dims({'time': 'state_dev'})
    xds = xds.rename_vars(dict_band_name)
    df = pd.DataFrame([row]*history_dates, index=xds.indexes['state_dev'])
    xds = xds.merge(df.to_xarray())

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

    cloud_mask = \
        (data.SCL != 0) & \
        (data.SCL != 1) & \
        (data.SCL != 3) & \
        (data.SCL != 6) & \
        (data.SCL != 8) & \
        (data.SCL != 9) & \
        (data.SCL != 10)

    data_filter = data.copy(deep=True).where(cloud_mask)
    data = process_data(data, row, history_dates)
    data_filter = process_data(data_filter, row, history_dates)
    
    return data, data_filter


def save_data_app(index_row, history_days=130, history_dates=24, resolution=10):
    data, data_filter = save_data(index_row[1], history_days, history_dates, resolution)
    return data, data_filter

def make_data(path, save_folder):
    list_data = []
    list_data_filter = []

    with mp.Pool(8) as p:
        df = pd.read_csv(path)
        print(f'\nRetrieve SAR data from {path.split("/")[-1]}...')
        for data, data_filter in tqdm(p.imap(save_data_app, df.iterrows()), total=df.shape[0]):
            list_data.append(data)
            list_data_filter.append(data_filter)
    
    data = xr.concat(list_data, dim='ts_id')
    data_filter = xr.concat(list_data_filter, dim='ts_id')
    # data = pd.concat(list_data, axis='index')
    # data_filter = pd.concat(list_data_filter, axis='index')

    print(f'\nSave SAR data from {path.split("/")[-1]}...')
    data.to_netcdf(f'{save_folder}/{path.split("/")[-1].split(".")[0]}.nc', engine='scipy')
    data_filter.to_netcdf(f'{save_folder}/{path.split("/")[-1].split(".")[0]}_filter.nc', engine='scipy')
    print(f'\nSAR data from {path.split("/")[-1]} saved!')

if __name__ == '__main__':
    # pandarallel.initialize(progress_bar=True, nb_workers=16)
    save_folder = create_folders()

    train_path = '../data/raw/train.csv'
    make_data(train_path, save_folder)

    test_path = '../data/raw/test.csv'
    make_data(test_path, save_folder)