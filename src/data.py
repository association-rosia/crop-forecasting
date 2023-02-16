import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandarallel import pandarallel
import numpy as np
from datetime import datetime, timedelta
import os
import pystac_client
from odc.stac import stac_load
import planetary_computer as pc
from dotenv import load_dotenv

load_dotenv()
pc.settings.set_subscription_key(os.getenv('SUBSCRIPTION_KEY'))

train_path = '../data/train.csv'
test_path = '../data/test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


def save_data(row, path, history=120, resolution=10, surrounding_box=0.1, num_images=20):
    longitude = row['Longitude']
    latitude = row['Latitude']
    min_longitude = longitude - surrounding_box / 2
    min_latitude = latitude - surrounding_box / 2
    max_longitude = longitude + surrounding_box / 2
    max_latitude = latitude + surrounding_box / 2
    bbox = (min_longitude, min_latitude, max_longitude, max_latitude)

    havest_date = row['Date of Harvest']
    havest_datetime = datetime.strptime(havest_date, '%d-%m-%Y')
    sowing_datetime = havest_datetime - timedelta(days=history)
    time_period = f'{sowing_datetime.strftime("%Y-%m-%d")}/{havest_datetime.strftime("%Y-%m-%d")}'
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_period)
    # items = list(search.get_all_items())
    items = list(search.item_collection())
    scale = resolution / 111320.0
    bands = ['red', 'green', 'blue', 'nir', 'rededge', 'B05', 'B06', 'B07', 'SCL']

    data = stac_load(
        items,
        bands=bands,
        crs="EPSG:4326",
        resolution=scale,
        chunks={"x": 2048, "y": 2048},
        dtype="uint16",
        patch_url=pc.sign,
        bbox=bbox
    )

    for i in range(1, num_images + 1):
        time = data.time[-i].values
        date = np.datetime_as_string(time, unit='D')

        for band in bands:
            file_name = f'{longitude}_{latitude}_{date}_{band}'.replace('.', '-')
            file_name = f'{path}/{file_name}.npy'

            if not os.path.isfile(file_name):
                array = data[band][-i].to_numpy()
                np.save(file_name, array)
                print(f'>>> {file_name} saved!')


os.makedirs('../data/raw', exist_ok=True)
pandarallel.initialize()

# Save train data 
train_path = '../data/raw/train'
os.makedirs(train_path, exist_ok=True)
train_df.parallel_apply(lambda row: save_data(row, train_path), axis=1)

# Save test data 
test_path = '../data/raw/test'
os.makedirs(test_path, exist_ok=True)
test_df.parallel_apply(lambda row: save_data(row, test_path), axis=1)
