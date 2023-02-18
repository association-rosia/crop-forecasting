import pystac_client
import planetary_computer as pc
import pandas as pd
from datetime import datetime, timedelta
from odc.stac import stac_load
import numpy as np
from PIL import Image
import os
from pandarallel import pandarallel

from dotenv import load_dotenv
load_dotenv()
pc.settings.set_subscription_key(os.getenv('PC_SDK_SUBSCRIPTION_KEY'))


def bands_to_image(data, index, img_size=512):
    red = data['red'][index].to_numpy()
    green = data['green'][index].to_numpy()
    blue = data['blue'][index].to_numpy()
    array = np.array([red, green, blue])
    array = np.transpose(array, axes=[1, 2, 0])
    array = np.clip(array, 0, 3000)
    array = (array / 3000 * 255).astype(np.uint8)
    image = Image.fromarray(array)
    image = image.resize((img_size, img_size))
    return image


def save_data(row, path, history=120, resolution=20, surrounding_box=0.1, num_images=20):
    longitude = row['Longitude']
    latitude = row['Latitude']
    min_longitude = longitude - surrounding_box / 2
    min_latitude = latitude - surrounding_box / 2
    max_longitude = longitude + surrounding_box / 2
    max_latitude = latitude + surrounding_box / 2
    bbox = (min_longitude, min_latitude, max_longitude, max_latitude)
    scale = resolution / 111320.0
    bands = ['red', 'green', 'blue', 'nir', 'SCL']

    havest_date = row['Date of Harvest']
    havest_datetime = datetime.strptime(havest_date, '%d-%m-%Y')
    sowing_datetime = havest_datetime - timedelta(days=history)
    time_period = f'{sowing_datetime.strftime("%Y-%m-%d")}/{havest_datetime.strftime("%Y-%m-%d")}'

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_period)
    items = search.item_collection()
    # print(f'>>> {len(items)} to load...')
    data = stac_load(items, bands=bands, crs="EPSG:4326", resolution=scale, bbox=bbox)
    # print(f'>>> {len(items)} loaded!')

    for i in range(1, num_images + 1):
        time = data.time[-i].values
        date = np.datetime_as_string(time, unit='D')
        file_name = f'{longitude}_{latitude}_{date}'.replace('.', '-')
        name = f'{path}/{file_name}.png'

        # Save the PNG image
        if not os.path.isfile(name):
            image = bands_to_image(data, i)
            image.save(name)
            # print(f'>>> {name} saved!')
        # else:
            # print(f'--- {name} already saved!')

        # Save the NumPy files
        for band in ['red', 'nir', 'SCL']:
            name = f'{path}/{file_name}_{band}.npy'

            if not os.path.isfile(name):
                array = data[band][-i].to_numpy()
                np.save(name, array)
                # print(f'>>> {name} saved!')
            # else:
                # print(f'--- {name} already saved!')


catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
train_path = '../data/train.csv'
test_path = '../data/test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

os.makedirs('../data/raw', exist_ok=True)
pandarallel.initialize(progress_bar=True)

# Save train data
train_path = '../data/raw/train'
os.makedirs(train_path, exist_ok=True)
train_df.parallel_apply(lambda row: save_data(row, train_path), axis=1)

# Save test data
test_path = '../data/raw/test'
os.makedirs(test_path, exist_ok=True)
test_df.parallel_apply(lambda row: save_data(row, test_path), axis=1)
