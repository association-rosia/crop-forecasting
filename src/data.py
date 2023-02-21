import pystac_client
import planetary_computer as pc
import pandas as pd
from datetime import datetime, timedelta
from odc.stac import stac_load
import numpy as np
from PIL import Image
import os
from glob import glob
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


def save_data(row, df, path, files, history=120, resolution=20, surrounding_box=0.1, nb_images=20):
    longitude = row['Longitude']
    latitude = row['Latitude']
    nb_rows = len(df[(df['Longitude'] == longitude) & (df['Latitude'] == latitude)])
    folder = f'{longitude}_{latitude}'.replace('.', '-')
    os.makedirs(f'{path}/{folder}', exist_ok=True)
    nb_files = len([f for f in files if folder in f])

    try:
        if nb_files != 4 * nb_rows * nb_images:
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

            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                                modifier=pc.sign_inplace)
            search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_period)
            items = search.item_collection()
            data = stac_load(items, bands=bands, crs="EPSG:4326", resolution=scale, bbox=bbox)

            for i in range(1, nb_images + 1):
                time = data.time[-i].values
                date = np.datetime_as_string(time, unit='D')

                # Save the PNG image
                file_name = f'{path}/{folder}/{date}.png'
                if not os.path.isfile(file_name):
                    image = bands_to_image(data, i)
                    image.save(file_name)

                # Save the NumPy files
                for band in ['red', 'nir', 'SCL']:
                    file_name = f'{path}/{folder}/{date}_{band}.npy'
                    if not os.path.isfile(file_name):
                        array = data[band][-i].to_numpy()
                        np.save(file_name, array)

    except Exception as e:
        print(f'{folder} - {e}')


catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
root = '/content/drive/MyDrive/PROJECTS/crop-forecasting/data'
train_path = f'{root}/train.csv'
test_path = f'{root}/test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

os.makedirs(f'{root}', exist_ok=True)
pandarallel.initialize(progress_bar=True, nb_workers=3)

# Save train data
train_path = f'{root}/raw/train'
os.makedirs(train_path, exist_ok=True)
train_files = [y for x in os.walk(train_path) for y in glob(os.path.join(x[0], '*.*'))]
train_df.parallel_apply(lambda row: save_data(row, train_df, train_path, train_files), axis=1)

# Save test data
test_path = f'{root}/raw/test'
os.makedirs(test_path, exist_ok=True)
test_files = [y for x in os.walk(test_path) for y in glob(os.path.join(x[0], '*.*'))]
test_df.parallel_apply(lambda row: save_data(row, test_df, test_path, test_files), axis=1)
