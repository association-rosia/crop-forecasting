import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timedelta
from random import random, uniform

import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import xarray as xr
from odc.stac import stac_load
from tqdm import tqdm

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from os.path import join

from utils import ROOT_DIR

# Make data constants
SIZE = "adaptative"  # 'fixed'
FACTOR = 1  # for 'adaptative'
NUM_AUGMENT = 40
MAX_AUGMENT = 5
DEGREE = 0.0014589825157734703  # = ha_to_degree(2.622685) # Field size (ha) mean = 2.622685 (train + test)


def ha_to_degree(field_size: float) -> float:  # Field_size (ha)
    """ Convert field size (ha) to degree.

    :param field_size: field size (ha)
    :type field_size: float
    :return: field width/length (degree)
    :rtype: float
    """

    # 1° ~= 111km
    # 1ha = 0.01km2
    # then, side_size = sqrt(0.01 * field_size) (km)
    # so, degree = side_size / 111 (°)
    side_size = math.sqrt(0.01 * field_size)
    degree = side_size / 111
    return degree


def create_folders() -> str:
    """ Create folders in function of the extraction type.

    :return: name of the folder created
    :rtype: str
    """
    save_folder = None

    if NUM_AUGMENT > 1:
        save_folder = join(
            ROOT_DIR,
            "data",
            "external",
            "satellite",
            f"augment_{NUM_AUGMENT}_{MAX_AUGMENT}",
        )
    elif SIZE == "fixed":
        degree = str(round(DEGREE, 5)).replace(".", "-")
        save_folder = join(ROOT_DIR, "data", "external", "satellite", f"fixed_{degree}")
    elif SIZE == "adaptative":
        save_folder = join(
            ROOT_DIR, "data", "external", "satellite", f"adaptative_factor_{FACTOR}"
        )

    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def get_factors() -> list[float]:
    """ Randomly draw factors to create augmented windows
        to retrieve different satellite images.

    :return: four random factors (between 1/MAX_AUGMENT and MAX_AUGMENT)
    :rtype: list[float]
    """
    factors = []
    for _ in range(4):
        factor = uniform(1, MAX_AUGMENT)
        if random() < 0.5:
            factor = 1 / factor
        factors.append(factor)

    return factors


def get_bbox(longitude: float, latitude: float, field_size: float) -> tuple[float, float, float, float]:
    """ Get the bounding box of the satellite image
        using augmented window factors.

    :param longitude: longitude of the satellite image
    :type longitude: float
    :param latitude: latitude of the satellite image
    :type latitude: float
    :param field_size: field size (ha)
    :type field_size: float
    :return: max and min longitude, min and max latitude
    :rtype: tuple[float, float, float, float]
    """

    if SIZE == "fixed":
        degree = DEGREE
    else:
        degree = ha_to_degree(field_size) * FACTOR

    length = degree / 2
    factors = get_factors()
    min_longitude = longitude - factors[0] * length
    min_latitude = latitude - factors[1] * length
    max_longitude = longitude + factors[2] * length
    max_latitude = latitude + factors[3] * length

    return min_longitude, min_latitude, max_longitude, max_latitude


def get_time_period(harvest_date: str, history_days: int) -> str:
    """ Get the time period using the harvest date
        and the number history days defined.

    :param harvest_date: Date of the harvest
    :type harvest_date: str (%d-%m-%Y date format)
    :param history_days: Number of history days chosen
    :type history_days: int
    :return: string with the (calculated) sowing and harvest date (%d-%m-%Y date format)
    :rtype: str
    """
    harvest_datetime = datetime.strptime(harvest_date, "%d-%m-%Y")
    sowing_datetime = harvest_datetime - timedelta(days=history_days)
    return f'{sowing_datetime.strftime("%Y-%m-%d")}/{harvest_datetime.strftime("%Y-%m-%d")}'


def get_data(bbox: tuple[float, float, float, float], time_period: str, bands: list[str], scale: float) -> xr.Dataset:
    """ Get satellite data.

    :param bbox: Bounding box of the satellite image.
    :type bbox: tuple[float, float, float, float]
    :param time_period: Time period of the satellite image.
    :type time_period: str
    :param bands: List of bands to retrieve.
    :type bands: list[str]
    :param scale: Resolution of the satellite image, defaults to 10.
    :type scale: float
    :return: Dataset processed of an observation.
    :rtype: xr.Dataset
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=time_period
    )
    items = search.item_collection()
    data = stac_load(items, bands=bands, crs="EPSG:4326", resolution=scale, bbox=bbox)
    return data


def save_data(row: pd.Series, history_days: int, history_dates: int, resolution: int) -> xr.Dataset:
    """ Get Satellite Dataset and process it to be used.

    :param row: Series representing an observation.
    :type row: pd.Series
    :param history_days: Number of day to take satellite data before the harvest.
    :type history_days: int
    :param history_dates: Number of satellite data to take before the harvest
    :type history_dates: int
    :param resolution: Resolution of the satellite image, defaults to 10.
    :type resolution: int
    :return: Dataset processed of an observation.
    :rtype: xr.Dataset
    """
    scale = resolution / 111320.0
    bands = ["red", "green", "blue", "B05", "B06", "B07", "nir", "B11", "SCL"]

    longitude = row["Longitude"]
    latitude = row["Latitude"]
    field_size = float(row["Field size (ha)"])
    bbox = get_bbox(longitude, latitude, field_size)

    # Get the time periode to retrieve statellite data
    harvest_date = row["Date of Harvest"]
    time_period = get_time_period(harvest_date, history_days)

    # Get the satellite data
    xds = get_data(bbox, time_period, bands, scale)

    # Cloud mask on SCL value to only keep clear data
    cloud_mask = (
        (xds.SCL != 0)
        & (xds.SCL != 1)
        & (xds.SCL != 3)
        & (xds.SCL != 6)
        & (xds.SCL != 8)
        & (xds.SCL != 9)
        & (xds.SCL != 10)
    )
    xds = xds.where(cloud_mask)

    # Keep only useful data
    xds = xds.drop(["spatial_ref", "SCL"])

    # Compute the mean of each image by localisation
    xds = xds.mean(dim=["latitude", "longitude"], skipna=True)

    # Sort data by time
    xds = xds.sortby("time", ascending=False)

    # Keep only the history_dates oldest data
    xds = xds.isel(time=slice(None, history_dates))

    # Format data
    xds["time"] = xds["time"].dt.strftime("%Y-%m-%d")

    # Create a Variable named state_dev which reprensent
    # the number of development state keep and set it as dimension
    xds["state_dev"] = ("time", np.arange(history_dates)[::-1])
    xds = xds.swap_dims({"time": "state_dev"})

    # Rename bands api name by more readable name
    # Dictionnary for matching api bands name and natural bands name
    dict_band_name = {
        "B05": "rededge1",
        "B06": "rededge2",
        "B07": "rededge3",
        "B11": "swir",
    }
    xds = xds.rename_vars(dict_band_name)

    return xds


def save_data_app(index_row: tuple[str, pd.Series], history_days: int = 130, history_dates: int = 24,
                  resolution: int = 10,) -> xr.Dataset:
    """ Get Satellite Datasets from an observation and concat them to one.

    :param index_row: Tuple of index string and a Series representing an observation.
    :type index_row: tuple[str, pd.Series]
    :param history_days: Number of day to take satellite data before the harvest, defaults to 130.
    :type history_days: int, optional
    :param history_dates: Number of satellite data to take before the harvest, defaults to 24.
    :type history_dates: int, optional
    :param resolution: Resolution of the satellite image, defaults to 10.
    :type resolution: int, optional
    :return: Concatenate Dataset of one observation.
    :rtype: xr.Dataset
    """
    list_xds = []
    # For the number of data augmentation desired get satellite data
    # and append it into a list
    for i in range(NUM_AUGMENT):
        xds = save_data(index_row[1], history_days, history_dates, resolution)
        xds = xds.expand_dims({"ts_aug": [i]})
        list_xds.append(xds)

    # Concat list of Dataset into a single one representing one observation * NUM_AUGMENT
    xds: xr.Dataset = xr.concat(list_xds, dim="ts_aug")

    # Create a new dimenstion called ts_obs representing the index of the observation.
    xds = xds.expand_dims({"ts_obs": [index_row[0]]})

    return xds


def init_df(df: pd.DataFrame, path: str) -> tuple[pd.DataFrame, list]:
    """ Check for missing observations on the dataset and make
        and filter the dataframe to only keep the missing ones.

    :param df: DataFrame of all observations.
    :type df: pd.DataFrame
    :param path: Dataset path of already retrieves data.
    :type path: str
    :return: Dataframe filtered and list with one Dataset if some data was already retrieves.
    :rtype: tuple[pd.DataFrame, list]
    """
    list_data = []
    df.index.name = "ts_obs"

    if os.path.exists(path=path):
        xdf = xr.open_dataset(path, engine="scipy")
        unique = np.unique(xdf["ts_obs"].values)
        list_data.append(xdf)

        df = df.loc[~df.index.isin(unique)]

    return df, list_data


class Checkpoint(Exception):
    """ Exception class to save data during the retrieval. """

    def __init__(self):
        pass


def make_data(path: str, save_file: str) -> bool:
    """ From a given csv at EY data format get satellite data
        corresponding to the localisation and date of each observation
        from microsoft api and save it into external directory.
        Implement an auto restart from the last observation saved.
        Save data as nc format using scipy engine.

    :param path: CSV path of EY data.
    :type path: str
    :param save_file: Directory to save the Dataset.
    :type save_file: str
    :raises Checkpoint: Auto save every hour.
    :return: True if a checkpoint is reached, False otherwise.
    :rtype: bool
    """
    start = time.time()
    checkpoint = False

    df: pd.DataFrame = pd.read_csv(path)

    df, list_data = init_df(df, save_file)

    print(f'\nRetrieve SAR data from {path.split("/")[-1]}...')
    try:
        with mp.Pool(4) as pool:
            # Multiprocessing data retrieval.
            # Create a list of dataset with all dataset represent one obervation
            # multiply by the number of augmentation desired.
            for xds in tqdm(pool.imap(save_data_app, df.iterrows()), total=len(df)):
                list_data.append(xds)
                if time.time() - start > 3600:
                    # Each houre raise a Checkpoint Exception to stop the process and save the data
                    raise Checkpoint("Checkpoint.")
    except Checkpoint as c:
        # If the error is a checkpoint set the boolean to True
        checkpoint = True
    finally:
        # Concat all Dataset to one and save it.
        data = xr.concat(list_data, dim="ts_obs")
        print(f'\nSave SAR data from {path.split("/")[-1]}...')
        data.to_netcdf(save_file, engine="scipy")
        print(f'\nSAR data from {path.split("/")[-1]} saved!')
        return checkpoint


if __name__ == "__main__":
    save_folder = create_folders()

    checkpoint = True
    while checkpoint:
        # While make data finish because of a checkpoint exception
        # Restarts satellite data retrieval.
        train_path = join(ROOT_DIR, "data", "raw", "train.csv")
        train_file = join(save_folder, "train.nc")
        checkpoint = make_data(train_path, train_file)

    checkpoint = True
    while checkpoint:
        # Same for Test data.
        test_path = join(ROOT_DIR, "data", "raw", "test.csv")
        test_file = join(save_folder, "test.nc")
        checkpoint = make_data(test_path, test_file)
