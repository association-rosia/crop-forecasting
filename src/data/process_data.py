import warnings

import joblib

warnings.filterwarnings("ignore")

import os
import sys
import glob
from os.path import join
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from src.data.datascaler import DatasetScaler
from src.data.preprocessing import Smoother
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from src.constants import (FOLDER, G_COLUMNS, M_COLUMNS, S_COLUMNS, TARGET,
                           TARGET_TEST)
from utils import ROOT_DIR


def merge_satellite(file: str)->xr.Dataset:
    """Merge Augmented 10 / 40 / 50 to another one.

    :param file: File name of all datasets.
    :type file: str
    :return: Merged dataset.
    :rtype: xr.Dataset
    """
    # Open dataset
    def open_dataset(folder):
        return xr.open_dataset(
            join(ROOT_DIR, "data", "external", "satellite", folder, file),
            engine="scipy",
        )

    folder = "augment_50_5"
    xds_50 = open_dataset(folder)

    folder = "augment_40_5"
    xds_40 = open_dataset(folder)
    # Change number of ts_aug to not be overwrite during the merge
    xds_40["ts_aug"] = np.arange(50, 90)

    folder = "augment_10_5"
    xds_10 = open_dataset(folder)
    # Same
    xds_10["ts_aug"] = np.arange(90, 100)

    xds_100 = xr.merge([xds_50, xds_40, xds_10], compat="no_conflicts")

    return xds_100


def add_observation(xds: xr.Dataset, test: bool) -> xr.Dataset:
    """Process and Merge EY data to Satellite Dataset.

    :param xds: Satellite Dataset that will be merged
    :type xds: xr.Dataset
    :param test: True if it is the test Dataset.
    :type test: bool
    :return: Merged Dataset.
    :rtype: xr.Dataset
    """
    def categorical_encoding(xds: xr.Dataset) -> xr.Dataset:
        # Encode Rice Crop Intensity feature D = 2 and T = 3
        xds["Rice Crop Intensity(D=Double, T=Triple)"] = (
            xds["Rice Crop Intensity(D=Double, T=Triple)"]
            .str.replace("D", "2")
            .str.replace("T", "3")
            .astype(np.int8)
        )
        return xds

    file_name = "train_enriched.csv"
    if test:
        file_name = "test_enriched.csv"

    path = join(ROOT_DIR, "data", "interim", file_name)
    # Read csv EY data
    df = pd.read_csv(path)
    # Set index name as ts_obs for linked both Dataset
    df.index.name = "ts_obs"
    # Convert pandas.DataFrame into xarray.Dataset and merge on ts_obs
    xds = xr.merge([xds, df.to_xarray()], compat='override')
    # Encode categoricals data
    xds = categorical_encoding(xds)

    return xds


def add_weather(xds: xr.Dataset) -> xr.Dataset:
    """Add meteorological data to the Dataset.

    :param xds: Dataset that will be merged.
    :type xds: xr.Dataset
    :return: Merged Dataset.
    :rtype: xr.Dataset
    """

    def features_modification(xds: xr.Dataset) -> xr.Dataset:
        # Crreate new features named solarexposure
        # It is the difference between sunset and sunrise
        xds["sunrise"] = xds["sunrise"].astype(np.datetime64)
        xds["sunset"] = xds["sunset"].astype(np.datetime64)

        xds["solarexposure"] = (xds["sunset"] - xds["sunrise"]).dt.seconds
        return xds

    # Read all weather csv and create a pandas.DataFrame of its
    weather = []
    for path in glob.glob(join(ROOT_DIR, "data", "external", "weather", "*.csv")):
        weather.append(pd.read_csv(path))
    df_weather = pd.concat(weather, axis="index")

    # Convert timestamp into datetime for future purpose
    df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
    # Format name to match District features
    df_weather["name"] = df_weather["name"].str.replace(" ", "_")
    # Set as index datetime and name to became dimensions with the
    # xarray.Dataset conversion
    df_weather.set_index(["datetime", "name"], inplace=True)
    xds_weather = df_weather.to_xarray().set_coords(["datetime", "name"])
    xds_weather["datetime"] = xds_weather["datetime"].dt.strftime("%Y-%m-%d")
    # Feature engineering on weather data
    xds_weather = features_modification(xds_weather)
    # Merge both Dataset
    xds = xr.merge([xds, xds_weather])

    return xds


def compute_vi(xds: xr.Dataset) -> xr.Dataset:
    """Compute vegetable indices. That include NDVI, SAVI, EVI, REP, OSAVI, RDVI, MTVI1, LSWI.

    :param xds: Dataset that include satellite band data, used to compute vegetable indice.
    :type xds: xr.Dataset
    :return: Merged Dataset.
    :rtype: xr.Dataset
    """
    # Compute vegetable indices

    def compute_ndvi(xds: xr.Dataset) -> xr.Dataset:
        # Compute ndvi indice
        return (xds.nir - xds.red) / (xds.nir + xds.red)

    def compute_savi(xds, L=0.5) -> xr.Dataset:
        # Compute savi indice
        return 1 + L * (xds.nir - xds.red) / (xds.nir + xds.red + L)

    def compute_evi(xds, G=2.5, L=1, C1=6, C2=7.5) -> xr.Dataset:
        # Compute evi indice
        return G * (xds.nir - xds.red) / (xds.nir + C1 * xds.red - C2 * xds.blue + L)

    def compute_rep(xds: xr.Dataset) -> xr.Dataset:
        # Compute rep indice
        rededge = (xds.red + xds.rededge3) / 2
        return 704 + 35 * (rededge - xds.rededge1) / (xds.rededge2 - xds.rededge1)

    def compute_osavi(xds: xr.Dataset) -> xr.Dataset:
        # Compute osavi indice
        return (xds.nir - xds.red) / (xds.nir + xds.red + 0.16)

    def compute_rdvi(xds: xr.Dataset) -> xr.Dataset:
        # Compute rdvi indice
        return (xds.nir - xds.red) / np.sqrt(xds.nir + xds.red)

    def compute_mtvi1(xds: xr.Dataset) -> xr.Dataset:
        # Compute mtvi1 indice
        return 1.2 * (1.2 * (xds.nir - xds.green) - 2.5 * (xds.red - xds.green))

    def compute_lswi(xds: xr.Dataset) -> xr.Dataset:
        # Compute lswi indice
        return (xds.nir - xds.swir) / (xds.nir + xds.swir)

    xds["ndvi"] = compute_ndvi(xds)
    xds["savi"] = compute_savi(xds)
    xds["evi"] = compute_evi(xds)
    xds["rep"] = compute_rep(xds)
    xds["osavi"] = compute_osavi(xds)
    xds["rdvi"] = compute_rdvi(xds)
    xds["mtvi1"] = compute_mtvi1(xds)
    xds["lswi"] = compute_lswi(xds)

    return xds


def statedev_fill(xds: xr.Dataset) -> xr.Dataset:
    # Fill missing vegetable indice and replace abnormal values

    def replaceinf(arr: np.ndarray) -> np.ndarray:
        if np.issubdtype(arr.dtype, np.number):
            arr[np.isinf(arr)] = np.nan
        return arr

    # replace ± infinite value by na
    xr.apply_ufunc(replaceinf, xds[S_COLUMNS])
    # compute mean of all stage of developpement and all obsevation
    xds_mean = xds[S_COLUMNS].mean(dim="ts_aug", skipna=True)
    # fill na value with computed mean
    xds[S_COLUMNS] = xds[S_COLUMNS].fillna(xds_mean)
    # compute mean of all stage of developpement of rice field to complete last na values
    xds_mean = xds_mean.mean(dim="ts_obs", skipna=True)
    # fill na value with computed mean
    xds[S_COLUMNS] = xds[S_COLUMNS].fillna(xds_mean)

    return xds


def features_modification(xds: xr.Dataset, test: bool) -> xr.Dataset:
    """Reduce dimension of the Dataset to only keep useful features for training.
    Transform features for training.

    :param xds: The Dataset used to perform dimension reduction and transform timestamp into numpy.datetime64.
    :type xds: xr.Dataset
    :param test: If True then the target name is 'Predicted Rice Yield (kg/ha)' else it is 'Rice Yield (kg/ha)'.
    :type test: bool
    :return: Transformed Dataset.
    :rtype: xr.Dataset
    """
    xds["time"] = xds["time"].astype(np.datetime64)
    xds["datetime"] = xds["datetime"].astype(np.datetime64)
    xds = xds.reset_coords("time")

    # time and District are keys to link with weather data
    columns = S_COLUMNS + G_COLUMNS + M_COLUMNS + ["time", "District"]
    if test:
        columns.append(TARGET_TEST)
    else:
        columns.append(TARGET)
    xds = xds[columns]

    return xds


def scale_data(xds: xr.Dataset, dir: str, test: bool) -> xr.Dataset:
    """Scale all features of the Dataset and save the scaler.

    :param xds: The Dataset used to perform the scaling.
    :type xds: xr.Dataset
    :param dir: Directory to save the scaler.
    :type dir: str
    :param test: If True then perform a transform else perform a fit_transform.
    :type test: bool
    :return: Transformed Dataset.
    :rtype: xr.Dataset
    """
    # Path for saving scaler
    path = join(dir, "scaler_dataset.joblib")
    # Perform a fit_transform else Perform a transform.
    if not test:
        # Initialised scaler and all subscaler
        scaler = DatasetScaler(
            scaler_s=StandardScaler(),
            columns_s=S_COLUMNS,
            scaler_g=StandardScaler(),
            columns_g=G_COLUMNS,
            scaler_m=StandardScaler(),
            columns_m=M_COLUMNS,
            scaler_t=MinMaxScaler(),
        )
        # Fit the scaler and Transform the data
        xds = scaler.fit_transform(xds, TARGET)
        # Save the scaler
        joblib.dump(scaler, path)
    else:
        # Load scaler and transform data
        scaler: DatasetScaler = joblib.load(path)
        xds = scaler.transform(xds)

    return xds


def create_id(xds: xr.Dataset) -> xr.Dataset:
    """Add the coordinate ts_id to be used as index in the Pytorch Dataset.

    :param xds: Dataset used to add IDs.
    :type xds: xr.Dataset
    :return: Transformed Dataset.
    :rtype: xr.Dataset
    """
    # Create np.ndarray with unique integer of the dimension number of Observation * number of Augmentation.
    ts_id = np.arange(xds.dims["ts_obs"] * xds.dims["ts_aug"])
    # Reshape and assign it as coordinate to the Dataset
    ts_id = ts_id.reshape((xds.dims["ts_obs"], xds.dims["ts_aug"]))
    xds = xds.assign_coords({"ts_id": (("ts_obs", "ts_aug"), ts_id)})
    return xds


def create_pb(nb_action: int, test: bool) -> tuple:
    """Initialise tqdm progressbar for preprossessing verbose purpose.

    :param nb_action: Number of preprossessing steps.
    :type nb_action: int
    :param test: True if it is the Test preprossessing.
    :type test: bool
    :return: Progressbar and Begining of the message for the progressbar.
    :rtype: tuple
    """
    progress_bar = tqdm(range(nb_action), leave=False)
    if test:
        msg = "Test Dataset - "
    else:
        msg = "Train Dataset - "
    return progress_bar, msg


def process_data(folder: str, test: bool = False) -> None:
    """Prepare data for Deep Learning and Machine Learning purpose and save it in processed directory.

    :param folder: Directory to load Satellite Dataset.
    :type folder: str
    :param test: True if it is the Test preprossessing, defaults to False
    :type test: bool, optional
    """
    # Create the progress bar
    pb, msg = create_pb(9, test)

    # Determine the name of the processed / original dataset file
    file_name = "train.nc"
    if test:
        file_name = "test.nc"

    # Create all directories useful
    processed_dir = join(ROOT_DIR, "data", "processed", folder)
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = join(processed_dir, file_name)

    interim_dir = join(ROOT_DIR, "data", "interim", folder)
    os.makedirs(interim_dir, exist_ok=True)
    interim_path = join(interim_dir, file_name)

    # Load Satellite Dataset
    pb.set_description(msg + "Read Data")
    if folder == "augment_100_5":
        xds = merge_satellite(file_name)
    else:
        path_sat = join(ROOT_DIR, "data", "external", "satellite", folder, file_name)
        xds = xr.open_dataset(path_sat, engine="scipy")

    # Concatenate and create all features
    pb.update(1)
    pb.refresh()
    pb.set_description(msg + "Add Paddies Data")
    # Process and Merge EY data to Satellite Dataset
    xds = add_observation(xds, test)

    pb.update(2)
    pb.refresh()
    pb.set_description(msg + "Add Meteorological Data")
    # Process and Merge Weather data to Satellite & EY Dataset
    xds = add_weather(xds)

    pb.update(3)
    pb.refresh()
    pb.set_description(msg + "Compute Vegetable Indices")
    # Compute vegetable indices
    xds = compute_vi(xds)

    # Save for ML
    xds.to_netcdf(interim_path, engine="scipy")

    pb.update(4)
    pb.refresh()
    pb.set_description(msg + "Fill NaN values")
    # Fill missing vegetable indice and replace abnormal values
    xds = statedev_fill(xds)

    # Smooth variable
    pb.update(5)
    pb.refresh()
    pb.set_description(msg + "Smooth VI")
    xds = Smoother(mode='savgol').transform(xds)

    # Create new features
    pb.update(6)
    pb.refresh()
    pb.set_description(msg + "Modification of Features")
    xds = features_modification(xds, test)

    # Scale data
    pb.update(7)
    pb.refresh()
    pb.set_description(msg + "Data Scaling")
    xds = scale_data(xds, processed_dir, test)

    # Add an id for each line
    pb.update(8)
    pb.refresh()
    pb.set_description(msg + "Create an Index 1D")
    xds = create_id(xds)

    # Save data for DL
    pb.update(9)
    pb.refresh()
    pb.set_description(msg + "Saving Data")
    xds.to_netcdf(processed_path, engine="scipy")


if __name__ == "__main__":
    process_data(FOLDER)
    process_data(FOLDER, test=True)
