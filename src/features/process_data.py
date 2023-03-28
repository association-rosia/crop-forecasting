import warnings

import joblib

warnings.filterwarnings("ignore")

import os
import sys

import numpy as np
import xarray as xr
from datascaler import DatasetScaler
from preprocessing import Concatenator, Smoother
from sklearn.preprocessing import MinMaxScaler, StandardScaler

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from tqdm import tqdm

from src.constants import (FOLDER, G_COLUMNS, M_COLUMNS, S_COLUMNS, TARGET,
                           TARGET_TEST)
from utils import ROOT_DIR


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
    path = os.path.join(dir, "scaler_dataset.joblib")
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
    pb, msg = create_pb(6, test)
    # Determine the name of the processed dataset file
    file_name = "train.nc"
    if test:
        file_name = "test.nc"
    # Create all directories useful
    processed_dir = os.path.join(ROOT_DIR, "data", "processed", folder)
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, file_name)
    interim_dir = os.path.join(ROOT_DIR, "data", "interim", folder)
    os.makedirs(interim_dir, exist_ok=True)
    interim_path = os.path.join(interim_dir, file_name)
    # Load Satellite Dataset
    pb.set_description(msg + "Read Data")
    path_sat = os.path.join(
        ROOT_DIR, "data", "external", "satellite", folder, file_name
    )
    xds = xr.open_dataset(path_sat, engine="scipy")

    # Concatenate and create all features
    pb.update(1)
    pb.refresh()
    pb.set_description(msg + "Concatenate Data")
    xds = Concatenator().transform(xds, test)

    # Save for ML
    xds.to_netcdf(interim_path, engine="scipy")

    # Smooth variable
    pb.update(2)
    pb.refresh()
    pb.set_description(msg + "Smooth VI")
    xds = Smoother(mode="savgol").transform(xds)

    # Create new features
    pb.update(3)
    pb.refresh()
    pb.set_description(msg + "Modification of Features")
    xds = features_modification(xds, test)

    # Scale data
    pb.update(4)
    pb.refresh()
    pb.set_description(msg + "Data Scaling")
    xds = scale_data(xds, processed_dir, test)

    # Add an id for each line
    pb.update(5)
    pb.refresh()
    pb.set_description(msg + "Create an Index 1D")
    xds = create_id(xds)

    # Save data for DL
    pb.update(6)
    pb.refresh()
    pb.set_description(msg + "Saving Data")
    xds.to_netcdf(processed_path, engine="scipy")


if __name__ == "__main__":
    process_data(FOLDER)
    process_data(FOLDER, test=True)
