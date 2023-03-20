import joblib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr


from preprocessing import Smoother, Concatenator
from datascaler import DatasetScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os, sys

parent = os.path.abspath(".")
sys.path.insert(1, parent)

from tqdm import tqdm

from src.constants import FOLDER, S_COLUMNS, G_COLUMNS, M_COLUMNS, TARGET, TARGET_TEST

from utils import ROOT_DIR
# from os.path import join


def features_modification(xdf: xr.Dataset, test: bool) -> xr.Dataset:
    xdf["time"] = xdf["time"].astype(np.datetime64)
    xdf["datetime"] = xdf["datetime"].astype(np.datetime64)
    xdf = xdf.reset_coords("time")

    # time and District are keys to link with weather data
    columns = S_COLUMNS + G_COLUMNS + M_COLUMNS + ["time", "District"]
    if test:
        columns.append(TARGET_TEST)
    else:
        columns.append(TARGET)
    xdf = xdf[columns]

    return xdf


def scale_data(xdf: xr.Dataset, dir: str, test: bool) -> xr.Dataset:
    # Path for saving scaler
    path = os.path.join(dir, "scaler_dataset.joblib")

    if not test:
        scaler = DatasetScaler(
            scaler_s=StandardScaler(),
            columns_s=S_COLUMNS,
            scaler_g=StandardScaler(),
            columns_g=G_COLUMNS,
            scaler_m=StandardScaler(),
            columns_m=M_COLUMNS,
            scaler_t=MinMaxScaler(),
        )
        xdf = scaler.fit_transform(xdf, TARGET)
        joblib.dump(scaler, path)
    else:
        scaler: DatasetScaler = joblib.load(path)
        xdf = scaler.transform(xdf)

    return xdf


def create_id(xdf: xr.Dataset) -> xr.Dataset:
        ts_id = np.arange(xdf.dims["ts_obs"] * xdf.dims["ts_aug"])
        ts_id = ts_id.reshape((xdf.dims["ts_obs"], xdf.dims["ts_aug"]))
        xdf = xdf.assign_coords({"ts_id": (("ts_obs", "ts_aug"), ts_id)})
        return xdf


def create_pb(nb_action: int, test: str):
    progress_bar = tqdm(range(nb_action), leave=False)
    if test:
        msg = "Test Dataset - "
    else:
        msg = "Train Dataset - "
    return progress_bar, msg


def process_data(folder: str, test: bool = False):
    pb, msg = create_pb(6, test)
    
    file_name = 'train.nc'
    if test:
        file_name = 'test.nc'

    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed', folder)
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, file_name)
    interim_dir = os.path.join(ROOT_DIR, 'data', 'interim', folder)
    os.makedirs(interim_dir, exist_ok=True)
    interim_path = os.path.join(interim_dir, file_name)

    pb.set_description(msg + "Read Data")
    path_sat = os.path.join(ROOT_DIR, 'data', 'external', 'satellite', folder, file_name)
    xdf = xr.open_dataset(path_sat, engine='scipy')

    # Concatenate and create all features
    pb.update(1)
    pb.refresh()
    pb.set_description(msg + "Concatenate Data")
    xdf = Concatenator().transform(xdf, test)

    # Save for ML 
    xdf.to_netcdf(interim_path, engine='scipy')

    # Smooth variable
    pb.update(2)
    pb.refresh()
    pb.set_description(msg + "Smooth VI")
    xdf = Smoother(mode='savgol').transform(xdf)

    # Create new features
    pb.update(3)
    pb.refresh()
    pb.set_description(msg + "Modification of Features")
    xdf = features_modification(xdf, test)


    # Scale data
    pb.update(4)
    pb.refresh()
    pb.set_description(msg + "Data Scaling")
    xdf = scale_data(xdf, processed_dir, test)

    # Add an id for each line
    pb.update(5)
    pb.refresh()
    pb.set_description(msg + "Create an Index 1D")
    xdf = create_id(xdf)

    # Save data
    pb.update(6)
    pb.refresh()
    pb.set_description(msg + "Saving Data")
    xdf.to_netcdf(processed_path, engine='scipy')

if __name__ == "__main__":
    # Cloud filtered data
    process_data(FOLDER)
    process_data(FOLDER, test=True)