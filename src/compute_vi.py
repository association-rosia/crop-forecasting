import xarray as xr
import numpy as np
from scipy.signal import savgol_filter


def compute_vi(xdf: xr.Dataset)->xr.Dataset:

    def compute_ndvi(xdf: xr.Dataset)->xr.Dataset:
        return (xdf.nir - xdf.red) / (xdf.nir + xdf.red)

    def compute_savi(xdf, L=0.5):
        return 1 + L * (xdf.nir - xdf.red) / (xdf.nir + xdf.red + L)

    def compute_evi(xdf, G=2.5, L=1, C1=6, C2=7.5):
        return G * (xdf.nir - xdf.red) / (xdf.nir + C1 * xdf.red - C2 * xdf.blue + L)

    def compute_rep(xdf: xr.Dataset)->xr.Dataset:
        rededge = (xdf.red + xdf.rededge3) / 2
        return 704 + 35 * (rededge - xdf.rededge1) / (xdf.rededge2 - xdf.rededge1)

    def compute_osavi(xdf: xr.Dataset)->xr.Dataset:
        return (xdf.nir - xdf.red) / (xdf.nir + xdf.red + 0.16)

    def compute_rdvi(xdf: xr.Dataset)->xr.Dataset:
        return (xdf.nir - xdf.red) / np.sqrt(xdf.nir + xdf.red)

    def compute_mtvi1(xdf: xr.Dataset)->xr.Dataset:
        return 1.2 * (1.2 * (xdf.nir - xdf.green) - 2.5 * (xdf.red - xdf.green))

    def compute_lswi(xdf: xr.Dataset)->xr.Dataset:
        return (xdf.nir - xdf.swir) / (xdf.nir + xdf.swir)

    # list of satellite band
    bands = ['red', 'green', 'blue', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir']
    # compute all vegetable indice
    xdf['ndvi'] = compute_ndvi(xdf)
    xdf['savi'] = compute_savi(xdf)
    xdf['evi'] = compute_evi(xdf)
    xdf['rep'] = compute_rep(xdf)
    xdf['osavi'] = compute_osavi(xdf)
    xdf['rdvi'] = compute_rdvi(xdf)
    xdf['mtvi1'] = compute_mtvi1(xdf)
    xdf['lswi'] = compute_lswi(xdf)
    # drop satellite band from the dataset
    xdf = xdf.drop(bands)
    return xdf


def statedev_fill(xdf: xr.Dataset)->xr.Dataset:
    # compute mean of all stage of developpement of rice field
    xdf_mean = xdf.mean('ts_id', skipna=True)
    # fill na value with computed mean
    xdf = xdf.fillna(xdf_mean)
    return xdf


def smooth_vi(xdf: xr.Dataset)->xr.Dataset:
    # list of vegetable indice
    vi = ['ndvi', 'savi', 'evi', 'rep','osavi','rdvi','mtvi1','lswi']
    # apply savgol_filter to vegetable indice
    xdf_vi = xr.apply_ufunc(savgol_filter, xdf[vi], kwargs={'axis': 1, 'window_length':12, 'polyorder':4})
    # merge both dataset and override old vegetable indice
    return xr.merge([xdf_vi, xdf], compat='override')


def process_data(path, fill=True):
    xdf = xr.open_dataset(path)
    
    xdf = compute_vi(xdf)
    path = '.'.join(path.split('.')[:-1]) + "_vi." + path.split('.')[-1]
    if not fill:
        xdf.to_netcdf(path, engine='scipy')

    if fill:
        xdf = statedev_fill(xdf)
        path = '.'.join(path.split('.')[:-1]) + "_fill." + path.split('.')[-1]
        xdf.to_netcdf(path, engine='scipy')

    xdf = smooth_vi(xdf)
    path = '.'.join(path.split('.')[:-1]) + "_smooth." + path.split('.')[-1]
    xdf.to_netcdf(path, engine='scipy')

if __name__ == '__main__':

    train_filter_path = '../data/processed/adaptative_factor_1/train_filter.nc'
    process_data(train_filter_path)

    train_path = '../data/processed/adaptative_factor_1/train.nc'
    process_data(train_path, fill=False)

    test_filter_path = '../data/processed/adaptative_factor_1/test_filter.nc'
    process_data(test_filter_path)

    test_path = '../data/processed/adaptative_factor_1/test.nc'
    process_data(test_path, fill=False)