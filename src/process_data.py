import glob
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xarray as xr

from scipy.signal import savgol_filter

from sklearn.preprocessing import MinMaxScaler, StandardScaler

BANDS = ['red', 'green', 'blue', 'rededge1', 'rededge2', 'rededge3', 'nir', 'swir']
VI = ['ndvi', 'savi', 'evi', 'rep','osavi','rdvi','mtvi1','lswi']
M_COLUMNS = ['tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'windspeed', 'winddir', 
             'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase', 'solarexposure']

S_COLUMNS = ['ndvi', 'savi', 'evi', 'rep', 'osavi', 'rdvi', 'mtvi1', 'lswi']
G_COLUMNS = [ 'Field size (ha)', 'Rice Crop Intensity(D=Double, T=Triple)']
TARGET = 'Rice Yield (kg/ha)'


def add_weather(xdf: xr.Dataset)->xr.Dataset:
    xdf = xdf#.set_coords('District')

    weather = []
    for path in glob.glob('../data/raw/weather/*.csv'):
        weather.append(pd.read_csv(path))

    df_weather = pd.concat(weather, axis='index')
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather['name'] = df_weather['name'].str.replace(' ', '_')
    # df_weather = df_weather.rename(columns={'datetime': 'time'})
    df_weather.set_index(['datetime', 'name'], inplace=True)
    xdf_weather = df_weather.to_xarray().set_coords(['datetime', 'name'])
    xdf_weather['datetime'] = xdf_weather['datetime'].dt.strftime('%Y-%m-%d')

    xdf = xr.merge([xdf, xdf_weather])

    return xdf


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

    # compute all vegetable indice
    xdf['ndvi'] = compute_ndvi(xdf)
    xdf['savi'] = compute_savi(xdf)
    xdf['evi'] = compute_evi(xdf)
    xdf['rep'] = compute_rep(xdf)
    xdf['osavi'] = compute_osavi(xdf)
    xdf['rdvi'] = compute_rdvi(xdf)
    xdf['mtvi1'] = compute_mtvi1(xdf)
    xdf['lswi'] = compute_lswi(xdf)

    return xdf


def statedev_fill(xdf: xr.Dataset)->xr.Dataset:
    # compute mean of all stage of developpement of rice field
    xdf_mean = xdf.mean('ts_id', skipna=True)
    # fill na value with computed mean
    xdf = xdf.fillna(xdf_mean)
    return xdf


def smooth(xdf: xr.Dataset)->xr.Dataset:
    # apply savgol_filter to vegetable indice
    xdf_s = xr.apply_ufunc(savgol_filter, xdf[S_COLUMNS], kwargs={'axis': 1, 'window_length':12, 'polyorder':4})
    # merge both dataset and override old vegetable indice and bands
    return xr.merge([xdf_s, xdf], compat='override')


def categorical_encoding(xdf: xr.Dataset)->xr.Dataset:
    xdf['Rice Crop Intensity(D=Double, T=Triple)'] = xdf['Rice Crop Intensity(D=Double, T=Triple)'].str.replace("D", "2").str.replace("T", "3").astype(np.int8)
    return xdf


def features_modification(xdf: xr.Dataset)->xr.Dataset:
    xdf['sunrise'] = xdf['sunrise'].astype(np.datetime64)
    xdf['sunset'] = xdf['sunset'].astype(np.datetime64)

    xdf['solarexposure'] = (xdf['sunset'] - xdf['sunrise']).dt.seconds

    xdf['time'] = xdf['time'].astype(np.datetime64)
    xdf['datetime'] = xdf['datetime'].astype(np.datetime64)
    xdf = xdf.reset_coords('time')

    return xdf


def scale_data(xdf: xr.Dataset, path: str, test: bool)->xr.Dataset:
    # Path for 
    path_s = '/'.join(path.split('/')[:-1]) + "/scaler_s.joblib"
    path_g = '/'.join(path.split('/')[:-1]) + "/scaler_g.joblib"
    path_m = '/'.join(path.split('/')[:-1]) + "/scaler_m.joblib"
    path_t = '/'.join(path.split('/')[:-1]) + "/scaler_t.joblib"

    if not test:
        # Scale S data
        scaler_s = StandardScaler()
        df = xdf.reset_coords('District')[S_COLUMNS].to_dataframe()
        df.loc[:, S_COLUMNS] = scaler_s.fit_transform(df[S_COLUMNS])
        xdf_scale = df.to_xarray()
        xdf = xr.merge([xdf_scale, xdf], compat='override')
        joblib.dump(scaler_s, path_s)

        # Scale G data
        scaler_g = StandardScaler()
        df = xdf[G_COLUMNS].to_dataframe()
        df.loc[:, G_COLUMNS] = scaler_g.fit_transform(df[G_COLUMNS])
        xdf_scale = df.to_xarray()
        xdf = xr.merge([xdf_scale, xdf], compat='override')
        joblib.dump(scaler_g, path_g)

        # Scale M data
        scaler_m = StandardScaler()
        df = xdf[M_COLUMNS].to_dataframe()
        df.loc[:, M_COLUMNS] = scaler_m.fit_transform(df[M_COLUMNS])
        xdf_scale = df.to_xarray()
        xdf = xr.merge([xdf_scale, xdf], compat='override')
        joblib.dump(scaler_m, path_m)

        # Scale Target
        scaler_t = MinMaxScaler()
        arr = xdf[TARGET].to_numpy().reshape(-1, 1)
        xdf[TARGET].values = scaler_t.fit_transform(arr).reshape(-1)
        joblib.dump(scaler_t, path_t)
    else:
        # Scale S data
        scaler_s = joblib.load(path_s)
        df = xdf[S_COLUMNS].to_dataframe()
        df.loc[:, S_COLUMNS] = scaler_s.transform(df[S_COLUMNS])
        xdf_scale = df.to_xarray()
        xdf = xr.merge([xdf_scale, xdf], compat='override')

        # Scale G data
        scaler_g = joblib.load(path_g)
        df = xdf[G_COLUMNS].to_dataframe()
        df.loc[:, G_COLUMNS] = scaler_g.transform(df[G_COLUMNS])
        xdf_scale = df.to_xarray()
        xdf = xr.merge([xdf_scale, xdf], compat='override')

        # Scale M data
        scaler_m = joblib.load(path_m)
        df = xdf[M_COLUMNS].to_dataframe()
        df.loc[:, M_COLUMNS] = scaler_m.transform(df[M_COLUMNS])
        xdf_scale = df.to_xarray()
        xdf = xr.merge([xdf_scale, xdf], compat='override')

    return xdf


def process_data(path: str, test: bool=False):
    xdf = xr.open_dataset(path)
    # Add weather to the dataset
    xdf = add_weather(xdf)
    # Compute vegetable indice
    xdf = compute_vi(xdf)
    # Fill na values
    xdf = statedev_fill(xdf)
    # Smooth variable
    xdf = smooth(xdf)
    # Create new features
    xdf = features_modification(xdf)
    # Encode categorical features
    xdf = categorical_encoding(xdf)
    # Scale data
    xdf = scale_data(xdf, path, test)
    # Save data
    path = '.'.join(path.split('.')[:-1]) + "_processed." + path.split('.')[-1]
    xdf.to_netcdf(path, engine='scipy')

if __name__ == '__main__':

    # Cloud filtered data
    train_filter_path = '../data/processed/augment_10_5/train_filter.nc'
    process_data(train_filter_path)
    test_filter_path = '../data/processed/augment_10_5/test_filter.nc'
    process_data(test_filter_path, test=True)

    # train_path = '../data/processed/adaptative_factor_1/train.nc'
    # process_data(train_path, fill=False)
    # test_path = '../data/processed/adaptative_factor_1/test.nc'
    # process_data(test_path, fill=False)