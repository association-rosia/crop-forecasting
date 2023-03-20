from typing import Union

import xarray as xr

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from src.constants import FOLDER, S_COLUMNS, G_COLUMNS, M_COLUMNS, TARGET

class DatasetScaler:
    def __init__(self) -> None:
        pass
    
    def fit(self, xdf: xr.Dataset):
        def fit_scaler(xdf: xr.Dataset, columns: list[str], mode: str='standard'):
            if mode == 'standard':
                scaler = StandardScaler()
            elif mode == 'minmax':
                scaler = MinMaxScaler()
            df = xdf[columns].to_dataframe()
            scaler.fit(df[columns])
            return scaler

        # Fit S data scaler
        self.scaler_s = fit_scaler(xdf, S_COLUMNS)
        # Fit G data scaler
        self.scaler_g = fit_scaler(xdf, G_COLUMNS)
        # Fit M data scaler
        self.scaler_m = fit_scaler(xdf, M_COLUMNS)
        # Fit Target data scaler
        self.scaler_t = fit_scaler(xdf, [TARGET], 'minmax')
    

    def transform(self, xdf: xr.Dataset, target: bool=False) -> xr.Dataset:
        def transform_data(xdf: xr.Dataset, columns: str, scaler: Union[StandardScaler, MinMaxScaler]) -> xr.Dataset:
            df = xdf[columns].to_dataframe()
            df.loc[:, columns] = scaler.transform(df[columns])
            xdf_scale = df.to_xarray()
            xdf = xr.merge([xdf_scale, xdf], compat='override')
            return xdf
        
        # Scale S data
        xdf = transform_data(xdf, S_COLUMNS, self.scaler_s)
        # Scale G data
        xdf = transform_data(xdf, G_COLUMNS, self.scaler_g)
        # Scale M data
        xdf = transform_data(xdf, M_COLUMNS, self.scaler_m)

        if target:
            # Scale M data
            xdf = transform_data(xdf, [TARGET], self.scaler_t)

        return xdf
    
    def fit_transform(self, xdf: xr.Dataset) -> xr.Dataset:
        self.fit(xdf)
        xdf = self.transform(xdf, True)
        return xdf