from typing import Union

import xarray as xr

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)


class DatasetScaler:
    def __init__(
            self,
            scaler_s: Union[StandardScaler, MinMaxScaler],
            columns_s: list[str],
            scaler_g: Union[StandardScaler, MinMaxScaler], 
            columns_g: list[str],
            scaler_m: Union[StandardScaler, MinMaxScaler],
            columns_m: list[str],
            scaler_t: Union[StandardScaler, MinMaxScaler],

        ) -> None:
        
        self.scaler_s = scaler_s
        self.columns_s = columns_s
        self.scaler_g = scaler_g
        self.columns_g = columns_g
        self.scaler_m = scaler_m
        self.columns_m = columns_m
        self.scaler_t = scaler_t

    
    def fit(self, xdf: xr.Dataset, target: str) -> None:
        def fit_scaler(xdf: xr.Dataset, columns: list[str], scaler: Union[StandardScaler, MinMaxScaler]):
            df = xdf[columns].to_dataframe()
            scaler.fit(df[columns])
            return scaler

        # Fit S data scaler
        self.scaler_s = fit_scaler(xdf, self.columns_s, self.scaler_s)
        # Fit G data scaler
        self.scaler_g = fit_scaler(xdf, self.columns_g, self.scaler_g)
        # Fit M data scaler
        self.scaler_m = fit_scaler(xdf, self.columns_m, self.scaler_m)
        # Fit Target data scaler
        self.scaler_t = fit_scaler(xdf, [target], self.scaler_t)
    

    def transform(self, xdf: xr.Dataset, target: str=None)->xr.Dataset:
        def transform_data(xdf: xr.Dataset, columns: str, scaler: Union[StandardScaler, MinMaxScaler])->xr.Dataset:
            df = xdf[columns].to_dataframe()
            df.loc[:, columns] = scaler.transform(df[columns])
            xdf_scale = df.to_xarray()
            xdf = xr.merge([xdf_scale, xdf], compat='override')
            return xdf
        
        # Scale S data
        xdf = transform_data(xdf, self.columns_s, self.scaler_s)
        # Scale G data
        xdf = transform_data(xdf, self.columns_g, self.scaler_g)
        # Scale M data
        xdf = transform_data(xdf, self.columns_m, self.scaler_m)

        if target:
            # Scale M data
            xdf = transform_data(xdf, [target], self.scaler_t)

        return xdf
    

    def fit_transform(self, xdf: xr.Dataset, target: str)->xr.Dataset:
        self.fit(xdf, target)
        xdf = self.transform(xdf, target)
        return xdf
    

    def inverse_transform(self, xdf: xr.Dataset, target: str=None):
        def inverse_transform_data(xdf: xr.Dataset, columns: str, scaler: Union[StandardScaler, MinMaxScaler])->xr.Dataset:
            df = xdf[columns].to_dataframe()
            df.loc[:, columns] = scaler.inverse_transform(df[columns])
            xdf_scale = df.to_xarray()
            xdf = xr.merge([xdf_scale, xdf], compat='override')
            return xdf
        
        # Scale S data
        xdf = inverse_transform_data(xdf, self.columns_s, self.scaler_s)
        # Scale G data
        xdf = inverse_transform_data(xdf, self.columns_g, self.scaler_g)
        # Scale M data
        xdf = inverse_transform_data(xdf, self.columns_m, self.scaler_m)

        if target:
            # Scale M data
            xdf = inverse_transform_data(xdf, [target], self.scaler_t)

        return xdf