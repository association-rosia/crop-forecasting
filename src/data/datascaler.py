import os
import sys
from typing import Union

import xarray as xr
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

parent = os.path.abspath("../features")
sys.path.insert(1, parent)


class DatasetScaler:
    """Scaler for Vegetable Indice, Geographical, Meteorological and Target.

    :param scaler_s: Scikit-Learn scaler for Vegetable Indice data
    :type scaler_s: Union[StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer]
    :param columns_s: Vegetable Indice columns name
    :type columns_s: list[str]
    :param scaler_g: Scikit-Learn scaler for Geographical data
    :type scaler_g: Union[StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer]
    :param columns_g: Geographical columns name
    :type columns_g: list[str]
    :param scaler_m: Scikit-Learn scaler for Meteorological data
    :type scaler_m: Union[StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer]
    :param columns_m: Meteorological columns name
    :type columns_m: list[str]
    :param scaler_t: Scikit-Learn scaler for Target data
    :type scaler_t: MinMaxScaler
    """
    def __init__(
        self,
        scaler_s: Union[
            StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
        ],
        columns_s: list[str],
        scaler_g: Union[
            StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
        ],
        columns_g: list[str],
        scaler_m: Union[
            StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
        ],
        columns_m: list[str],
        scaler_t: MinMaxScaler,
    ) -> None:
        self.scaler_s = scaler_s
        self.columns_s = columns_s
        self.scaler_g = scaler_g
        self.columns_g = columns_g
        self.scaler_m = scaler_m
        self.columns_m = columns_m
        self.scaler_t = scaler_t

    def fit(self, xdf: xr.Dataset, target: str) -> object:
        """Fit all scalers to be used for later scaling.

        :param xdf: The data used to fit all scalers, used for later scaling along the features axis.
        :type xdf: xr.Dataset
        :param target: Column name to fit the target scaler, used for later scaling along the target axis.
        :type target: str
        :return: Fitted scaler.
        :rtype: object
        """

        def fit_scaler(
            xdf: xr.Dataset,
            columns: list[str],
            scaler: Union[
                StandardScaler,
                RobustScaler,
                PowerTransformer,
                QuantileTransformer,
                MinMaxScaler,
            ],
        ):
            df = xdf[columns].to_dataframe()

            return scaler.fit(df[columns])

        # Fit S data scaler
        self.scaler_s = fit_scaler(xdf, self.columns_s, self.scaler_s)
        # Fit G data scaler
        self.scaler_g = fit_scaler(xdf, self.columns_g, self.scaler_g)
        # Fit M data scaler
        self.scaler_m = fit_scaler(xdf, self.columns_m, self.scaler_m)
        # Fit Target data scaler
        self.scaler_t = fit_scaler(xdf, [target], self.scaler_t)

        return self

    def transform(self, xdf: xr.Dataset, target: str = None) -> xr.Dataset:
        """Perform transform of each scaler.

        :param xdf: The Dataset used to scale along the features axis.
        :type xdf: xr.Dataset
        :param target: Column name used to scale along the Target axis, defaults to None
        :type target: str, optional
        :return: Transformed Dataset.
        :rtype: xr.Dataset
        """

        def transform_data(
            xdf: xr.Dataset,
            columns: str,
            scaler: Union[
                StandardScaler,
                RobustScaler,
                PowerTransformer,
                QuantileTransformer,
                MinMaxScaler,
            ],
        ) -> xr.Dataset:
            df = xdf[columns].to_dataframe()
            df.loc[:, columns] = scaler.transform(df[columns])
            xdf_scale = df[columns].to_xarray()
            xdf = xr.merge([xdf_scale, xdf], compat="override")
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

    def fit_transform(self, xdf: xr.Dataset, target: str) -> xr.Dataset:
        """Fit to data, then transform it.

        :param xdf: The data used to perform fit and transform.
        :type xdf: xr.Dataset
        :param target: Column name used to scale along the Target axis
        :type target: str
        :return: Transformed Dataset.
        :rtype: xr.Dataset
        """
        return self.fit(xdf, target).transform(xdf, target)

    def inverse_transform(self, xdf: xr.Dataset, target: str = None) -> xr.Dataset:
        """Scale back the data to the original representation.

        :param xdf: The data used to scale along the features axis.
        :type xdf: xr.Dataset
        :param target: Column name used to scale along the Target axis, defaults to None
        :type target: str, optional
        :return: Transformed Dataset.
        :rtype: xr.Dataset
        """

        def inverse_transform_data(
            xdf: xr.Dataset,
            columns: str,
            scaler: Union[
                StandardScaler,
                RobustScaler,
                PowerTransformer,
                QuantileTransformer,
                MinMaxScaler,
            ],
        ) -> xr.Dataset:
            df = xdf[columns].to_dataframe()
            df.loc[:, columns] = scaler.inverse_transform(df[columns])
            xdf_scale = df[columns].to_xarray()
            xdf = xr.merge([xdf_scale, xdf], compat="override")
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
