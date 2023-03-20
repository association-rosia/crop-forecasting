# Native library
import os, sys
path = os.path.join('.')
sys.path.insert(1, path)

from utils import ROOT_DIR

# Save object
import joblib
import json
from random import uniform

from tqdm import tqdm

# Data management
import numpy as np
import pandas as pd
import xarray as xr

from src.constants import TARGET, TARGET_TEST, FOLDER, S_COLUMNS, G_COLUMNS, M_COLUMNS

# Data prepocessing
from src.features.preprocessing import Smoother, Convertor, Scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from umap import UMAP

# Hyperoptimization
from src.features.model_selection import OptunaSearch

# Regressor models
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

# Callback
import wandb

# Model evaluation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

import optuna


DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', FOLDER)


def main():
    path = os.path.join(ROOT_DIR, 'data', 'interim', FOLDER, 'train.nc')
    ds = xr.open_dataset(path, engine='scipy')
    
    X, y = ds.drop(TARGET), preprocess_y(ds, TARGET)
    
    step_list = [
        ('smoother', Smoother()),
        ('convertor', Convertor()),
        ('scaler', Scaler(StandardScaler())),
        ('dim_reductor', UMAP()),
        ('estimator', XGBRegressor())
    ]

    params_optuna = {
        'suggest_categorical': ('smoother__mode', {'choices': [None, 'savgol']})
    }

    pipeline = Pipeline(step_list)

    index_test = get_index_test()

    optunasearch = OptunaSearch(pipeline, params_optuna=params_optuna, index_test=index_test, n_trials=2)
    optunasearch = optunasearch.fit(X, y)

    print(optunasearch.study.best_trial)



def preprocess_y(ds: xr.Dataset, target: str)->pd.DataFrame:
    df = ds[[target]+S_COLUMNS].to_dataframe()
    y = df[['Rice Yield (kg/ha)']].groupby(['ts_aug', 'ts_obs']).first()
    return y.reorder_levels(['ts_obs', 'ts_aug']).sort_index()


def get_index_test():
    path = os.path.join(ROOT_DIR, 'data', 'interim', FOLDER, 'index.json')
    with open(path, 'r') as f:
        index_dict = json.loads(f)
    index_dict['clusters'][0]['train']['ts_obs']
    return None#index_test


if __name__ == '__main__':
    main()