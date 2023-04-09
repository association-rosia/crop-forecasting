# Native library
import os, sys

path = os.path.join(".")
sys.path.insert(1, path)

from utils import ROOT_DIR

# Data management
import numpy as np
import xarray as xr

from src.constants import TARGET, FOLDER, S_COLUMNS

from tqdm import tqdm

# Data prepocessing
from src.data.preprocessing import Smoother, Convertor, Filler, Sorter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Hyperparameter Optimization
import wandb

# Regressor models
from xgboost import XGBRegressor

# Training
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


def main():
    data_path = os.path.join(ROOT_DIR, "data", "interim", FOLDER, "train.nc")
    xds = xr.open_dataset(data_path, engine="scipy")

    obs_idx = xds["ts_obs"].values
    obs_idx = obs_idx.reshape(-1, 1)

    wandb.init(
        project="winged-bull",
        group="Machine Learning",
    )

    pipeline = init_pipeline()

    val_R2_score = 0
    n_splits = wandb.config.n_splits

    p_bar = tqdm(KFold(n_splits=n_splits).split(obs_idx), total=n_splits, leave=False)
    for i, (index_train, index_test) in enumerate(p_bar):
        xds_train = xds.sel(ts_obs=obs_idx[index_train].reshape(-1))
        xds_test = xds.sel(ts_obs=obs_idx[index_test].reshape(-1))

        y_train = preprocess_y(xds_train)
        y_test = preprocess_y(xds_test)

        pipeline.fit(X=xds_train, y=y_train)
        val_split_R2_score = pipeline.score(X=xds_test, y=y_test)
        val_R2_score += val_split_R2_score
        p_bar.write(f"Split {i + 1}/{n_splits}: R2 score = {val_split_R2_score:.5f}")
        wandb.log({"val_split_r2_score": val_split_R2_score})

    print(f"Mean R2 score = {(val_R2_score / n_splits):.5f}")
    wandb.log({"val_r2_score": val_R2_score / n_splits})


def init_pipeline() -> Pipeline:
    params_pipeline = {
        "smoother__mode": None if isinstance(wandb.config.vi, bool) else wandb.config.vi,
        "convertor__agg": wandb.config.dim_reduction == "Aggregate",
        "convertor__weather": wandb.config.weather,
        "convertor__vi": isinstance(wandb.config.vi, str) or  wandb.config.vi,
        "estimator__n_estimators": wandb.config.n_estimators,
        "estimator__colsample_bytree": wandb.config.colsample_bytree,
        "estimator__colsample_bylevel": wandb.config.colsample_bylevel,
        "estimator__colsample_bynode": wandb.config.colsample_bynode,
        "estimator__subsample": wandb.config.subsample,
        "estimator__max_depth": wandb.config.max_depth,
        "estimator__learning_rate": wandb.config.learning_rate,
    }

    steps_pipeline = [
        ("filler", Filler()),
        ("smoother", Smoother()),
        ("convertor", Convertor()),
        ("sorter", Sorter()),
    ]

    if wandb.config.dim_reduction == "PCA":
        steps_pipeline.append(("scaler", StandardScaler()))
        steps_pipeline.append(("dim_reductor", PCA(n_components="mle")))

    steps_pipeline.append(("estimator", XGBRegressor()))

    pipeline = Pipeline(steps_pipeline)
    pipeline.set_params(**params_pipeline)

    return pipeline


def preprocess_y(xds: xr.Dataset) -> np.ndarray:
    df = xds[[TARGET] + S_COLUMNS].to_dataframe()
    y = df[[TARGET]].groupby(["ts_obs", "ts_aug"]).first()
    return y.reorder_levels(["ts_obs", "ts_aug"]).sort_index().to_numpy()


if __name__ == "__main__":
    main()
