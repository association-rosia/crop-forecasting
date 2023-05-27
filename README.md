# 2023 EY Open Science Data Challenge - Crop forecasting

## Project Description

The project 2023 EY Open Science Data Challenge - Crop Forecasting is a Data Science project conducted as part of the challenge proposed by EY, Microsoft, and Cornell University. The objective of this project is to predict the yield of rice fields using satellite image data provided by Microsoft Planetary, meteorological data, and field data.

## Our model architecture

## Documentation

The project documentation, generated using Sphinx, can be found in the `docs/` directory. It provides detailed information about the project's setup, usage, implementation, tutorial.


## Directory Structure

The directory structure of the project is as follows:

```
.
├── Makefile
├── README.md
├── data
│   ├── external
│   │   ├── satellite
│   │   │   ├── augment_10_5
│   │   │   │   ├── test.nc
│   │   │   │   └── train.nc
│   │   │   ├── augment_40_5
│   │   │   │   ├── test.nc
│   │   │   │   └── train.nc
│   │   │   └── augment_50_5
│   │   │       ├── test.nc
│   │   │       └── train.nc
│   │   └── weather
│   │       ├── Chau Phu.csv
│   │       ├── Chau Thanh.csv
│   │       └── Thoai Son.csv
│   ├── interim
│   │   ├── augment_100_5
│   │   │   ├── test.nc
│   │   │   └── train.nc
│   │   ├── index.json
│   │   ├── test_enriched.csv
│   │   └── train_enriched.csv
│   ├── processed
│   │   ├── augment_100_5
│   │   │   ├── scaler_dataset.joblib
│   │   │   ├── test.nc
│   │   │   ├── test_enriched.nc
│   │   │   ├── train.nc
│   │   │   └── train_enriched.nc
│   │   ├── augment_10_5
│   │   │   ├── scaler_dataset.joblib
│   │   │   ├── test.nc
│   │   │   ├── test_enriched.nc
│   │   │   ├── train.nc
│   │   │   └── train_enriched.nc
│   │   ├── augment_40_5
│   │   │   ├── scaler_dataset.joblib
│   │   │   ├── test.nc
│   │   │   ├── test_enriched.nc
│   │   │   ├── train.nc
│   │   │   └── train_enriched.nc
│   │   ├── augment_50_5
│   │   │   ├── scaler_dataset.joblib
│   │   │   ├── test.nc
│   │   │   ├── test_enriched.nc
│   │   │   ├── train.nc
│   │   │   └── train_enriched.nc
│   │   └── train_index.csv
│   └── raw
│       ├── test.csv
│       └── train.csv
├── docs
├── environment.yml
├── models
├── notebooks
│   ├── data
│   │   ├── concat_data.ipynb
│   │   ├── enrich_csv.ipynb
│   │   ├── enrich_xarray.ipynb
│   │   ├── jupiterdataloader.ipynb
│   │   ├── make_data.ipynb
│   │   ├── normalize_data.ipynb
│   │   └── process_data.ipynb
│   ├── exploration
│   │   ├── EDA.ipynb
│   │   ├── clustering.ipynb
│   │   ├── distance.ipynb
│   │   └── visualization.ipynb
│   └── ml
│       ├── pipeline_ml.ipynb
│       └── xgboost_agg.ipynb
├── references
│   ├── Estimation and Forecasting of Rice Yield Using Phenology-Based Algorithm and Linear Regression Model on Sentinel-II Satellite Data.pdf
│   └── Predicting rice yield at pixel scale through synthetic use of crop and deep learning models with satellite data in South and North Korea.pdf
├── src
│   ├── constants.py
│   ├── data
│   │   ├── datascaler.py
│   │   ├── make_data.py
│   │   ├── make_preprocessing.py
│   │   └── preprocessing.py
│   └── models
│       ├── dataloader.py
│       ├── make_submission.py
│       ├── make_train.py
│       ├── ml
│       │   ├── make_train.py
│       │   └── sweep.yaml
│       ├── model.py
│       ├── sweep.yaml
│       └── trainer.py
├── submissions
└── utils.py
```
