"""
This script calculates the cloud-free statistics for each band feature across the time series.

Stats calculated are: min, max, standard deviation, mean, and median.

Pixels on dates where the modeled cloud probability is above 0 are masked, and not incorporated in the statistic.
"""

import pandas as pd
import numpy as np

train = pd.read_csv('./csv_data/train_derived_features.csv')
test = pd.read_csv('./csv_data/test_derived_features.csv') 

dates = ['0606', '0701', '0706', '0711', '0721', '0805', '0815', '0825', '0909', '0919', '0924', '1004', '1103']
band_features = ['B01', 'B02','B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 
                    'CLD', 'ndvi', 'ndwi_11', 'ndwi_12', 'nirv','evi', 'savi', 'cig']
stats = {'min': np.nanmin, 'max': np.nanmax, 'std': np.nanstd, 'mean': np.nanmean, 'median': np.nanmedian}


train_cube = []
test_cube  = []

for date in dates:
    date_filter = train.columns[train.columns.str.contains(date)]

    train_date = train.loc[:, date_filter]
    test_date = test.loc[:, date_filter]

    # create CLD filter
    train_cloud = (train_date[f'{date}_CLD'] != 0)
    test_cloud = (test_date[f'{date}_CLD'] != 0)

    # convert cloudy pixels to null
    train_date.loc[train_cloud, :] = np.nan
    test_date.loc[test_cloud, :] = np.nan
    
    # append to temporal cube
    train_cube.append(np.array(train_date))
    test_cube.append(np.array(test_date))
    
train_array = np.array(train_cube)
test_array = np.array(test_cube)

for stat, func in stats.items():
    print(f'Calculating {stat} for all bands...')
    train_stat = np.apply_along_axis(func, axis=0, arr=train_array)
    test_stat = np.apply_along_axis(func, axis=0, arr=test_array)

    for idx, band in enumerate(band_features):
        train[f'{band}_{stat}'] = train_stat[:, idx]
        test[f'{band}_{stat}'] = test_stat[:, idx]

train.to_csv('./csv_data/train_stat_features.csv', index=False)
test.to_csv('./csv_data/test_stat_features.csv', index=False)
