"""
This script will read the raw tif data and extract field-level pixel data.

Sorts data into "train" and "test", where train is the labeled fields, and test is the unlabeled prediciton data.
"""

import pandas as pd
import numpy as np
import tifffile as tiff
import os

def load_file(fp):
    """
    Takes a PosixPath object or string filepath
    and returns np array
    """
    return tiff.imread(fp.__str__())

def tiff_to_df(scene_id):
    """
    Converts tiff files to vector (uni-dimensional np array);
    maps array into dataframe.
    
    - Requires data to be stored in directories for each tile (e.g. 00, 01, 02, 03)
    - Requires time series data to be stored in directories for each date.
    """
    ids_file_path = f"./tif_data/{scene_id}/{scene_id[-1]}_field_id.tif"
    labels_file_path = f"./tif_data/{scene_id}/{scene_id[-1]}_label.tif"
    
    ids = load_file(ids_file_path)
    labels = load_file(labels_file_path)
    
    df = pd.DataFrame(columns=['field_id', 'label', 'scene_id'])
    
    df['field_id'] = ids.ravel()
    df['label'] = labels.ravel()
    df['scene_id'] = scene_id
    
    return df

def add_image_data(scene_id, label_df, dates = None, bands=None):
    """
    Adds band data to a dataframe of field ids and field labels
    
    Defaults to all bands, and all dates of the time series.
    """
    if not dates:
        dates = ['20190606', '20190701', '20190706', '20190711', '20190721', '20190805',
                 '20190815', '20190825', '20190909', '20190919', '20190924', '20191004', '20191103']
    
    if not bands:
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
                 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']
        
    for date in dates:
        date_path = f"./tif_data/{scene_id}/{date}"
        for band in bands:
            band_path = date_path + f"/{scene_id[-1]}_{band}_{date}.tif"
            band_data = load_file(band_path).ravel()
        
            label_df[f"{date[4:]}_{band}"] = band_data
    
    return label_df  


if __name__ == "__main__":
    
    scenes = ['00', '01', '02', '03']
              
    labeled_tiles = [(scene, tiff_to_df(scene)) for scene in scenes]
    
    if not os.path.isdir('./csv_data'):
        os.mkdir('./csv_data')

    train = []
    test = []

    for scene, tile in labeled_tiles:
        print(f'Adding image data to scene {scene}...')
        data = add_image_data(scene, tile)
        
        print('Parsing crop labels...')
        field_data = (data['field_id'] != 0)

        labeled = data[field_data]
        
        train.append(labeled.loc[(labeled['label'] != 0), :])
        test.append(labeled.loc[(labeled['label'] == 0), :])
        
        print(f'Scene {scene} data parsed!\n')

    train_data = pd.concat(train, axis=0)
    test_data = pd.concat(test, axis=0)

    train_data.to_csv('./csv_data/train_pixels.csv', index=False)
    test_data.to_csv('./csv_data/test_pixels.csv', index=False)

    print('Pixel data successfully extracted.')