import pandas as pd
import numpy as np

# NDVI
def get_NDVI(
    data,
    date
):
    """
    Create NDVI from raw imagery data

    NDVI = NIR - Red / NIR + RED
    NIR = Band 8
    Red = Band 4

    Returns NDVI as pandas series
    """
    nir = data[f'{date}_B08']
    red = data[f'{date}_B04']
    return (nir - red) / (nir + red)

# nirv   
def get_NirV(
    data,
    date
):
    """
    Create NirV from raw imagery data

    NirV = NDVI * NIR
    NIR = Band 8
    Red = Band 4

    Returns NirV as pandas series
    """
    nir = data[f'{date}_B08']
    red = data[f'{date}_B04']
    return ((nir - red) / (nir + red)) * nir
    
# ndwi - 11
def get_NDWI_11(
    data,
    date
):
    """
    Create Normalized Difference Water Index from raw imagery data
    - This version of NDWI uses B11 of SWIR from Sentinel-2.
    - Source: 

    NDWI = NIR - SWIR(11) / NIR + SWIR(11)
    NIR = Band 8
    SWIR = B11

    Returns NDWI-11 as pandas series
    """
    nir = data[f'{date}_B08']
    swir_11 = data[f'{date}_B11']

    return (nir - swir_11) / (nir + swir_11)

def get_NDWI_12(
    data,
    date
):
    """
    Create Normalized Difference Water Index from raw imagery data
    - This version of NDWI uses B12 of SWIR from Sentinel-2.
    - Source: 

    NDWI = NIR - SWIR(12) / NIR + SWIR(12)
    NIR = Band 8
    SWIR = B12

    Returns NDWI-12 as pandas series
    """
    nir = data[f'{date}_B08']
    swir_12 = data[f'{date}_B12']

    return (nir - swir_12) / (nir + swir_12)

# EVI
def get_EVI(
    data,
    date
):
    """
    Create Enhanced Vegetation Index from raw imagery data
    
    𝐸𝑉𝐼 = 2.5 * 𝑁𝐼𝑅−𝑅𝐸𝐷 / 𝑁𝐼𝑅 +(6 * 𝑅𝐸𝐷) − (7.5 * 𝐵𝐿𝑈𝐸) + 1
    
    Returns EVI as pandas series
    """
    nir = data[f'{date}_B08'] 
    red = data[f'{date}_B04']
    blue = data[f'{date}_B02']

    return 2.5 * (nir - red) / (nir + (6 * red) - (7.6 * blue) + 1)
    
# savi
def get_SAVI(
    data,
    date
):
    """
    Create Soil-Adjusted Vegetation Index from raw imagery data

    𝑆𝐴𝑉𝐼 = (1+𝐿) * (𝑁𝐼𝑅−𝑅𝑒𝑑) / (𝑁𝐼𝑅 + 𝑅𝑒𝑑 + 𝐿); where L = 0.5

    Returns EVI as pandas series
    """
    L = 0.5
    nir = data[f'{date}_B08']
    red = data[f'{date}_B04']

    return (1.5 * (nir - red)) / (nir + red + L)
    
# CIg
def get_CIG(
    data,
    date
):
    """
    Create Chlorophyll Index-Green

    𝐶𝐼𝑔 = (𝑁𝐼𝑅 / 𝐺𝑟𝑒𝑒𝑛) − 1

    Return 𝐶𝐼𝑔 as pandas series
    """
    nir = data[f'{date}_B08']
    green = data[f'{date}_B03']

    return (nir / green) - 1
    

if __name__ == '__main__':

    train = pd.read_csv('./csv_data/train_pixels.csv')
    test = pd.read_csv('./csv_data/test_pixels.csv')

    dates = ['0606', '0701', '0706', '0711', '0721', '0805', '0815', '0825', '0909', '0919', '0924', '1004', '1103']

    indices = [('ndvi', get_NDVI), 
               ('nirv', get_NirV), 
               ('ndwi_11', get_NDWI_11), 
               ('ndwi_12', get_NDWI_12), 
               ('evi', get_EVI), 
               ('savi', get_SAVI), 
               ('cig', get_CIG)
               ]

    for date in dates:
        for col, feature in indices:
            train[f'{date}_{col}'] = feature(train, date)
            test[f'{date}_{col}'] = feature(test, date)

    train.to_csv('./csv_data/train_w_indices.csv', index=False)
    test.to_csv('./csv_data/test_w_indices.csv', index=False)   