# ICLR Workshop Challenge #2: Radiant Earth Computer Vision for Crop Detection from Satellite Imagery

A Data Science competition on [Zindi](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition).

# To Reproduce

1. To download image tifs: Run `download_data.py`
2. To convert to pixel/field-level data: Run `extract_pixel_data.py`
3. To calculate remote sensing indexes: Run `get_features.py`
4. To calculate time series statistics per pixel: Run `get_pixel_stats.py`
5. To train model and make predictions: Run `main.py`

# Exploratory Data Analysis

- `/notebooks/01-visualize-data.ipynb`
- `/notebooks/02-eda.ipynb`