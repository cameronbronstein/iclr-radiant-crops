# ICLR Workshop Challenge #2: Radiant Earth Computer Vision for Crop Detection from Satellite Imagery

A Data Science competition on [Zindi](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition).

1. Create a [RadiantMLHub](https://mlhub.earth/index.html#home) API Key, and store it in your project directory as `api_key.txt`. Add the file to your `.gitignore`.
2. To download raw data (~ 16 GB on disk): Run `download_data.py`
3. To extract pixel/field-level data: Run `extract_pixel_data.py`
4. To calculate remote sensing indexes: Run `get_features.py`
5. To calculate time series statistics per pixel: Run `get_pixel_stats.py`
6. To train model and make predictions:
    - View script parameters: `python3 main.py --h`
    - Run script with parameters and save submission file under `./submissions/<save_path>.csv`:

        ```python3 main.py -cv True -bs False -md RandomForest -ne 500 -rs 123 -sp SampleSubmission```

    - Write standard output to file, e.g. `python3 main.py ...args... >> model_notes.txt`

For more about this project, read the [Project Summary](https://github.com/cameronbronstein/iclr-crop-detection/tree/master/Project%20Summary)
