# ICLR Workshop Challenge #2: Radiant Earth Computer Vision for Crop Detection from Satellite Imagery

A Data Science competition on [Zindi](https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition).

1. Create a [RadiantMLHub](https://mlhub.earth/index.html#home) API Key, and store it in your project directory as `api_key.txt`. Add the file to your `.gitignore`.
2. To download raw data (~16 GB on disk): Run `download_data.py`
3. To extract pixel/field-level data: Run `extract_pixel_data.py`
4. To calculate remote sensing indexes: Run `get_features.py`
5. To calculate time series statistics per pixel: Run `get_pixel_stats.py`
6. To train model and make predictions:
    - View script parameters: `python3 main.py --h`
    - Run script with parameters and save submission file under `./submissions/<current_time>-submission.csv`:

        ```python main.py -cv -fd -md RandomForest -ne 100 -cw -rs 123 -sp```

    - Write standard output to file, e.g. `python3 main.py ...args... >> model_notes.txt`

## Docker Image

A docker image is available for a modeling development environment. The image was made to be light weight, and includes only the derived data and training scripts needed to run step 6 above to reproduce my results.

You can run the image by downloading [docker](https://docs.docker.com/) and running the following commands (the `-v` flag binds your local directory with the container so you can access any new submissions files you create):

```
docker pull cambostein/iclr-radiant-crops:1.2
docker run -v "$(pwd)"/submissions:/app/submissions -it cambostein/iclr-radiant-crops:1.2
```

Once the container is running and you are connected via `bash`, follow the instructions above under step 6. You can read more about the project and modeling specifics in the [Project Summary](https://github.com/cameronbronstein/iclr-crop-detection/tree/master/Project%20Summary)
