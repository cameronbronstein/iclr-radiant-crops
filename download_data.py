# code from Radiant ML Hub Docs
import requests
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

def get_download_url(item, asset_key, headers):
    asset = item.get('assets', {}).get(asset_key, None)
    if asset is None:
        print(f"Asset {asset_key} does not exist in this item.")
        return None
    r = requests.get(asset.get('href'), headers=headers, allow_redirects=False)
    return r.headers.get('Location')

def download_label(url, output_path, tileid):
    filename = urlparse(url).path.split('/')[-1]
    outpath = output_path/tileid
    outpath.mkdir(parents=True, exist_ok=True)
    
    r = requests.get(url)
    f = open(outpath/filename, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()
    print(f'Downloaded {filename}')
    return 

def download_imagery(url, output_path, tileid, date):
    filename = urlparse(url).path.split('/')[-1]
    outpath = output_path/tileid/date
    outpath.mkdir(parents=True, exist_ok=True)
    
    r = requests.get(url)
    f = open(outpath/filename, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()
    print(f'Downloaded {filename}')
    return

if __name__ == '__main__':
    output_path = Path("raw_tif_data/")

    # pull in api key - I store mine as a text file in the repo directory.
    with open('./api_key.txt', 'r') as f:
        ACCESS_TOKEN = f.read()[:-1]

    # these headers will be used in each request
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept':'application/json'
    }

    # Download Labels

    # paste the id of the labels collection:
    collectionId = 'ref_african_crops_kenya_02_labels'

    # these optional parameters can be used to control what items are returned. 
    # Here, we want to download all the items so:
    limit = 500
    bounding_box = []
    date_time = []

    # retrieves the items and their metadata in the collection
    r = requests.get(f'https://api.radiant.earth/mlhub/v1/collections/{collectionId}/items', 
                    params={'limit':limit, 'bbox':bounding_box,'datetime':date_time}, 
                    headers=headers)

    if r.status_code == 200:
        collection = r.json()
    else:
        raise ValueError('Faulty API Request!')

    # retrieve list of features (in this case tiles) in the collection
    for feature in collection.get('features', []):
        assets = feature.get('assets').keys()
        print("Feature", feature.get('id'), 'with the following assets', list(assets))

    for feature in collection.get('features', []):
        
        tileid = feature.get('id').split('tile_')[-1][:2]

        # download labels
        download_url = get_download_url(feature, 'labels', headers)
        download_label(download_url, output_path, tileid)
        
        #download field_ids
        download_url = get_download_url(feature, 'field_ids', headers)
        download_label(download_url, output_path, tileid)

    # Download image data - this will take awhile


    # The size of data is about 1.5 GB, and download time depends on your internet connection. 
    # Note that you only need to run this cell and download the data once.
    for feature in collection.get('features', []):
        print('Downloading data from {}'.format(feature['id'][-13:-6]))
        for link in feature.get('links', []):
            if link.get('rel') != 'source':
                continue
		    
            r = requests.get(link['href'], headers=headers)
            feature = r.json()
            assets = feature.get('assets').keys()
            tileid = feature.get('id').split('tile_')[-1][:2]
            date = datetime.strftime(datetime.strptime(feature.get('properties')['datetime'], 
							   "%Y-%m-%dT%H:%M:%SZ"), "%Y%m%d")
            for asset in assets:
                download_url = get_download_url(feature, asset, headers)
                download_imagery(download_url, output_path, tileid, date)    
