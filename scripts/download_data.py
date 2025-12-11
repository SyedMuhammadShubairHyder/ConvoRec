import os
import requests
import zipfile
import io
from tqdm import tqdm

def download_movielens(url, save_path):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if response.status_code == 200:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            with io.BytesIO() as bio:
                for chunk in response.iter_content(chunk_size=1024):
                    bio.write(chunk)
                    pbar.update(len(chunk))
                
                print("\nExtracting...")
                with zipfile.ZipFile(bio) as zip_ref:
                    zip_ref.extractall(save_path)
        print("Done!")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    RAW_DATA_PATH = os.path.join("data", "raw")
    
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    # Check if already exists
    if os.path.exists(os.path.join(RAW_DATA_PATH, "ml-25m")):
        print("Data already exists in data/raw/ml-25m")
    else:
        download_movielens(URL, RAW_DATA_PATH)
