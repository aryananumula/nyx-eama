"""Loading data from the 3DTennisDS dataset."""
import os
import shutil
import zipfile

import requests

URLS_FILE = "urls.txt"
ZIP_DIR = "temp_zips"
EXTRACT_DIR = "data"

def download_zip(url, dest_folder):
    local_filename = os.path.join(dest_folder, url.split("/")[-1] or "file.zip")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded: {local_filename}")
        return local_filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted {zip_path} to {extract_to}")
    except Exception as e:
        print(f"Failed to extract {zip_path}: {e}")

def download():
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with open(URLS_FILE) as f:
        urls = [line.strip() for line in f if line.strip()]

    zip_files = []
    for url in urls:
        zip_file = download_zip(url, ZIP_DIR)
        if zip_file:
            zip_files.append(zip_file)

    for zip_file in zip_files:
        extract_zip(zip_file, EXTRACT_DIR)
        os.remove(zip_file)

    shutil.rmtree(ZIP_DIR, ignore_errors=True)
    print("All done.")


# Only run download() if EXTRACT_DIR ("data/") does not already exist
if not os.path.exists(EXTRACT_DIR):
    download()
else:
    print(f'"{EXTRACT_DIR}" already exists. No action taken. Delete it to re-download the dataset.')
