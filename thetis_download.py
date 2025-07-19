import os

import pandas as pd
import requests
from tqdm.auto import tqdm

repo_owner = "THETIS-dataset"
repo_name = "dataset"
base_folder = "VIDEO_Skelet3D"
local_base = "thetis_data/VIDEO_Skelet3D"

folders = []

# Make sure base local target exists
os.makedirs(local_base, exist_ok=True)
try:
    # Get action subdirectories
    github_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_folder}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(github_api_url, headers=headers)
    folders = [item for item in response.json() if item["type"] == "dir"]

    # For each action subfolder (like 'backhand'), download .avi files
    for folder in tqdm(folders, desc="Subfolders"):
        # build subfolder API path
        folder_name = folder["name"]
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_folder}/{folder_name}"
        r = requests.get(api_url, headers=headers)
        files = [f for f in r.json() if f["name"].endswith(".avi")]
        local_folder = os.path.join(local_base, folder_name)
        os.makedirs(local_folder, exist_ok=True)
        for file in tqdm(files, desc=f"Downloading {folder_name}", leave=False):
            raw_url = file["download_url"]
            dest_path = os.path.join(local_folder, file["name"])
            if not os.path.exists(dest_path):  # skip if already downloaded
                file_content = requests.get(raw_url).content
                with open(dest_path, "wb") as f:
                    f.write(file_content)

    print("All VIDEO_Skelet3D .avi files downloaded to thetis_data/")
except Exception as e:
    print(f"Error fetching data from GitHub: {e}")

# Create thetis_data.csv file
d = {"actor": [], "action": [], "rectype": [], "sequence": [], "file_path": []}
for folder in tqdm(folders, desc="Creating CSV"):
    folder_name = folder["name"]
    local_folder = os.path.join(local_base, folder_name)
    for file_name in os.listdir(local_folder):
        if file_name.endswith(".avi"):
            actor, action, rectype, sequence = file_name.split(".avi")[0].split("_")
            d["actor"].append(actor)
            d["action"].append(action)
            d["rectype"].append(rectype)

            d["sequence"].append(sequence)
            d["file_path"].append(os.path.join(local_folder, file_name))

thetis = pd.DataFrame(data=d)
thetis_output_path = os.path.join("thetis_output", "thetis_data.csv")
print("Creating thetis_data.csv at", thetis_output_path)
os.makedirs(os.path.dirname(thetis_output_path), exist_ok=True)
thetis.to_csv(thetis_output_path, index=False)
print(f"thetis_data.csv created at {thetis_output_path}")
