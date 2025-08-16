#!/usr/bin/env python3
"""
THETIS dataset downloader for tennis motion capture data
"""

import os
import zipfile
import requests
from tqdm import tqdm
import pandas as pd
import numpy as np

def download_thetis_data():
    """
    Download THETIS tennis dataset from the URLs file
    """
    # Read URLs
    with open('urls.txt', 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Create output directory
    output_dir = "thetis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {len(urls)} THETIS dataset files...")
    
    downloaded_files = []
    
    for url in tqdm(urls, desc="Downloading THETIS data"):
        filename = url.split('/')[-1]
        filepath = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            downloaded_files.append(filepath)
            continue
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            downloaded_files.append(filepath)
            print(f"Downloaded {filename}")
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    
    print(f"\nDownloaded {len(downloaded_files)} files to {output_dir}/")
    
    # Extract zip files
    extract_thetis_files(downloaded_files)
    
    return downloaded_files

def extract_thetis_files(zip_files):
    """
    Extract downloaded zip files and organize data
    """
    print("Extracting THETIS files...")
    
    extract_dir = "thetis_output/extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    for zip_path in tqdm(zip_files, desc="Extracting"):
        if not zip_path.endswith('.zip'):
            continue
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to subdirectory based on filename
                base_name = os.path.basename(zip_path).replace('.zip', '')
                extract_subdir = os.path.join(extract_dir, base_name)
                os.makedirs(extract_subdir, exist_ok=True)
                
                zip_ref.extractall(extract_subdir)
                print(f"Extracted {base_name}")
                
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")

def load_thetis_motion_data():
    """
    Load and process THETIS motion capture data
    """
    extract_dir = "thetis_output/extracted"
    
    if not os.path.exists(extract_dir):
        print("No extracted THETIS data found. Please download first.")
        return {}
    
    data_samples = {}
    
    # Look for motion capture files
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(('.csv', '.txt', '.c3d')):
                file_path = os.path.join(root, file)
                sample_name = os.path.basename(root) + "_" + file.replace('.csv', '').replace('.txt', '').replace('.c3d', '')
                
                try:
                    # Try to load as CSV first
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        data_samples[sample_name] = df
                        print(f"Loaded {sample_name}: {df.shape}")
                    
                    elif file.endswith('.txt'):
                        # Try loading as space/tab separated
                        df = pd.read_csv(file_path, sep='\s+', header=None)
                        data_samples[sample_name] = df
                        print(f"Loaded {sample_name}: {df.shape}")
                        
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return data_samples

def main():
    """
    Main function to download and prepare THETIS data
    """
    print("THETIS Dataset Downloader")
    print("=" * 40)
    
    # Try to download real THETIS data
    try:
        downloaded_files = download_thetis_data()
        
        # Load the extracted data
        data_samples = load_thetis_motion_data()
        
        if not data_samples:
            print("No motion data found in extracted files.")
            print("Creating sample data for testing...")
        
    except Exception as e:
        print(f"Error downloading THETIS data: {e}")
        print("Creating sample data for testing...")

    print(f"\nDataset ready with {len(data_samples)} samples!")
    return data_samples

if __name__ == "__main__":
    main()
