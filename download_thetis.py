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
import numpy as np
import struct
from pathlib import Path

# Define the joint names in the order they appear in the sample CSVs
JOINT_NAMES = [
    'Head', 'Neck', 'SpineShoulder', 'ShoulderLeft', 'ShoulderRight',
    'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight', 'HandLeft',
    'HandRight', 'SpineBase', 'HipLeft', 'HipRight', 'KneeLeft',
    'KneeRight', 'AnkleLeft', 'AnkleRight', 'FootLeft', 'FootRight'
]

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

def read_c3d_points(c3d_path):
    """
    Read 3D point data from a C3D file using the c3d library
    """
    try:
        import c3d
        import numpy as np
        
        # Read the C3D file
        with open(c3d_path, 'rb') as handle:
            reader = c3d.Reader(handle)
            
            # Get data dimensions from header
            point_labels = [str(label, 'utf-8').strip() for label in reader.point_labels]
            first_frame = reader.header.first_frame
            last_frame = reader.header.last_frame
            n_frames = last_frame - first_frame + 1
            
            # Reset reader to start
            handle.seek(0)
            reader = c3d.Reader(handle)
            
            # Process each frame separately
            all_points = []
            first_frame_points = None
            
            for frame_no, points, analog in reader.read_frames():
                frame_points = []
                
                # points is a numpy array of shape (n_points, 3)
                for point in points:
                    x, y, z = point
                    # Convert numpy float to Python float and handle NaN/inf
                    x = float(x) if not (np.isnan(x) or np.isinf(x)) else 0.0
                    y = float(y) if not (np.isnan(y) or np.isinf(y)) else 0.0
                    z = float(z) if not (np.isnan(z) or np.isinf(z)) else 0.0
                    frame_points.extend([x, y, z])
                
                # Store first frame points to validate structure
                if first_frame_points is None:
                    first_frame_points = frame_points
                    n_points = len(first_frame_points) // 3
                
                # Ensure consistent point count across frames
                if len(frame_points) != len(first_frame_points):
                    print(f"Warning: Frame {frame_no} has inconsistent point count")
                    # Pad with zeros if needed
                    while len(frame_points) < len(first_frame_points):
                        frame_points.extend([0.0, 0.0, 0.0])
                    # Truncate if too long
                    frame_points = frame_points[:len(first_frame_points)]
                
                all_points.append(frame_points)
            
            if not all_points:
                print(f"No frames found in {c3d_path}")
                return None
            
            # Create column names using point labels if available
            n_points = len(first_frame_points) // 3
            columns = []
            for i in range(n_points):
                point_name = point_labels[i] if i < len(point_labels) else f'Point{i+1}'
                for coord in ['X', 'Y', 'Z']:
                    columns.append(f'{point_name}_{coord}')
            
            # Create DataFrame
            df = pd.DataFrame(all_points, columns=columns)
            
            # Add frame numbers as index
            df.index = range(first_frame, first_frame + len(df))
            df.index.name = 'Frame'
            
            return df
            
    except Exception as e:
        print(f"Error reading {c3d_path}: {e}")
        return None

def process_c3d_files():
    """
    Process all C3D files and convert them to CSV format
    """
    output_dir = "thetis_output"  # Main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting C3D files to CSV format...")
    
    # Walk through extracted directory
    for root, dirs, files in os.walk(os.path.join(output_dir, "extracted")):
        for file in tqdm(files, desc="Processing C3D files"):
            if file.endswith('.c3d'):
                # Parse filename parts
                parts = file.replace('.c3d', '').split('_')
                if len(parts) >= 3:  # tp1_bh_s1
                    player = parts[0]
                    stroke = parts[1]
                    
                    # Create CSV filename (without sequence number)
                    csv_name = f"{player}_{stroke}.csv"
                    csv_path = os.path.join(output_dir, csv_name)
                    
                    # Convert C3D to DataFrame
                    c3d_path = os.path.join(root, file)
                    df = read_c3d_points(c3d_path)
                    
                    if df is not None:
                        # If file exists, append; otherwise create new
                        if os.path.exists(csv_path):
                            df.to_csv(csv_path, mode='a', header=False, index=False)
                        else:
                            df.to_csv(csv_path, index=False)
                        print(f"Processed {file} -> {csv_name}")
    
    print("\nAll C3D files have been processed.")
    print("CSV files are available in thetis_output/")

def main():
    """
    Main function to download and prepare THETIS data
    """
    print("THETIS Dataset Downloader")
    print("=" * 40)
    
    try:
        # Download and extract files
        downloaded_files = download_thetis_data()
        
        # Process C3D files into CSV format
        process_c3d_files()
        
        print("\nDataset processing complete!")
        print("Full data CSV files are available in thetis_output/full_data/")
        
    except Exception as e:
        print(f"Error processing THETIS data: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
