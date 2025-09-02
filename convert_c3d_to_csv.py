#!/usr/bin/env python3
"""
Convert C3D files to CSV format for THETIS dataset processing.

This script converts C3D files from the extracted directories to CSV format
so that all THETIS dataset files can be processed by the biomechanical feature extraction.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import glob

# Import C3D reading function
from convert_c3d import read_c3d_points


def find_c3d_files(base_dir: str) -> List[str]:
    """
    Find all C3D files in the extracted directories.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of C3D file paths
    """
    c3d_files = []
    extracted_dir = os.path.join(base_dir, "extracted")
    
    if not os.path.exists(extracted_dir):
        print(f"Extracted directory not found: {extracted_dir}")
        return c3d_files
    
    # Search for C3D files recursively
    pattern = os.path.join(extracted_dir, "**", "*.c3d")
    c3d_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(c3d_files)} C3D files")
    return c3d_files


def convert_c3d_to_csv(c3d_file: str, output_dir: str) -> Optional[str]:
    """
    Convert a single C3D file to CSV format.
    
    Args:
        c3d_file: Path to the C3D file
        output_dir: Directory to save the CSV file
        
    Returns:
        Path to the created CSV file, or None if conversion failed
    """
    try:
        print(f"Converting {c3d_file}...")
        
        # Load the C3D file using the existing function
        data = read_c3d_points(c3d_file)
        
        if data is None or data.empty:
            print(f"Warning: No data loaded from {c3d_file}")
            return None
        
        # Generate output filename
        c3d_filename = os.path.basename(c3d_file)
        base_name = os.path.splitext(c3d_filename)[0]
        csv_filename = f"{base_name}.csv"
        output_path = os.path.join(output_dir, csv_filename)
        
        # Save to CSV
        data.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error converting {c3d_file}: {e}")
        return None


def convert_all_c3d_files(input_dir: str = "thetis_output", 
                         output_dir: str = "thetis_output") -> List[str]:
    """
    Convert all C3D files to CSV format.
    
    Args:
        input_dir: Input directory containing extracted C3D files
        output_dir: Output directory for CSV files
        
    Returns:
        List of created CSV file paths
    """
    # Find all C3D files
    c3d_files = find_c3d_files(input_dir)
    
    if not c3d_files:
        print("No C3D files found to convert")
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each C3D file
    converted_files = []
    for c3d_file in c3d_files:
        csv_file = convert_c3d_to_csv(c3d_file, output_dir)
        if csv_file:
            converted_files.append(csv_file)
    
    print(f"Successfully converted {len(converted_files)} files")
    return converted_files


def main():
    """Main function to handle command line arguments and run conversion."""
    parser = argparse.ArgumentParser(description="Convert C3D files to CSV format")
    parser.add_argument("--input_dir", default="thetis_output", 
                       help="Input directory containing extracted C3D files")
    parser.add_argument("--output_dir", default="thetis_output", 
                       help="Output directory for CSV files")
    parser.add_argument("--single_file", 
                       help="Convert a single C3D file")
    
    args = parser.parse_args()
    
    if args.single_file:
        # Convert a single file
        if not os.path.exists(args.single_file):
            print(f"File not found: {args.single_file}")
            return
        
        csv_file = convert_c3d_to_csv(args.single_file, args.output_dir)
        if csv_file:
            print(f"Successfully converted: {csv_file}")
        else:
            print("Conversion failed")
    else:
        # Convert all files
        converted_files = convert_all_c3d_files(args.input_dir, args.output_dir)
        print(f"Conversion complete. {len(converted_files)} files converted.")


if __name__ == "__main__":
    main()
