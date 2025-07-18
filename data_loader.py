import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ezc3d
import numpy as np
import pandas as pd


class C3DDataLoader:
    """
    A comprehensive loader for C3D motion capture data files.
    Extracts all data from C3D files and converts to pandas DataFrames.
    """

    def __init__(self, data_directory: str = "data"):
        """
        Initialize the C3D data loader.

        Args:
            data_directory: Path to the directory containing C3D files
        """
        self.data_directory = Path(data_directory)
        self.c3d_files = []
        self.find_c3d_files()

    def find_c3d_files(self) -> None:
        """Find all C3D files in the data directory."""
        self.c3d_files = list(self.data_directory.rglob("*.c3d"))
        print(f"Found {len(self.c3d_files)} C3D files")

    def extract_file_metadata(self, file_path: Path) -> Dict:
        """
        Extract metadata from the file path.

        Args:
            file_path: Path to the C3D file

        Returns:
            Dictionary containing file metadata
        """
        parts = file_path.parts
        metadata = {
            'file_path': str(file_path),
            'filename': file_path.name,
            'file_stem': file_path.stem,
        }

        # Extract trial and condition information from path
        if len(parts) >= 3:
            trial_condition = parts[-3]  # e.g., 'tp3_bhb'
            session = parts[-2]  # e.g., 's5'

            # Parse trial and condition
            if '_' in trial_condition:
                trial_parts = trial_condition.split('_')
                metadata['trial'] = trial_parts[0]  # e.g., 'tp3'
                metadata['condition'] = '_'.join(trial_parts[1:])  # e.g., 'bhb'
            else:
                metadata['trial'] = trial_condition
                metadata['condition'] = 'unknown'

            metadata['session'] = session

        return metadata

    def load_single_c3d(self, file_path: Path) -> Dict:
        """
        Load a single C3D file and extract all data.

        Args:
            file_path: Path to the C3D file

        Returns:
            Dictionary containing all extracted data
        """
        try:
            # Load C3D file
            c3d = ezc3d.c3d(str(file_path))

            # Extract metadata
            metadata = self.extract_file_metadata(file_path)

            # Get basic file information
            metadata.update({
                'frame_rate': c3d['header']['points']['frame_rate'],
                'first_frame': c3d['header']['points']['first_frame'],
                'last_frame': c3d['header']['points']['last_frame'],
                'num_frames': c3d['header']['points']['last_frame'] - c3d['header']['points']['first_frame'] + 1,
                'num_points': c3d['header']['points']['size'],
                'num_analogs': c3d['header']['analogs']['size'],
                'analog_frame_rate': c3d['header']['analogs']['frame_rate'],
            })

            data = {'metadata': metadata}

            # Extract point data (3D coordinates)
            if 'points' in c3d['data'] and c3d['data']['points'].size > 0:
                points_data = c3d['data']['points']  # Shape: (4, num_points, num_frames)
                point_labels = c3d['parameters']['POINT']['LABELS']['value']

                # Create DataFrame for point data
                point_records = []
                num_frames = points_data.shape[2]
                num_points = points_data.shape[1]

                for frame_idx in range(num_frames):
                    frame_num = metadata['first_frame'] + frame_idx
                    for point_idx in range(min(num_points, len(point_labels))):
                        point_label = point_labels[point_idx].strip()
                        x, y, z, confidence = points_data[:, point_idx, frame_idx]
                        point_records.append({
                            'file_path': str(file_path),
                            'trial': metadata.get('trial', ''),
                            'condition': metadata.get('condition', ''),
                            'session': metadata.get('session', ''),
                            'frame': frame_num,
                            'point_label': point_label,
                            'x': x,
                            'y': y,
                            'z': z,
                            'confidence': confidence
                        })

                data['points_df'] = pd.DataFrame(point_records)

            # Extract analog data
            if 'analogs' in c3d['data'] and c3d['data']['analogs'].size > 0:
                analogs_data = c3d['data']['analogs']  # Shape: (1, num_channels, num_samples)
                analog_labels = c3d['parameters']['ANALOG']['LABELS']['value']

                # Create DataFrame for analog data
                analog_records = []
                analog_sample_rate = metadata['analog_frame_rate']

                # Handle different analog data shapes
                if len(analogs_data.shape) == 3:
                    # Shape: (1, num_channels, num_samples)
                    num_samples = analogs_data.shape[2]
                    num_channels = analogs_data.shape[1]

                    for sample_idx in range(num_samples):
                        time = sample_idx / analog_sample_rate
                        for channel_idx in range(min(num_channels, len(analog_labels))):
                            channel_label = analog_labels[channel_idx].strip()
                            value = analogs_data[0, channel_idx, sample_idx]
                            analog_records.append({
                                'file_path': str(file_path),
                                'trial': metadata.get('trial', ''),
                                'condition': metadata.get('condition', ''),
                                'session': metadata.get('session', ''),
                                'sample': sample_idx,
                                'time': time,
                                'channel_label': channel_label,
                                'value': value
                            })
                elif len(analogs_data.shape) == 2:
                    # Shape: (num_channels, num_samples)
                    num_samples = analogs_data.shape[1]
                    num_channels = analogs_data.shape[0]

                    for sample_idx in range(num_samples):
                        time = sample_idx / analog_sample_rate
                        for channel_idx in range(min(num_channels, len(analog_labels))):
                            channel_label = analog_labels[channel_idx].strip()
                            value = analogs_data[channel_idx, sample_idx]
                            analog_records.append({
                                'file_path': str(file_path),
                                'trial': metadata.get('trial', ''),
                                'condition': metadata.get('condition', ''),
                                'session': metadata.get('session', ''),
                                'sample': sample_idx,
                                'time': time,
                                'channel_label': channel_label,
                                'value': value
                            })

                data['analogs_df'] = pd.DataFrame(analog_records)

            # Extract parameter data
            parameters = {}
            for group_name, group_data in c3d['parameters'].items():
                if isinstance(group_data, dict):
                    for param_name, param_data in group_data.items():
                        if isinstance(param_data, dict) and 'value' in param_data:
                            param_key = f"{group_name}_{param_name}"
                            parameters[param_key] = param_data['value']

            data['parameters'] = parameters

            return data

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return {
                'metadata': self.extract_file_metadata(file_path),
                'error': str(e)
            }

    def load_all_c3d_files(self, max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all C3D files and combine into comprehensive DataFrames.

        Args:
            max_files: Maximum number of files to process (for testing)

        Returns:
            Dictionary containing combined DataFrames for all data types
        """
        print("Loading all C3D files...")

        all_points = []
        all_analogs = []
        all_metadata = []
        all_parameters = []
        errors = []

        files_to_process = self.c3d_files[:max_files] if max_files else self.c3d_files

        for i, file_path in enumerate(files_to_process):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing file {i+1}/{len(files_to_process)}: {file_path.name}")

            data = self.load_single_c3d(file_path)

            # Collect metadata
            all_metadata.append(data['metadata'])

            # Collect error information
            if 'error' in data:
                errors.append({
                    'file_path': str(file_path),
                    'error': data['error']
                })
                continue

            # Collect points data
            if 'points_df' in data and not data['points_df'].empty:
                all_points.append(data['points_df'])

            # Collect analog data
            if 'analogs_df' in data and not data['analogs_df'].empty:
                all_analogs.append(data['analogs_df'])

            # Collect parameters
            if 'parameters' in data:
                param_record = data['metadata'].copy()
                param_record.update(data['parameters'])
                all_parameters.append(param_record)

        # Combine all data into DataFrames
        result = {}

        if all_metadata:
            result['metadata'] = pd.DataFrame(all_metadata)
            print(f"Loaded metadata for {len(all_metadata)} files")

        if all_points:
            result['points'] = pd.concat(all_points, ignore_index=True)
            print(f"Loaded {len(result['points']):,} point measurements")

        if all_analogs:
            result['analogs'] = pd.concat(all_analogs, ignore_index=True)
            print(f"Loaded {len(result['analogs']):,} analog measurements")

        if all_parameters:
            result['parameters'] = pd.DataFrame(all_parameters)
            print(f"Loaded parameters for {len(all_parameters)} files")

        if errors:
            result['errors'] = pd.DataFrame(errors)
            print(f"Encountered errors in {len(errors)} files")

        return result

    def get_summary_statistics(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Print summary statistics for the loaded data.

        Args:
            data: Dictionary containing the loaded DataFrames
        """
        print("\n" + "="*80)
        print("DATA SUMMARY STATISTICS")
        print("="*80)

        if 'metadata' in data:
            df = data['metadata']
            print(f"\nFILES LOADED: {len(df)}")
            if 'trial' in df.columns:
                print(f"Unique trials: {df['trial'].nunique()}")
                print(f"Trials: {sorted(df['trial'].unique())}")
            if 'condition' in df.columns:
                print(f"Unique conditions: {df['condition'].nunique()}")
                print(f"Conditions: {sorted(df['condition'].unique())}")
            if 'session' in df.columns:
                print(f"Unique sessions: {df['session'].nunique()}")

            if 'frame_rate' in df.columns:
                print(f"Frame rates: {df['frame_rate'].unique()}")
            if 'num_frames' in df.columns:
                print(f"Frame count range: {df['num_frames'].min()} - {df['num_frames'].max()}")
            if 'num_points' in df.columns:
                print(f"Point count range: {df['num_points'].min()} - {df['num_points'].max()}")

        if 'points' in data:
            df = data['points']
            print(f"\nPOINT DATA: {len(df):,} measurements")
            print(f"Unique points: {df['point_label'].nunique()}")
            print(f"Some point labels: {sorted(df['point_label'].unique())[:20]}")

            # Show coordinate ranges (excluding invalid points)
            valid_points = df[(df['x'] != 0) | (df['y'] != 0) | (df['z'] != 0)]
            if len(valid_points) > 0:
                print(f"X range: {valid_points['x'].min():.2f} to {valid_points['x'].max():.2f}")
                print(f"Y range: {valid_points['y'].min():.2f} to {valid_points['y'].max():.2f}")
                print(f"Z range: {valid_points['z'].min():.2f} to {valid_points['z'].max():.2f}")

        if 'analogs' in data:
            df = data['analogs']
            print(f"\nANALOG DATA: {len(df):,} measurements")
            print(f"Unique channels: {df['channel_label'].nunique()}")
            print(f"Channel labels: {sorted(df['channel_label'].unique())}")

            if 'time' in df.columns:
                print(f"Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds")

        if 'errors' in data:
            df = data['errors']
            print(f"\nERRORS: {len(df)} files had loading errors")
            if len(df) > 0:
                print("Error types:")
                for error in df['error'].unique():
                    count = len(df[df['error'] == error])
                    print(f"  - {error}: {count} files")

    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "output") -> None:
        """
        Save all DataFrames to CSV files.

        Args:
            data: Dictionary containing the loaded DataFrames
            output_dir: Directory to save the CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for data_type, df in data.items():
            if isinstance(df, pd.DataFrame):
                file_path = output_path / f"{data_type}.csv"
                df.to_csv(file_path, index=False)
                print(f"Saved {data_type} to {file_path} ({len(df):,} rows)")


def main():
    """Main function to load and display all C3D data."""
    # Initialize the data loader
    loader = C3DDataLoader()

    if not loader.c3d_files:
        print("No C3D files found in the data directory!")
        return

    # Load all data (you can set max_files for testing with fewer files)
    all_data = loader.load_all_c3d_files()

    # Display summary statistics
    loader.get_summary_statistics(all_data)

    # Print DataFrame information
    print("\n" + "="*80)
    print("DATAFRAME DETAILS")
    print("="*80)

    for data_type, df in all_data.items():
        if isinstance(df, pd.DataFrame):
            print(f"\n{data_type.upper()} DataFrame:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Show first few rows
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())

    # Save data to CSV files
    print("\n" + "="*80)
    print("SAVING DATA TO CSV FILES")
    print("="*80)
    loader.save_data(all_data)

    return all_data


if __name__ == "__main__":
    # Run the main function
    data = main()
