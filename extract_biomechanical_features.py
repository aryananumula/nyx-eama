#!/usr/bin/env python3
"""
Biomechanical Feature Extraction for THETIS Dataset

This script extracts comprehensive biomechanical features from the THETIS dataset,
including joint angles, velocities, accelerations, and tennis-specific features
for body points like shoulders, elbows, wrists, etc.

Usage:
    python extract_biomechanical_features.py [--input_dir thetis_output] [--output_dir biomechanical_features]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

from models.biomechanical_features import THETISBiomechanicalFeatures, extract_biomechanical_features_from_file


def find_csv_files(input_dir: str) -> List[str]:
    """
    Find all CSV files in the input directory.
    
    Args:
        input_dir: Directory to search for CSV files
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def process_single_file(file_path: str, output_dir: str, frame_rate: float = 30.0) -> Dict:
    """
    Process a single CSV file and extract biomechanical features.
    
    Args:
        file_path: Path to the CSV file
        output_dir: Directory to save output files
        frame_rate: Frame rate of the data
        
    Returns:
        Dictionary containing processing results
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        # Extract features
        features = extract_biomechanical_features_from_file(
            file_path=file_path,
            output_dir=output_dir,
            frame_rate=frame_rate
        )
        
        # Get summary
        extractor = THETISBiomechanicalFeatures(frame_rate=frame_rate)
        summary = extractor.get_feature_summary(features)
        
        return {
            'file_path': file_path,
            'success': True,
            'features': features,
            'summary': summary,
            'n_frames': len(next(iter(features.values()))) if features else 0
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {
            'file_path': file_path,
            'success': False,
            'error': str(e)
        }


def create_combined_summary(results: List[Dict], output_dir: str) -> None:
    """
    Create a combined summary of all processed files.
    
    Args:
        results: List of processing results
        output_dir: Directory to save the summary
    """
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results to summarize.")
        return
    
    # Combine all summaries
    all_summaries = []
    for result in successful_results:
        summary = result['summary'].copy()
        summary['file'] = os.path.basename(result['file_path'])
        all_summaries.append(summary)
    
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    # Save combined summary
    summary_path = os.path.join(output_dir, 'combined_feature_summary.csv')
    combined_summary.to_csv(summary_path, index=False)
    print(f"\nCombined summary saved to: {summary_path}")
    
    # Create summary statistics
    summary_stats = []
    for result in successful_results:
        file_name = os.path.basename(result['file_path'])
        n_frames = result['n_frames']
        duration = n_frames / 30.0  # Assuming 30 fps
        
        # Get some key statistics
        summary = result['summary']
        
        # Find some key features
        key_features = {}
        for _, row in summary.iterrows():
            if 'shoulder' in row['feature'].lower() and 'angle' in row['feature'].lower():
                key_features['shoulder_angle_mean'] = row['mean']
                key_features['shoulder_angle_max'] = row['max']
            elif 'elbow' in row['feature'].lower() and 'angle' in row['feature'].lower():
                key_features['elbow_angle_mean'] = row['mean']
                key_features['elbow_angle_max'] = row['max']
            elif 'velocity' in row['feature'].lower() and 'magnitude' in row['feature'].lower():
                key_features['max_velocity'] = row['max']
        
        summary_stats.append({
            'file': file_name,
            'n_frames': n_frames,
            'duration_seconds': duration,
            **key_features
        })
    
    stats_df = pd.DataFrame(summary_stats)
    stats_path = os.path.join(output_dir, 'file_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"File statistics saved to: {stats_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}")
    
    if successful_results:
        total_frames = sum(r['n_frames'] for r in successful_results)
        total_duration = total_frames / 30.0
        print(f"Total frames processed: {total_frames:,}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Average frames per file: {total_frames / len(successful_results):.0f}")


def create_feature_visualizations(results: List[Dict], output_dir: str) -> None:
    """
    Create visualizations of the extracted features.
    
    Args:
        results: List of processing results
        output_dir: Directory to save visualizations
    """
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results to visualize.")
        return
    
    # Create feature comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. Shoulder angles comparison
    shoulder_angles = []
    file_names = []
    for result in successful_results:
        features = result['features']
        if 'joint_angles' in features:
            for angle_name, angle_values in features['joint_angles'].items():
                if 'shoulder' in angle_name and 'right' in angle_name:
                    shoulder_angles.append(angle_values)
                    file_names.append(os.path.basename(result['file_path']))
                    break
    
    if shoulder_angles:
        axes[0].boxplot(shoulder_angles, labels=[f.split('_')[0] for f in file_names])
        axes[0].set_title('Right Shoulder Flexion Angles')
        axes[0].set_ylabel('Angle (degrees)')
        axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Elbow angles comparison
    elbow_angles = []
    for result in successful_results:
        features = result['features']
        if 'joint_angles' in features:
            for angle_name, angle_values in features['joint_angles'].items():
                if 'elbow' in angle_name and 'right' in angle_name:
                    elbow_angles.append(angle_values)
                    break
    
    if elbow_angles:
        axes[1].boxplot(elbow_angles, labels=[f.split('_')[0] for f in file_names])
        axes[1].set_title('Right Elbow Flexion Angles')
        axes[1].set_ylabel('Angle (degrees)')
        axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Movement duration comparison
    durations = [r['n_frames'] / 30.0 for r in successful_results]
    file_labels = [os.path.basename(r['file_path']).split('_')[0] for r in successful_results]
    
    axes[2].bar(range(len(durations)), durations)
    axes[2].set_title('Movement Duration by File')
    axes[2].set_ylabel('Duration (seconds)')
    axes[2].set_xticks(range(len(durations)))
    axes[2].set_xticklabels(file_labels, rotation=45)
    
    # 4. Max velocities comparison
    max_velocities = []
    for result in successful_results:
        features = result['features']
        if 'joint_velocities' in features:
            max_vel = 0
            for vel_name, vel_values in features['joint_velocities'].items():
                if 'magnitude' in vel_name:
                    max_vel = max(max_vel, np.max(vel_values))
            max_velocities.append(max_vel)
    
    if max_velocities:
        axes[3].bar(range(len(max_velocities)), max_velocities)
        axes[3].set_title('Maximum Joint Velocities')
        axes[3].set_ylabel('Velocity (units/frame)')
        axes[3].set_xticks(range(len(max_velocities)))
        axes[3].set_xticklabels(file_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'feature_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Feature comparison plot saved to: {plot_path}")
    plt.show()


def main():
    """Main function to process THETIS dataset files."""
    parser = argparse.ArgumentParser(
        description='Extract biomechanical features from THETIS dataset'
    )
    parser.add_argument(
        '--input_dir', 
        default='thetis_output',
        help='Directory containing CSV files (default: thetis_output)'
    )
    parser.add_argument(
        '--output_dir', 
        default='biomechanical_features',
        help='Directory to save extracted features (default: biomechanical_features)'
    )
    parser.add_argument(
        '--frame_rate', 
        type=float, 
        default=30.0,
        help='Frame rate of the motion capture data (default: 30.0)'
    )
    parser.add_argument(
        '--single_file',
        help='Process only a single file (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find CSV files
    if args.single_file:
        if os.path.exists(args.single_file):
            csv_files = [args.single_file]
        else:
            print(f"Error: File '{args.single_file}' does not exist.")
            return
    else:
        csv_files = find_csv_files(args.input_dir)
        if not csv_files:
            print(f"No CSV files found in '{args.input_dir}'")
            return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process files
    results = []
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\nProcessing file {i}/{len(csv_files)}")
        result = process_single_file(csv_file, args.output_dir, args.frame_rate)
        results.append(result)
    
    # Create combined summary
    create_combined_summary(results, args.output_dir)
    
    # Create visualizations
    create_feature_visualizations(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

