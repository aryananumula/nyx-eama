# Dataset Documentation for Tennis Stroke Classification Project

## Overview
This project utilizes 3D motion data to classify tennis strokes. The datasets included in this project are essential for training and evaluating the models developed for action recognition and biomechanical feedback generation.

## Datasets
1. **3D Motion Data**
   - **Description**: This dataset contains 3D motion capture data of various tennis strokes performed by players. Each stroke is recorded with multiple markers placed on the player's body to capture the motion accurately.
   - **Source**: [Insert source or link to where the data can be accessed]
   - **Format**: The data is provided in CSV format, where each row represents a frame of motion data, and columns correspond to the coordinates of the markers and other relevant features.

2. **Preprocessing Steps**
   - The raw motion data may require preprocessing, including normalization, filtering, and segmentation of strokes. Ensure that the data is cleaned and formatted correctly before using it in the model.

## Usage
- The datasets can be loaded using the `load_data` function from `thetis_data.py`, which reads the CSV files and prepares them for analysis.
- For further exploration and analysis, refer to the Jupyter notebook located in the `notebooks` directory.

## Acknowledgments
- [Include any acknowledgments or references to contributors, datasets, or tools used in the project]