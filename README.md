# Talking Tennis: Causal Language Feedback from 3D Biomechanical Action Recognition

This project aims to develop a pipeline for classifying tennis strokes using 3D motion data. The pipeline incorporates components for data loading, feature extraction, action recognition, and feedback generation using a large language model (LLM).

## Project Structure

- **src/**: Contains the main source code for the project.
  - **data/**: Includes scripts for data handling.
    - `thetis_data.py`: Functions for loading and exploring data from CSV files.
    - `thetis_download.py`: Functions for downloading necessary datasets.
  - **models/**: Contains the machine learning models.
    - `action_recognition.py`: Implements the Vision Transformer (ViT) for stroke classification.
    - `feature_extraction.py`: Functions for extracting biomechanical features from motion data.
  - **feedback/**: Integrates LLM for generating feedback.
    - `llm_feedback.py`: Functions for processing input data and generating feedback.
  - `pipeline.py`: Orchestrates the entire classification pipeline.
  - **utils/**: Utility functions for various tasks.
    - `helpers.py`: Contains helper functions for data preprocessing and visualization.

- **tests/**: Contains unit tests for the project components.
  - `test_thetis_data.py`: Tests for data loading and exploration functions.
  - `test_action_recognition.py`: Tests for the action recognition model.
  - `test_feature_extraction.py`: Tests for feature extraction functions.
  - `test_llm_feedback.py`: Tests for LLM feedback generation functions.

- **data/**: Documentation about the datasets used in the project.
  - `README.md`: Descriptions, sources, and preprocessing steps for datasets.

- **notebooks/**: Jupyter notebooks for exploratory data analysis.
  - `exploration.ipynb`: Visualizations and experiments with modeling approaches.

- **requirements.txt**: Lists dependencies required for the project.

- **README.md**: Overview of the project, setup instructions, and usage guidelines.

- **.gitignore**: Specifies files and directories to be ignored by version control.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd tennis-stroke-classification
   ```

2. Install the required dependencies:
   ```
   pip install -r pyproject.txt
   ```
   

3. Download the necessary datasets using the `thetis_download.py` script. Download the model file from [this google drive](https://drive.google.com/drive/folders/1g345rC3bImH04ZnDhoBwHOTYlUCbCC2c?usp=sharing) and place model.pt in the project directory.

4. Run the pipeline using the `pipeline.py` script to classify tennis strokes and generate feedback.

## Usage Guidelines

- Use the `exploration.ipynb` notebook for exploratory data analysis and visualizations.
- Modify the `pipeline.py` script to customize the workflow as needed.
- Refer to the `dataset-readme/README.md` for detailed information about the datasets used in the project.
