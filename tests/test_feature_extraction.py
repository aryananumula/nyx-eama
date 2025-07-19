from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models.feature_extraction import extract_features

# Sample data for testing
@pytest.fixture
def sample_data():
    # Create a sample DataFrame to simulate 3D motion data
    data = {
        'frame': [1, 2, 3, 4, 5],
        'joint_1_x': [0.1, 0.2, 0.3, 0.4, 0.5],
        'joint_1_y': [0.1, 0.1, 0.1, 0.1, 0.1],
        'joint_2_x': [0.2, 0.3, 0.4, 0.5, 0.6],
        'joint_2_y': [0.2, 0.2, 0.2, 0.2, 0.2],
    }
    return pd.DataFrame(data)

def test_extract_features(sample_data):
    features = extract_features(sample_data)
    
    # Check if the features are extracted correctly
    assert 'joint_1_velocity' in features
    assert 'joint_2_velocity' in features
    assert features['joint_1_velocity'].shape[0] == sample_data.shape[0] - 1
    assert features['joint_2_velocity'].shape[0] == sample_data.shape[0] - 1

def test_feature_shape(sample_data):
    features = extract_features(sample_data)
    
    # Ensure the shape of the extracted features is as expected
    assert features.shape[1] == 2  # Assuming two features are extracted
    assert features.shape[0] == sample_data.shape[0] - 1  # One less than input data

def test_feature_values(sample_data):
    features = extract_features(sample_data)
    
    # Check if the values of the features are within expected ranges
    assert np.all(features['joint_1_velocity'] >= 0)
    assert np.all(features['joint_2_velocity'] >= 0)