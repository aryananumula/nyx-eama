from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models.feature_extraction import extract_joint_angles, extract_limb_velocities, extract_kinetic_chain_patterns, perform_causal_analysis

# Sample data for testing
@pytest.fixture
def sample_data():
    # Create a sample DataFrame to simulate 2D motion data
    data = {
        'joint1': [np.array([x, x + 1]) for x in range(5)],
        'joint2': [np.array([x + 1, x + 2]) for x in range(5)],
        'limb1': [np.array([x, x + 2]) for x in range(5)],
        'limb2': [np.array([x + 2, x + 3]) for x in range(5)],
    }
    return pd.DataFrame(data)

def test_extract_joint_angles(sample_data):
    angles = extract_joint_angles(sample_data)
    assert isinstance(angles, dict)
    assert 'joint1' in angles
    assert len(angles['joint1']) == sample_data.shape[0] - 1

def test_extract_limb_velocities(sample_data):
    velocities = extract_limb_velocities(sample_data)
    assert isinstance(velocities, dict)
    assert 'limb1' in velocities
    assert velocities['limb1'].shape[0] == sample_data.shape[0]

def test_extract_kinetic_chain_patterns(sample_data):
    patterns = extract_kinetic_chain_patterns(sample_data)
    assert isinstance(patterns, dict)
    assert 'joint1' in patterns
    assert len(patterns['joint1']) == sample_data.shape[0] - 1

def test_perform_causal_analysis(sample_data):
    features = {
        'joint_angles': extract_joint_angles(sample_data),
        'limb_velocities': extract_limb_velocities(sample_data),
        'kinetic_patterns': extract_kinetic_chain_patterns(sample_data),
    }
    causal_relationships = perform_causal_analysis(features)
    assert isinstance(causal_relationships, dict)
    assert len(causal_relationships) > 0