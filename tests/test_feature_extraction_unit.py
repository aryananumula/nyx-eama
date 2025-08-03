"""
Unit tests for feature extraction functionality
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.feature_extraction import (
    extract_features_with_validation,
    get_feature_summary,
    extract_joint_angles,
    extract_racket_dynamics,
    get_joint_mapping,
    ensure_numpy
)

class TestFeatureExtraction:
    """Test class for feature extraction functions"""
    
    def test_ensure_numpy_with_dataframe(self, sample_tennis_data):
        """Test ensure_numpy function with DataFrame input"""
        result = ensure_numpy(sample_tennis_data)
        assert isinstance(result, pd.DataFrame)
    
    def test_ensure_numpy_with_numpy_array(self):
        """Test ensure_numpy function with numpy array input"""
        arr = np.random.randn(10, 3)
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
    
    def test_joint_mapping_detection(self, sample_tennis_data):
        """Test joint mapping functionality"""
        mapping = get_joint_mapping(sample_tennis_data)
        assert isinstance(mapping, dict)
        assert 'shoulder' in mapping  # Should find right_shoulder
        assert 'elbow' in mapping     # Should find right_elbow
    
    def test_feature_extraction_success(self, sample_tennis_data):
        """Test that feature extraction completes successfully"""
        result = extract_features_with_validation(sample_tennis_data)
        assert result['success'] == True
        assert 'features' in result
        assert 'validation' in result
    
    def test_feature_extraction_structure(self, sample_tennis_data):
        """Test that extracted features have expected structure"""
        result = extract_features_with_validation(sample_tennis_data)
        if result['success']:
            features = result['features']
            expected_features = ['joint_angles', 'limb_velocities', 'kinetic_chain', 
                               'racket_dynamics', 'body_rotation', 'timing_features', 
                               'power_generation']
            
            for feature_type in expected_features:
                assert feature_type in features
    
    def test_joint_angles_extraction(self, sample_tennis_data):
        """Test joint angle extraction"""
        angles = extract_joint_angles(sample_tennis_data)
        assert isinstance(angles, dict)
        # Should have at least some angles calculated
        assert len(angles) > 0
        
        # Check that angles are reasonable (0-180 degrees)
        for angle_name, angle_values in angles.items():
            if isinstance(angle_values, np.ndarray) and len(angle_values) > 0:
                valid_angles = angle_values[angle_values > 0]  # Exclude zeros (missing data)
                if len(valid_angles) > 0:
                    assert np.all(valid_angles >= 0)
                    assert np.all(valid_angles <= 180)
    
    def test_racket_dynamics_extraction(self, sample_tennis_data):
        """Test racket dynamics extraction"""
        dynamics = extract_racket_dynamics(sample_tennis_data)
        assert isinstance(dynamics, dict)
        
        # Check expected keys
        expected_keys = ['velocity', 'speed', 'acceleration', 'acceleration_magnitude', 
                        'impact_frame', 'max_velocity']
        for key in expected_keys:
            assert key in dynamics
        
        if 'max_velocity' in dynamics:
            assert dynamics['max_velocity'] >= 0
        
        if 'impact_frame' in dynamics:
            assert isinstance(dynamics['impact_frame'], (int, np.integer))
    
    def test_feature_summary_generation(self, sample_tennis_data):
        """Test feature summary generation"""
        result = extract_features_with_validation(sample_tennis_data)
        if result['success']:
            summary = get_feature_summary(result['features'])
            assert isinstance(summary, dict)
            
            # Check that summary contains expected information
            for feature_type, feature_summary in summary.items():
                if isinstance(feature_summary, dict):
                    # Check that numerical features have statistical summaries
                    for key, value in feature_summary.items():
                        if isinstance(value, dict) and 'shape' in value:
                            assert 'mean' in value
                            assert 'std' in value
                            assert 'min' in value
                            assert 'max' in value
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data"""
        empty_data = pd.DataFrame()
        result = extract_features_with_validation(empty_data)
        # Should handle gracefully, either succeed with empty features or fail cleanly
        assert isinstance(result, dict)
        assert 'success' in result
    
    def test_missing_joints_handling(self):
        """Test handling when key joints are missing"""
        minimal_data = pd.DataFrame({
            'some_joint': [np.random.randn(3) for _ in range(10)]
        })
        result = extract_features_with_validation(minimal_data)
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            # Should still extract some features even with missing joints
            assert 'features' in result
    
    def test_data_validation(self, sample_tennis_data):
        """Test data validation functionality"""
        result = extract_features_with_validation(sample_tennis_data)
        if result['success']:
            validation = result['validation']
            assert 'completeness' in validation
            assert isinstance(validation['completeness'], dict)
            
            # Should check all expected feature types
            expected_features = ['joint_angles', 'limb_velocities', 'kinetic_chain', 
                               'racket_dynamics', 'body_rotation', 'timing_features', 
                               'power_generation']
            for feature_type in expected_features:
                assert feature_type in validation['completeness']

@pytest.fixture
def sample_tennis_data():
    """Fixture providing sample tennis motion data"""
    frames = 50
    data = {
        'right_shoulder': [np.random.randn(3) + [1, 1.5, 0] for _ in range(frames)],
        'left_shoulder': [np.random.randn(3) + [-1, 1.5, 0] for _ in range(frames)],
        'right_elbow': [np.random.randn(3) + [1.5, 1, 0] for _ in range(frames)],
        'right_wrist': [np.random.randn(3) + [2, 0.8, 0] for _ in range(frames)],
        'spine': [np.random.randn(3) + [0, 1.2, 0] for _ in range(frames)],
        'right_hip': [np.random.randn(3) + [0.5, 0.5, 0] for _ in range(frames)],
        'right_knee': [np.random.randn(3) + [0.5, 0, 0] for _ in range(frames)],
        'right_ankle': [np.random.randn(3) + [0.5, -0.5, 0] for _ in range(frames)],
        'racket_tip': [np.random.randn(3) + [2.5, 0.5, 0] for _ in range(frames)]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
