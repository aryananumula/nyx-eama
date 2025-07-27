from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import torch

def extract_joint_angles(data):
    data = ensure_numpy(data)  # Ensure compatibility with PyTorch tensors
    angles = {}
    for joint in data.columns:
        if 'joint' in joint:
            # Calculate angles based on joint positions
            angles[joint] = calculate_joint_angle(data[joint])
    return angles

def calculate_joint_angle(joint_data):
    """
    Calculate angles between joints using their 2D positions.
    """
    angles = []
    for i in range(len(joint_data) - 1):
        # Compute vectors between consecutive joints
        vector1 = joint_data[i]
        vector2 = joint_data[i + 1]
        # Calculate angle using dot product and magnitude
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        angles.append(np.degrees(angle))
    return angles

def extract_limb_velocities(data):
    data = ensure_numpy(data)  # Ensure compatibility with PyTorch tensors
    velocities = {}
    for limb in data.columns:
        if 'limb' in limb:
            # Calculate velocities based on limb positions
            velocities[limb] = calculate_limb_velocity(data[limb])
    return velocities

def calculate_limb_velocity(limb_data):
    """
    Calculate velocities of limbs based on their positions over time.
    """
    velocities = np.gradient(limb_data, axis=0)  # Compute gradient along time axis
    return velocities

def extract_kinetic_chain_patterns(data):
    """
    Extract kinetic chain patterns by analyzing sequential activation of joints and limbs.
    """
    data = ensure_numpy(data)  # Ensure compatibility with PyTorch tensors
    patterns = {}
    for joint in data.columns:
        if 'joint' in joint:
            # Analyze sequential activation (placeholder logic)
            patterns[joint] = analyze_kinetic_chain(data[joint])
    return patterns

def analyze_kinetic_chain(joint_data):
    """
    Analyze sequential activation of joints and limbs.
    """
    activation_patterns = []
    for i in range(len(joint_data) - 1):
        # Compute movement magnitude between consecutive frames
        movement_magnitude = np.linalg.norm(joint_data[i + 1] - joint_data[i])
        activation_patterns.append(movement_magnitude)
    return activation_patterns

def perform_causal_analysis(features):
    """
    Identify causal relationships relevant to stroke performance or injury risk.
    """
    causal_relationships = {}
    for feature in features:
        causal_relationships[feature] = analyze_causality(features[feature])
    return causal_relationships

def analyze_causality(feature_data):
    """
    Perform statistical analysis to identify causal relationships.
    """
    # Example: Compute correlation coefficients
    correlations = {}
    for feature1 in feature_data.columns:
        for feature2 in feature_data.columns:
            if feature1 != feature2:
                correlation = np.corrcoef(feature_data[feature1], feature_data[feature2])[0, 1]
                correlations[(feature1, feature2)] = correlation
    return correlations

def ensure_numpy(data):
    """
    Ensures the input data is a numpy array or pandas DataFrame.
    Converts PyTorch tensors to numpy arrays if necessary.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data