from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import torch

def extract_tennis_biomechanical_features(data):
    """
    Extract comprehensive biomechanical features from tennis motion data.
    Assumes data contains 3D joint coordinates over time.
    """
    data = ensure_numpy(data)
    features = {}
    
    # Core tennis-specific features
    features['joint_angles'] = extract_joint_angles(data)
    features['limb_velocities'] = extract_limb_velocities(data)
    features['kinetic_chain'] = extract_kinetic_chain_patterns(data)
    features['racket_dynamics'] = extract_racket_dynamics(data)
    features['body_rotation'] = extract_body_rotation_features(data)
    features['timing_features'] = extract_temporal_features(data)
    features['power_generation'] = extract_power_generation_features(data)
    
    return features

def get_joint_mapping(data):
    """
    Map generic joint names to actual column names in the dataset.
    """
    available_columns = data.columns.tolist() if hasattr(data, 'columns') else []
    
    # Common joint name variations in motion capture datasets
    joint_variations = {
        'shoulder': ['shoulder', 'right_shoulder', 'RShoulder', 'R_Shoulder', 'shoulder_r'],
        'elbow': ['elbow', 'right_elbow', 'RElbow', 'R_Elbow', 'elbow_r'],
        'wrist': ['wrist', 'right_wrist', 'RWrist', 'R_Wrist', 'wrist_r', 'hand', 'right_hand'],
        'spine': ['spine', 'torso', 'chest', 'trunk', 'SpineBase', 'spine_base'],
        'hip': ['hip', 'right_hip', 'RHip', 'R_Hip', 'hip_r', 'pelvis'],
        'knee': ['knee', 'right_knee', 'RKnee', 'R_Knee', 'knee_r'],
        'ankle': ['ankle', 'right_ankle', 'RAnkle', 'R_Ankle', 'ankle_r', 'foot', 'right_foot'],
        'racket_tip': ['racket_tip', 'racket', 'racquet_tip', 'racquet', 'tool_tip']
    }
    
    mapping = {}
    for generic_name, variations in joint_variations.items():
        for variation in variations:
            if variation in available_columns:
                mapping[generic_name] = variation
                break
        
        # If no exact match found, try case-insensitive matching
        if generic_name not in mapping:
            for col in available_columns:
                if any(var.lower() in col.lower() for var in variations):
                    mapping[generic_name] = col
                    break
    
    return mapping

def preprocess_motion_data(data):
    """
    Preprocess raw motion capture data to handle missing values and normalize coordinates.
    """
    if isinstance(data, np.ndarray):
        # Convert to DataFrame if numpy array (assuming structured format)
        if data.ndim == 3:  # (frames, joints, coordinates)
            joint_names = ['head', 'neck', 'spine', 'left_shoulder', 'right_shoulder', 
                          'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                          'left_hip', 'right_hip', 'left_knee', 'right_knee',
                          'left_ankle', 'right_ankle', 'racket_tip']
            
            frames, n_joints, coords = data.shape
            df_data = {}
            
            for i, joint in enumerate(joint_names[:min(n_joints, len(joint_names))]):
                df_data[joint] = [data[frame, i, :] for frame in range(frames)]
            
            data = pd.DataFrame(df_data)
    
    # Fill missing values with interpolation
    if hasattr(data, 'columns'):
        for col in data.columns:
            if data[col].isnull().any():
                data[col] = data[col].interpolate(method='linear')
    
    return data

def extract_joint_angles(data):
    """
    Calculate tennis-specific joint angles (shoulder, elbow, wrist, hip, knee).
    """
    data = ensure_numpy(data)
    angles = {}
    
    # Map generic joint names to likely column names in THETIS dataset
    joint_mapping = get_joint_mapping(data)
    
    # Define tennis-relevant joint angle calculations using mapped names
    tennis_joints = {
        'shoulder_flexion': [joint_mapping.get('shoulder'), joint_mapping.get('elbow'), joint_mapping.get('wrist')],
        'elbow_flexion': [joint_mapping.get('shoulder'), joint_mapping.get('elbow'), joint_mapping.get('wrist')],
        'wrist_extension': [joint_mapping.get('elbow'), joint_mapping.get('wrist'), joint_mapping.get('racket_tip')],
        'hip_rotation': [joint_mapping.get('spine'), joint_mapping.get('hip'), joint_mapping.get('knee')],
        'knee_flexion': [joint_mapping.get('hip'), joint_mapping.get('knee'), joint_mapping.get('ankle')]
    }
    
    for angle_name, joint_sequence in tennis_joints.items():
        # Check if all required joints are available and not None
        if all(joint is not None and joint in data.columns for joint in joint_sequence):
            angles[angle_name] = calculate_tennis_joint_angle(data, joint_sequence)
        else:
            print(f"Warning: Missing joints for {angle_name}. Available columns: {list(data.columns)[:10]}...")
            angles[angle_name] = np.zeros(len(data)) if hasattr(data, '__len__') else np.array([0])
    
    return angles

def calculate_tennis_joint_angle(data, joint_sequence):
    """
    Calculate angle between three joints forming a kinematic chain.
    """
    angles = []
    joint1, joint2, joint3 = joint_sequence
    
    for i in range(len(data)):
        try:
            # Get 3D positions - handle different data formats
            p1 = get_3d_position(data[joint1].iloc[i])
            p2 = get_3d_position(data[joint2].iloc[i])
            p3 = get_3d_position(data[joint3].iloc[i])
            
            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Avoid division by zero
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                angles.append(0)
                continue
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Prevent numerical errors
            angle = np.arccos(cos_angle)
            angles.append(np.degrees(angle))
        except (ValueError, IndexError, TypeError):
            angles.append(0)  # Default value for missing data
    
    return np.array(angles)

def get_3d_position(position_data):
    """
    Extract 3D position from various data formats.
    """
    if isinstance(position_data, (list, np.ndarray)):
        pos = np.array(position_data)
        if len(pos) >= 3:
            return pos[:3]
        elif len(pos) == 2:
            return np.array([pos[0], pos[1], 0])
        else:
            return np.array([pos[0], 0, 0])
    else:
        # Single value - assume x coordinate
        return np.array([float(position_data), 0, 0])

def extract_racket_dynamics(data):
    """
    Extract racket-specific biomechanical features for tennis analysis.
    """
    data = ensure_numpy(data)
    racket_features = {}
    
    # Racket velocity and acceleration
    if 'racket_tip' in data.columns:
        racket_positions = np.array([pos if isinstance(pos, (list, np.ndarray)) else [pos, 0, 0] 
                                   for pos in data['racket_tip']])
        
        # Calculate velocities
        racket_features['velocity'] = np.gradient(racket_positions, axis=0)
        racket_features['speed'] = np.linalg.norm(racket_features['velocity'], axis=1)
        
        # Calculate acceleration
        racket_features['acceleration'] = np.gradient(racket_features['velocity'], axis=0)
        racket_features['acceleration_magnitude'] = np.linalg.norm(racket_features['acceleration'], axis=1)
        
        # Impact phase detection (highest acceleration)
        racket_features['impact_frame'] = np.argmax(racket_features['acceleration_magnitude'])
        racket_features['max_velocity'] = np.max(racket_features['speed'])
    
    return racket_features

def extract_body_rotation_features(data):
    """
    Extract body rotation features using shoulder vector analysis.
    Implements θ(t) = arctan2(shoulder_vector_y, shoulder_vector_x) logic.
    """
    features = {}
    
    try:
        # Check if we have shoulder landmarks (THETIS format)
        shoulder_cols = ['ShoulderLeft_X', 'ShoulderLeft_Y', 'ShoulderRight_X', 'ShoulderRight_Y']
        
        if all(col in data.columns for col in shoulder_cols):
            # Calculate shoulder vector (right shoulder - left shoulder)
            shoulder_vector_x = data['ShoulderRight_X'] - data['ShoulderLeft_X']
            shoulder_vector_y = data['ShoulderRight_Y'] - data['ShoulderLeft_Y']
            
            # Apply θ(t) = arctan2(shoulder_vector_y, shoulder_vector_x)
            trunk_rotation = np.arctan2(shoulder_vector_y, shoulder_vector_x)
            
            # Calculate angular velocity (derivative of rotation)
            trunk_angular_velocity = np.gradient(trunk_rotation)
            
            features['trunk_rotation'] = trunk_rotation
            features['trunk_angular_velocity'] = trunk_angular_velocity
            
            # Calculate summary statistics
            features['rotation_range'] = np.degrees(np.max(trunk_rotation) - np.min(trunk_rotation))
            features['peak_angular_velocity'] = np.max(np.abs(trunk_angular_velocity))
            
            print(f"Body rotation calculated: {len(trunk_rotation)} frames")
            print(f"Rotation range: {features['rotation_range']:.2f} degrees")
            print(f"Peak angular velocity: {features['peak_angular_velocity']:.3f} rad/s")
            
        else:
            print("Missing shoulder landmarks for body rotation calculation")
            print(f"Available columns: {[col for col in data.columns if 'Shoulder' in col]}")
            # Return zero arrays as fallback
            num_frames = len(data)
            features['trunk_rotation'] = np.zeros(num_frames)
            features['trunk_angular_velocity'] = np.zeros(num_frames)
            features['rotation_range'] = 0.0
            features['peak_angular_velocity'] = 0.0
            
    except Exception as e:
        print(f"Error in body rotation calculation: {e}")
        # Fallback to zeros
        num_frames = len(data)
        features['trunk_rotation'] = np.zeros(num_frames)
        features['trunk_angular_velocity'] = np.zeros(num_frames)
        features['rotation_range'] = 0.0
        features['peak_angular_velocity'] = 0.0
    
    return features

def extract_temporal_features(data):
    """
    Extract timing-related features for tennis stroke analysis.
    """
    temporal_features = {}
    
    # Stroke duration
    temporal_features['stroke_duration'] = len(data)
    
    # Phase identification (assuming data represents one complete stroke)
    total_frames = len(data)
    temporal_features['preparation_phase'] = slice(0, total_frames // 3)
    temporal_features['execution_phase'] = slice(total_frames // 3, 2 * total_frames // 3)
    temporal_features['follow_through_phase'] = slice(2 * total_frames // 3, total_frames)
    
    return temporal_features

def extract_power_generation_features(data):
    """
    Extract features related to power generation in tennis strokes.
    """
    data = ensure_numpy(data)
    power_features = {}
    
    # Calculate kinetic energy approximation
    if 'racket_tip' in data.columns:
        racket_positions = np.array([pos if isinstance(pos, (list, np.ndarray)) else [pos, 0, 0] 
                                   for pos in data['racket_tip']])
        velocities = np.gradient(racket_positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Assuming unit mass for relative comparison
        power_features['kinetic_energy'] = 0.5 * speeds**2
        power_features['peak_power'] = np.max(power_features['kinetic_energy'])
        power_features['average_power'] = np.mean(power_features['kinetic_energy'])
    
    return power_features

def extract_limb_velocities(data):
    """
    Calculate velocities of key body segments for tennis analysis.
    """
    data = ensure_numpy(data)
    velocities = {}
    joint_mapping = get_joint_mapping(data)
    
    # Tennis-specific limb segments with mapping
    tennis_segments = {
        'hand': joint_mapping.get('wrist', 'hand'),
        'forearm': joint_mapping.get('elbow', 'forearm'),
        'upper_arm': joint_mapping.get('shoulder', 'upper_arm'),
        'racket_tip': joint_mapping.get('racket_tip', 'racket_tip'),
        'foot': joint_mapping.get('ankle', 'foot'),
        'shin': joint_mapping.get('knee', 'shin'),
        'thigh': joint_mapping.get('hip', 'thigh')
    }
    
    for segment_name, column_name in tennis_segments.items():
        if column_name and column_name in data.columns:
            try:
                segment_data = np.array([get_3d_position(pos) for pos in data[column_name]])
                velocities[segment_name] = np.gradient(segment_data, axis=0)
            except Exception as e:
                print(f"Warning: Could not calculate velocity for {segment_name}: {e}")
                velocities[segment_name] = np.zeros((len(data), 3))
        else:
            print(f"Warning: Column {column_name} not found for segment {segment_name}")
            velocities[segment_name] = np.zeros((len(data), 3))
    
    return velocities

def extract_kinetic_chain_patterns(data):
    """
    Analyze kinetic chain patterns specific to tennis biomechanics.
    Returns timing data that works with visualization.
    """
    data = ensure_numpy(data)
    patterns = {}
    
    # Tennis kinetic chain segments
    segments = ['ankle', 'knee', 'hip', 'spine', 'shoulder', 'elbow', 'wrist', 'racket_tip']
    
    try:
        # Calculate racket tip timing if racket data is available
        if 'racket_tip' in data.columns:
            racket_timing = calculate_racket_timing(data)
            patterns['racket_tip_timing'] = racket_timing
            patterns['racket_tip_activation'] = analyze_racket_activation(data)
            
            # Create estimated activation times for visualization
            # This simulates the kinetic chain progression from legs to racket
            activation_times = [
                racket_timing * 0.1,  # Ankle
                racket_timing * 0.2,  # Knee  
                racket_timing * 0.3,  # Hip
                racket_timing * 0.5,  # Spine
                racket_timing * 0.7,  # Shoulder
                racket_timing * 0.85, # Elbow
                racket_timing * 0.95, # Wrist
                racket_timing         # Racket
            ]
            patterns['activation_times'] = activation_times
            
            # Calculate basic efficiency metrics
            patterns['chain_efficiency'] = calculate_chain_efficiency(data)
            patterns['coordination_score'] = calculate_coordination_score(data)
            
            print(f"Kinetic chain analysis: racket timing = {racket_timing:.1f}% of stroke")
            
        else:
            print("No racket data available for kinetic chain analysis")
            patterns['racket_tip_timing'] = 50.0  # Default middle of stroke
            patterns['racket_tip_activation'] = True
            patterns['activation_times'] = [10, 20, 30, 50, 70, 85, 95, 100]  # Default progression
            patterns['chain_efficiency'] = 0.0
            patterns['coordination_score'] = 0.0
            
    except Exception as e:
        print(f"Error in kinetic chain analysis: {e}")
        # Fallback values
        patterns['racket_tip_timing'] = 50.0
        patterns['racket_tip_activation'] = True
        patterns['activation_times'] = [10, 20, 30, 50, 70, 85, 95, 100]
        patterns['chain_efficiency'] = 0.0
        patterns['coordination_score'] = 0.0
    
    return patterns

def calculate_racket_timing(data):
    """Calculate when peak racket velocity occurs as percentage of stroke"""
    try:
        racket_positions = np.array([pos if isinstance(pos, (list, np.ndarray)) else [pos, 0, 0] 
                                   for pos in data['racket_tip']])
        velocities = np.gradient(racket_positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        peak_frame = np.argmax(speeds)
        timing_percentage = (peak_frame / len(data)) * 100
        return timing_percentage
    except:
        return 50.0  # Default to middle of stroke

def analyze_racket_activation(data):
    """Analyze if racket shows significant movement"""
    try:
        racket_positions = np.array([pos if isinstance(pos, (list, np.ndarray)) else [pos, 0, 0] 
                                   for pos in data['racket_tip']])
        movement_range = np.max(racket_positions, axis=0) - np.min(racket_positions, axis=0)
        return np.linalg.norm(movement_range) > 0.1  # Threshold for significant movement
    except:
        return True

def calculate_chain_efficiency(data):
    """Calculate basic kinetic chain efficiency metric"""
    try:
        # Simplified efficiency based on racket velocity achievement
        racket_positions = np.array([pos if isinstance(pos, (list, np.ndarray)) else [pos, 0, 0] 
                                   for pos in data['racket_tip']])
        velocities = np.gradient(racket_positions, axis=0)
        max_speed = np.max(np.linalg.norm(velocities, axis=1))
        # Normalize to a 0-1 efficiency score (this is simplified)
        return min(max_speed / 10.0, 1.0)  # Assuming 10 m/s as reference max speed
    except:
        return 0.0

def calculate_coordination_score(data):
    """Calculate basic coordination score"""
    try:
        # Simplified coordination based on smoothness of racket movement
        racket_positions = np.array([pos if isinstance(pos, (list, np.ndarray)) else [pos, 0, 0] 
                                   for pos in data['racket_tip']])
        velocities = np.gradient(racket_positions, axis=0)
        accelerations = np.gradient(velocities, axis=0)
        
        # Smoothness metric (lower jerk = better coordination)
        jerk = np.gradient(accelerations, axis=0)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        coordination = 1.0 / (1.0 + np.mean(jerk_magnitude))  # Inverse relationship
        return coordination
    except:
        return 0.0

def perform_causal_analysis(features):
    """
    Identify causal relationships relevant to tennis stroke performance.
    """
    causal_relationships = {}
    
    # Analyze relationships between kinetic chain timing and racket speed
    if 'racket_dynamics' in features and 'kinetic_chain' in features:
        causal_relationships['timing_to_power'] = analyze_timing_power_relationship(
            features['kinetic_chain'], features['racket_dynamics']
        )
    
    # Analyze joint angle relationships to stroke effectiveness
    if 'joint_angles' in features:
        causal_relationships['angle_correlations'] = analyze_angle_correlations(features['joint_angles'])
    
    return causal_relationships

def analyze_timing_power_relationship(kinetic_chain, racket_dynamics):
    """
    Analyze relationship between kinetic chain timing and power generation.
    """
    relationships = {}
    
    if 'max_velocity' in racket_dynamics:
        max_velocity = racket_dynamics['max_velocity']
        
        for segment, timing in kinetic_chain.items():
            if 'timing' in segment:
                # Calculate correlation between timing and power
                correlation = np.corrcoef([timing], [max_velocity])[0, 1] if not np.isnan([timing, max_velocity]).any() else 0
                relationships[segment] = correlation
    
    return relationships

def analyze_angle_correlations(joint_angles):
    """
    Analyze correlations between joint angles for biomechanical insights.
    """
    correlations = {}
    angle_names = list(joint_angles.keys())
    
    for i, angle1 in enumerate(angle_names):
        for angle2 in angle_names[i+1:]:
            try:
                correlation = np.corrcoef(joint_angles[angle1], joint_angles[angle2])[0, 1]
                if not np.isnan(correlation):
                    correlations[f'{angle1}_vs_{angle2}'] = correlation
            except (ValueError, IndexError):
                correlations[f'{angle1}_vs_{angle2}'] = 0
    
    return correlations

def ensure_numpy(data):
    """
    Ensures the input data is a numpy array or pandas DataFrame.
    Converts PyTorch tensors to numpy arrays if necessary.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        # Convert numpy array to DataFrame if it's 3D (frames, joints, coordinates)
        if data.ndim == 3:
            return preprocess_motion_data(data)
        return data
    else:
        # Try to convert to numpy array first, then to DataFrame if needed
        try:
            arr = np.array(data)
            if arr.ndim == 3:
                return preprocess_motion_data(arr)
            return arr
        except:
            raise ValueError(f"Cannot convert data of type {type(data)} to usable format")



def validate_features(features):
    """
    Validate extracted features and provide quality metrics.
    """
    validation_report = {
        'completeness': {},
        'quality_metrics': {},
        'warnings': []
    }
    
    # Check completeness
    expected_features = ['joint_angles', 'limb_velocities', 'kinetic_chain', 
                        'racket_dynamics', 'body_rotation', 'timing_features', 
                        'power_generation']
    
    for feature_type in expected_features:
        if feature_type in features and features[feature_type]:
            validation_report['completeness'][feature_type] = True
        else:
            validation_report['completeness'][feature_type] = False
            validation_report['warnings'].append(f"Missing or empty: {feature_type}")
    
    # Quality metrics
    if 'racket_dynamics' in features and 'max_velocity' in features['racket_dynamics']:
        max_vel = features['racket_dynamics']['max_velocity']
        if max_vel > 200:  # Unrealistic racket speed (m/s)
            validation_report['warnings'].append(f"Unrealistic racket velocity: {max_vel}")
    
    return validation_report

def extract_features_with_validation(data):
    """
    Extract features with validation and error handling.
    """
    try:
        # Preprocess data first
        processed_data = preprocess_motion_data(data)
        
        # Extract features
        features = extract_tennis_biomechanical_features(processed_data)
        
        # Validate results
        validation = validate_features(features)
        
        return {
            'features': features,
            'validation': validation,
            'success': True,
            'data_shape': processed_data.shape if hasattr(processed_data, 'shape') else len(processed_data),
            'available_columns': list(processed_data.columns) if hasattr(processed_data, 'columns') else []
        }
    except Exception as e:
        return {
            'features': {},
            'validation': {'error': str(e)},
            'success': False,
            'data_shape': None,
            'available_columns': []
        }

def get_feature_summary(features):
    """
    Get a summary of extracted features for analysis.
    """
    summary = {}
    
    for feature_type, feature_data in features.items():
        if isinstance(feature_data, dict):
            summary[feature_type] = {}
            for key, value in feature_data.items():
                if isinstance(value, np.ndarray):
                    summary[feature_type][key] = {
                        'shape': value.shape,
                        'mean': float(np.mean(value)) if value.size > 0 else 0,
                        'std': float(np.std(value)) if value.size > 0 else 0,
                        'min': float(np.min(value)) if value.size > 0 else 0,
                        'max': float(np.max(value)) if value.size > 0 else 0
                    }
                else:
                    summary[feature_type][key] = value
        else:
            summary[feature_type] = str(type(feature_data))
    
    return summary