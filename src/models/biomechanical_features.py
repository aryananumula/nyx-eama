import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class THETISBiomechanicalFeatures:
    """
    Comprehensive biomechanical feature extraction for THETIS dataset.
    Extracts features from 3D motion capture data including joint angles,
    velocities, accelerations, and tennis-specific biomechanical patterns.
    """
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize the biomechanical feature extractor.
        
        Args:
            frame_rate: Frame rate of the motion capture data (default: 30.0 Hz)
        """
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        
        # Joint mapping for THETIS dataset (204 points)
        # Based on typical motion capture marker sets
        self.joint_mapping = {
            # Head and neck
            'head': 1,  # Point1 - Head center
            'neck': 2,  # Point2 - Neck/C7
            
            # Shoulders
            'left_shoulder': 3,   # Point3 - Left shoulder
            'right_shoulder': 4,  # Point4 - Right shoulder
            
            # Arms
            'left_elbow': 5,      # Point5 - Left elbow
            'right_elbow': 6,     # Point6 - Right elbow
            'left_wrist': 7,      # Point7 - Left wrist
            'right_wrist': 8,     # Point8 - Right wrist
            
            # Torso
            'spine_chest': 9,     # Point9 - Upper spine/chest
            'spine_waist': 10,    # Point10 - Lower spine/waist
            'left_hip': 11,       # Point11 - Left hip
            'right_hip': 12,      # Point12 - Right hip
            
            # Legs
            'left_knee': 13,      # Point13 - Left knee
            'right_knee': 14,     # Point14 - Right knee
            'left_ankle': 15,     # Point15 - Left ankle
            'right_ankle': 16,    # Point16 - Right ankle
            'left_foot': 17,      # Point17 - Left foot
            'right_foot': 18,     # Point18 - Right foot
            
            # Tennis racket (if available)
            'racket_head': 19,    # Point19 - Racket head
            'racket_handle': 20,  # Point20 - Racket handle
            
            # Additional body points for more detailed analysis
            'left_shoulder_blade': 21,  # Point21 - Left scapula
            'right_shoulder_blade': 22, # Point22 - Right scapula
            'left_upper_arm': 23,       # Point23 - Left upper arm
            'right_upper_arm': 24,      # Point24 - Right upper arm
            'left_forearm': 25,         # Point25 - Left forearm
            'right_forearm': 26,        # Point26 - Right forearm
            'left_hand': 27,            # Point27 - Left hand
            'right_hand': 28,           # Point28 - Right hand
            'left_thigh': 29,           # Point29 - Left thigh
            'right_thigh': 30,          # Point30 - Right thigh
            'left_shin': 31,            # Point31 - Left shin
            'right_shin': 32,           # Point32 - Right shin
        }
        
        # Joint chains for angle calculations
        self.joint_chains = {
            'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
            'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
            'spine': ['spine_waist', 'spine_chest', 'neck'],
            'left_shoulder_complex': ['left_shoulder_blade', 'left_shoulder', 'left_elbow'],
            'right_shoulder_complex': ['right_shoulder_blade', 'right_shoulder', 'right_elbow'],
        }
        
        # Tennis-specific joint combinations
        self.tennis_joints = {
            'serve_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'serve_legs': ['right_hip', 'right_knee', 'right_ankle'],
            'forehand_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
            'backhand_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
            'stance_legs': ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle'],
        }
    
    def extract_all_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract all biomechanical features from the motion capture data.
        
        Args:
            data: DataFrame with columns Point1_X, Point1_Y, Point1_Z, etc.
            
        Returns:
            Dictionary containing all extracted features
        """
        print("Extracting biomechanical features from THETIS dataset...")
        
        # Convert data to 3D point format
        points_3d = self._convert_to_3d_points(data)
        
        features = {}
        
        # 1. Joint angles
        print("  - Calculating joint angles...")
        features['joint_angles'] = self._calculate_joint_angles(points_3d)
        
        # 2. Joint velocities
        print("  - Calculating joint velocities...")
        features['joint_velocities'] = self._calculate_joint_velocities(points_3d)
        
        # 3. Joint accelerations
        print("  - Calculating joint accelerations...")
        features['joint_accelerations'] = self._calculate_joint_accelerations(points_3d)
        
        # 4. Body segment lengths
        print("  - Calculating body segment lengths...")
        features['segment_lengths'] = self._calculate_segment_lengths(points_3d)
        
        # 5. Center of mass
        print("  - Calculating center of mass...")
        features['center_of_mass'] = self._calculate_center_of_mass(points_3d)
        
        # 6. Body rotation features
        print("  - Calculating body rotation features...")
        features['body_rotation'] = self._calculate_body_rotation_features(points_3d)
        
        # 7. Tennis-specific features
        print("  - Calculating tennis-specific features...")
        features['tennis_specific'] = self._calculate_tennis_specific_features(points_3d)
        
        # 8. Temporal features
        print("  - Calculating temporal features...")
        features['temporal'] = self._calculate_temporal_features(points_3d)
        
        # 9. Power and energy features
        print("  - Calculating power and energy features...")
        features['power_energy'] = self._calculate_power_energy_features(points_3d)
        
        print("Feature extraction completed!")
        return features
    
    def _convert_to_3d_points(self, data: pd.DataFrame) -> np.ndarray:
        """
        Convert DataFrame with PointX_Y columns to 3D point array.
        
        Args:
            data: DataFrame with columns Point1_X, Point1_Y, Point1_Z, etc.
            
        Returns:
            3D array of shape (frames, points, 3) with X, Y, Z coordinates
        """
        n_frames = len(data)
        n_points = len([col for col in data.columns if col.endswith('_X')])
        
        points_3d = np.zeros((n_frames, n_points, 3))
        
        for i in range(n_points):
            x_col = f'Point{i+1}_X'
            y_col = f'Point{i+1}_Y'
            z_col = f'Point{i+1}_Z'
            
            if x_col in data.columns and y_col in data.columns and z_col in data.columns:
                points_3d[:, i, 0] = data[x_col].values
                points_3d[:, i, 1] = data[y_col].values
                points_3d[:, i, 2] = data[z_col].values
        
        return points_3d
    
    def _calculate_joint_angles(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate joint angles for tennis-relevant joints.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of joint angles over time
        """
        angles = {}
        
        for angle_name, joint_sequence in self.tennis_joints.items():
            try:
                # Get joint indices
                joint1_idx = self.joint_mapping[joint_sequence[0]]
                joint2_idx = self.joint_mapping[joint_sequence[1]]
                joint3_idx = self.joint_mapping[joint_sequence[2]]
                
                # Calculate angle over time
                angle_values = []
                for frame in range(len(points_3d)):
                    p1 = points_3d[frame, joint1_idx]
                    p2 = points_3d[frame, joint2_idx]
                    p3 = points_3d[frame, joint3_idx]
                    
                    # Calculate vectors
                    v1 = p1 - p2
                    v2 = p3 - p2
                    
                    # Calculate angle
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angle_values.append(np.degrees(angle))
                
                angles[angle_name] = np.array(angle_values)
                
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not calculate {angle_name}: {e}")
                angles[angle_name] = np.zeros(len(points_3d))
        
        return angles
    
    def _calculate_joint_velocities(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate joint velocities using finite differences.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of joint velocities over time
        """
        velocities = {}
        
        for joint_name, joint_idx in self.joint_mapping.items():
            try:
                # Calculate velocity using finite differences
                joint_positions = points_3d[:, joint_idx - 1, :]  # Convert to 0-based indexing
                velocity = np.gradient(joint_positions, axis=0) / self.dt
                
                # Calculate velocity magnitude
                velocity_magnitude = np.linalg.norm(velocity, axis=1)
                
                velocities[f'{joint_name}_velocity'] = velocity
                velocities[f'{joint_name}_velocity_magnitude'] = velocity_magnitude
                
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not calculate velocity for {joint_name}: {e}")
        
        return velocities
    
    def _calculate_joint_accelerations(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate joint accelerations using finite differences.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of joint accelerations over time
        """
        accelerations = {}
        
        for joint_name, joint_idx in self.joint_mapping.items():
            try:
                # Calculate acceleration using finite differences
                joint_positions = points_3d[:, joint_idx - 1, :]  # Convert to 0-based indexing
                velocity = np.gradient(joint_positions, axis=0) / self.dt
                acceleration = np.gradient(velocity, axis=0) / self.dt
                
                # Calculate acceleration magnitude
                acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
                
                accelerations[f'{joint_name}_acceleration'] = acceleration
                accelerations[f'{joint_name}_acceleration_magnitude'] = acceleration_magnitude
                
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not calculate acceleration for {joint_name}: {e}")
        
        return accelerations
    
    def _calculate_segment_lengths(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate body segment lengths over time.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of segment lengths over time
        """
        segment_lengths = {}
        
        for segment_name, point_indices in self.joint_chains.items():
            try:
                if len(point_indices) >= 2:
                    # Calculate length between first and last point in segment
                    start_idx = self.joint_mapping[point_indices[0]] - 1  # Convert to 0-based indexing
                    end_idx = self.joint_mapping[point_indices[-1]] - 1
                    
                    lengths = []
                    for frame in range(len(points_3d)):
                        start_point = points_3d[frame, start_idx]
                        end_point = points_3d[frame, end_idx]
                        length = np.linalg.norm(end_point - start_point)
                        lengths.append(length)
                    
                    segment_lengths[f'{segment_name}_length'] = np.array(lengths)
                    
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not calculate length for {segment_name}: {e}")
        
        return segment_lengths
    
    def _calculate_center_of_mass(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate center of mass and related features.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of center of mass features
        """
        com_features = {}
        
        # Calculate center of mass as mean of all points
        com_positions = np.mean(points_3d, axis=1)
        
        # Calculate COM velocity
        com_velocity = np.gradient(com_positions, axis=0) / self.dt
        
        # Calculate COM acceleration
        com_acceleration = np.gradient(com_velocity, axis=0) / self.dt
        
        com_features['com_position'] = com_positions
        com_features['com_velocity'] = com_velocity
        com_features['com_acceleration'] = com_acceleration
        com_features['com_velocity_magnitude'] = np.linalg.norm(com_velocity, axis=1)
        com_features['com_acceleration_magnitude'] = np.linalg.norm(com_acceleration, axis=1)
        
        return com_features
    
    def _calculate_body_rotation_features(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate body rotation features using shoulder vector analysis.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of body rotation features
        """
        rotation_features = {}
        
        try:
            # Get shoulder positions
            left_shoulder_idx = self.joint_mapping['left_shoulder'] - 1  # Convert to 0-based indexing
            right_shoulder_idx = self.joint_mapping['right_shoulder'] - 1
            
            # Calculate shoulder vector
            shoulder_vectors = points_3d[:, right_shoulder_idx] - points_3d[:, left_shoulder_idx]
            
            # Calculate rotation angle in XY plane (horizontal rotation)
            rotation_angles = np.arctan2(shoulder_vectors[:, 1], shoulder_vectors[:, 0])
            rotation_angles_deg = np.degrees(rotation_angles)
            
            # Calculate rotation velocity
            rotation_velocity = np.gradient(rotation_angles, axis=0) / self.dt
            
            # Calculate rotation acceleration
            rotation_acceleration = np.gradient(rotation_velocity, axis=0) / self.dt
            
            rotation_features['shoulder_rotation_angle'] = rotation_angles_deg
            rotation_features['shoulder_rotation_velocity'] = rotation_velocity
            rotation_features['shoulder_rotation_acceleration'] = rotation_acceleration
            
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not calculate body rotation features: {e}")
            rotation_features['shoulder_rotation_angle'] = np.zeros(len(points_3d))
            rotation_features['shoulder_rotation_velocity'] = np.zeros(len(points_3d))
            rotation_features['shoulder_rotation_acceleration'] = np.zeros(len(points_3d))
        
        return rotation_features
    
    def _calculate_tennis_specific_features(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate tennis-specific biomechanical features.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of tennis-specific features
        """
        tennis_features = {}
        
        try:
            # 1. Racket arm extension (assuming right hand is racket hand)
            right_shoulder_idx = self.joint_mapping['right_shoulder']
            right_elbow_idx = self.joint_mapping['right_elbow']
            right_wrist_idx = self.joint_mapping['right_wrist']
            
            arm_extension = []
            for frame in range(len(points_3d)):
                shoulder = points_3d[frame, right_shoulder_idx]
                elbow = points_3d[frame, right_elbow_idx]
                wrist = points_3d[frame, right_wrist_idx]
                
                # Calculate arm extension as distance from shoulder to wrist
                extension = np.linalg.norm(wrist - shoulder)
                arm_extension.append(extension)
            
            tennis_features['racket_arm_extension'] = np.array(arm_extension)
            
            # 2. Shoulder-hip separation (kinetic chain)
            right_shoulder_idx = self.joint_mapping['right_shoulder']
            right_hip_idx = self.joint_mapping['right_hip']
            
            shoulder_hip_separation = []
            for frame in range(len(points_3d)):
                shoulder = points_3d[frame, right_shoulder_idx]
                hip = points_3d[frame, right_hip_idx]
                
                # Calculate horizontal separation
                separation = np.linalg.norm(shoulder[:2] - hip[:2])  # Only X and Y
                shoulder_hip_separation.append(separation)
            
            tennis_features['shoulder_hip_separation'] = np.array(shoulder_hip_separation)
            
            # 3. Knee bend (average of both knees)
            left_knee_idx = self.joint_mapping['left_knee']
            right_knee_idx = self.joint_mapping['right_knee']
            left_hip_idx = self.joint_mapping['left_hip']
            right_hip_idx = self.joint_mapping['right_hip']
            
            knee_bend = []
            for frame in range(len(points_3d)):
                left_hip = points_3d[frame, left_hip_idx]
                left_knee = points_3d[frame, left_knee_idx]
                right_hip = points_3d[frame, right_hip_idx]
                right_knee = points_3d[frame, right_knee_idx]
                
                # Calculate knee bend as vertical distance from hip to knee
                left_bend = abs(left_hip[2] - left_knee[2])
                right_bend = abs(right_hip[2] - right_knee[2])
                avg_bend = (left_bend + right_bend) / 2
                knee_bend.append(avg_bend)
            
            tennis_features['knee_bend'] = np.array(knee_bend)
            
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not calculate tennis-specific features: {e}")
            tennis_features['racket_arm_extension'] = np.zeros(len(points_3d))
            tennis_features['shoulder_hip_separation'] = np.zeros(len(points_3d))
            tennis_features['knee_bend'] = np.zeros(len(points_3d))
        
        return tennis_features
    
    def _calculate_temporal_features(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate temporal features and phase detection.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of temporal features
        """
        temporal_features = {}
        
        # Calculate overall movement magnitude
        movement_magnitude = []
        for frame in range(1, len(points_3d)):
            frame_diff = points_3d[frame] - points_3d[frame-1]
            magnitude = np.mean(np.linalg.norm(frame_diff, axis=1))
            movement_magnitude.append(magnitude)
        
        # Add zero for first frame
        movement_magnitude.insert(0, 0)
        temporal_features['movement_magnitude'] = np.array(movement_magnitude)
        
        # Detect movement phases
        threshold = np.percentile(movement_magnitude, 75)
        is_moving = movement_magnitude > threshold
        temporal_features['is_moving'] = is_moving
        
        # Calculate movement duration
        movement_duration = np.sum(is_moving) * self.dt
        temporal_features['total_movement_duration'] = movement_duration
        
        return temporal_features
    
    def _calculate_power_energy_features(self, points_3d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate power and energy-related features.
        
        Args:
            points_3d: 3D point array of shape (frames, points, 3)
            
        Returns:
            Dictionary of power and energy features
        """
        power_features = {}
        
        try:
            # Calculate kinetic energy (simplified - assuming unit mass)
            com_velocity = np.gradient(np.mean(points_3d, axis=1), axis=0) / self.dt
            kinetic_energy = 0.5 * np.linalg.norm(com_velocity, axis=1)**2
            
            # Calculate power (rate of change of kinetic energy)
            power = np.gradient(kinetic_energy, axis=0) / self.dt
            
            power_features['kinetic_energy'] = kinetic_energy
            power_features['power'] = power
            power_features['max_power'] = np.max(power)
            power_features['power_impulse'] = np.trapz(np.abs(power)) * self.dt
            
        except Exception as e:
            print(f"Warning: Could not calculate power features: {e}")
            power_features['kinetic_energy'] = np.zeros(len(points_3d))
            power_features['power'] = np.zeros(len(points_3d))
            power_features['max_power'] = 0
            power_features['power_impulse'] = 0
        
        return power_features
    
    def get_feature_summary(self, features: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Create a summary DataFrame of all extracted features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            DataFrame with feature statistics
        """
        summary_data = []
        
        for feature_category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for feature_name, feature_values in feature_dict.items():
                    if isinstance(feature_values, np.ndarray) and len(feature_values) > 0:
                        # Handle multi-dimensional arrays
                        if feature_values.ndim == 1:
                            summary_data.append({
                                'category': feature_category,
                                'feature': feature_name,
                                'mean': np.mean(feature_values),
                                'std': np.std(feature_values),
                                'min': np.min(feature_values),
                                'max': np.max(feature_values),
                                'median': np.median(feature_values),
                                'length': len(feature_values)
                            })
                        elif feature_values.ndim == 2:
                            # For 2D arrays, calculate stats for each dimension
                            for i in range(feature_values.shape[1]):
                                dim_values = feature_values[:, i]
                                summary_data.append({
                                    'category': feature_category,
                                    'feature': f'{feature_name}_dim{i}',
                                    'mean': np.mean(dim_values),
                                    'std': np.std(dim_values),
                                    'min': np.min(dim_values),
                                    'max': np.max(dim_values),
                                    'median': np.median(dim_values),
                                    'length': len(dim_values)
                                })
                    elif isinstance(feature_values, (int, float)):
                        summary_data.append({
                            'category': feature_category,
                            'feature': feature_name,
                            'value': feature_values,
                            'length': 1
                        })
        
        return pd.DataFrame(summary_data)
    
    def plot_feature_trajectories(self, features: Dict[str, np.ndarray], 
                                save_path: Optional[str] = None) -> None:
        """
        Plot feature trajectories over time.
        
        Args:
            features: Dictionary of extracted features
            save_path: Optional path to save the plot
        """
        # Count total features
        n_features = 0
        for feature_category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for feature_name, feature_values in feature_dict.items():
                    if isinstance(feature_values, np.ndarray) and len(feature_values) > 0:
                        if feature_values.ndim == 1:
                            n_features += 1
                        elif feature_values.ndim == 2:
                            n_features += min(3, feature_values.shape[1])  # Plot first 3 dimensions
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for feature_category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for feature_name, feature_values in feature_dict.items():
                    if isinstance(feature_values, np.ndarray) and len(feature_values) > 0:
                        if plot_idx < len(axes):
                            time = np.arange(len(feature_values)) * self.dt
                            
                            if feature_values.ndim == 1:
                                axes[plot_idx].plot(time, feature_values)
                                axes[plot_idx].set_title(f'{feature_category}: {feature_name}')
                            elif feature_values.ndim == 2:
                                # Plot first 3 dimensions
                                for i in range(min(3, feature_values.shape[1])):
                                    axes[plot_idx].plot(time, feature_values[:, i], 
                                                      label=f'Dim {i}')
                                axes[plot_idx].set_title(f'{feature_category}: {feature_name}')
                                axes[plot_idx].legend()
                            
                            axes[plot_idx].set_xlabel('Time (s)')
                            axes[plot_idx].set_ylabel('Value')
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature trajectories plot saved to {save_path}")
        
        plt.show()
    
    def export_features_to_csv(self, features: Dict[str, np.ndarray], 
                              output_path: str) -> None:
        """
        Export all features to CSV files.
        
        Args:
            features: Dictionary of extracted features
            output_path: Base path for output files
        """
        # Create a combined DataFrame for all features
        all_features = {}
        
        # Get the length of the first array to determine frame count
        frame_count = None
        for feature_category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for feature_name, feature_values in feature_dict.items():
                    if isinstance(feature_values, np.ndarray) and len(feature_values) > 0:
                        frame_count = len(feature_values)
                        break
                if frame_count is not None:
                    break
        
        if frame_count is None:
            print("Warning: No valid feature arrays found")
            return
        
        for feature_category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for feature_name, feature_values in feature_dict.items():
                    if isinstance(feature_values, np.ndarray):
                        # Handle multi-dimensional arrays
                        if feature_values.ndim == 1:
                            all_features[f'{feature_category}_{feature_name}'] = feature_values
                        elif feature_values.ndim == 2:
                            # For 2D arrays (e.g., velocity vectors), create separate columns
                            for i in range(feature_values.shape[1]):
                                all_features[f'{feature_category}_{feature_name}_dim{i}'] = feature_values[:, i]
                        else:
                            # For higher dimensional arrays, flatten or skip
                            print(f"Warning: Skipping {feature_name} - too many dimensions")
                    elif isinstance(feature_values, (int, float)):
                        # For scalar values, repeat for all frames
                        all_features[f'{feature_category}_{feature_name}'] = [feature_values] * frame_count
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Features exported to {output_path}")
        
        # Also save summary
        summary = self.get_feature_summary(features)
        summary_path = output_path.replace('.csv', '_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Feature summary exported to {summary_path}")


def extract_biomechanical_features_from_file(file_path: str, 
                                           output_dir: str = "biomechanical_features",
                                           frame_rate: float = 30.0) -> Dict[str, np.ndarray]:
    """
    Extract biomechanical features from a THETIS dataset CSV file.
    
    Args:
        file_path: Path to the CSV file
        output_dir: Directory to save output files
        frame_rate: Frame rate of the data
        
    Returns:
        Dictionary of extracted features
    """
    # Load data
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    
    # Initialize feature extractor
    extractor = THETISBiomechanicalFeatures(frame_rate=frame_rate)
    
    # Extract features
    features = extractor.extract_all_features(data)
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_biomechanical_features.csv")
    
    # Export features
    extractor.export_features_to_csv(features, output_path)
    
    # Create plots
    plot_path = os.path.join(output_dir, f"{base_name}_feature_trajectories.png")
    extractor.plot_feature_trajectories(features, save_path=plot_path)
    
    return features


if __name__ == "__main__":
    # Example usage
    file_path = "thetis_output/tp4_bh_s1.csv"
    features = extract_biomechanical_features_from_file(file_path)
    
    # Print summary
    extractor = THETISBiomechanicalFeatures()
    summary = extractor.get_feature_summary(features)
    print("\nFeature Summary:")
    print(summary)
