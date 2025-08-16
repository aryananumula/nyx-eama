import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd
from pathlib import Path

def visualize_tennis_biomechanics(data, features, output_path="tennis_analysis.png"):
    """
    Create comprehensive biomechanical visualization of tennis stroke analysis
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Joint Angles Over Time
    ax1 = fig.add_subplot(gs[0, :2])
    plot_joint_angles(ax1, features['joint_angles'])
    
    # 2. Racket Trajectory 3D
    ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
    plot_racket_trajectory_3d(ax2, data, features['racket_dynamics'])
    
    # 3. Kinetic Chain Sequence
    ax3 = fig.add_subplot(gs[1, :2])
    plot_kinetic_chain(ax3, features['kinetic_chain'])
    
    # 4. Velocity Profiles
    ax4 = fig.add_subplot(gs[1, 2:])
    plot_velocity_profiles(ax4, features['limb_velocities'])
    
    # 5. Body Rotation
    ax5 = fig.add_subplot(gs[2, :2])
    plot_body_rotation(ax5, features['body_rotation'])
    
    # 6. Power Generation
    ax6 = fig.add_subplot(gs[2, 2:])
    plot_power_generation(ax6, features['power_generation'])
    
    # 7. Stroke Phases
    ax7 = fig.add_subplot(gs[3, :2])
    plot_stroke_phases(ax7, features['timing_features'], features['racket_dynamics'])
    
    # 8. Feature Summary Heatmap
    ax8 = fig.add_subplot(gs[3, 2:])
    plot_feature_summary(ax8, features)
    
    plt.suptitle('Tennis Biomechanical Analysis Dashboard', fontsize=20, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_joint_angles(ax, joint_angles):
    """Plot key joint angles throughout the stroke"""
    time_axis = np.linspace(0, 100, len(next(iter(joint_angles.values()))))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (joint, angles) in enumerate(joint_angles.items()):
        if isinstance(angles, np.ndarray) and len(angles) > 0:
            ax.plot(time_axis[:len(angles)], np.degrees(angles), 
                   label=joint.replace('_', ' ').title(), 
                   color=colors[i % len(colors)], linewidth=2)
    
    ax.set_xlabel('Stroke Progress (%)')
    ax.set_ylabel('Joint Angle (degrees)')
    ax.set_title('Joint Kinematics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_racket_trajectory_3d(ax, data, racket_dynamics):
    """3D visualization of racket path with velocity color coding"""
    if 'racket_tip' in data.columns:
        racket_data = np.array(data['racket_tip'].tolist())
        x, y, z = racket_data[:, 0], racket_data[:, 1], racket_data[:, 2]
        
        # Color by velocity magnitude
        velocities = racket_dynamics.get('velocity', np.zeros(len(x)))
        if len(velocities) > 0 and np.max(velocities) > 0:
            # Normalize velocities
            norm_velocities = velocities / np.max(velocities)
            
            for i in range(len(x)-1):
                # Get normalized velocity for this segment
                vel_norm = norm_velocities[i] if i < len(norm_velocities) else 0
                # Ensure vel_norm is a scalar
                if isinstance(vel_norm, np.ndarray):
                    vel_norm = vel_norm.item() if vel_norm.size == 1 else vel_norm[0]
                
                # Get single color tuple from colormap
                color = plt.cm.plasma(float(vel_norm))
                # Ensure color is a single tuple, not an array
                if isinstance(color, np.ndarray) and color.ndim > 1:
                    color = color[0] if len(color) > 0 else 'blue'
                
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                       color=color, linewidth=3)
        else:
            # Fallback: plot with single color if no velocity data
            ax.plot(x, y, z, color='blue', linewidth=3)
        
        # Mark impact point
        impact_frame = racket_dynamics.get('impact_frame', len(x)//2)
        if impact_frame < len(x):
            ax.scatter(x[impact_frame], y[impact_frame], z[impact_frame], 
                      c='red', s=100, marker='*', label='Ball Impact')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Racket Trajectory', fontweight='bold')
    ax.legend()

def plot_kinetic_chain(ax, kinetic_chain):
    """Visualize kinetic chain activation sequence"""
    segments = ['Ankle', 'Knee', 'Hip', 'Spine', 'Shoulder', 'Elbow', 'Wrist', 'Racket']
    
    # Check what data we actually have
    if 'activation_times' in kinetic_chain:
        # Original expected format
        activation_times = kinetic_chain['activation_times']
        colors = plt.cm.RdYlBu(np.linspace(0, 1, len(segments)))
        bars = ax.barh(segments, activation_times, color=colors)
    elif 'racket_tip_timing' in kinetic_chain:
        # Work with available data - create a simplified visualization
        racket_timing = kinetic_chain.get('racket_tip_timing', 50)
        
        # Create dummy progression for visualization (you can improve this later)
        # Simulate kinetic chain progression: legs -> trunk -> arm -> racket
        dummy_times = [
            racket_timing * 0.1,  # Ankle
            racket_timing * 0.2,  # Knee  
            racket_timing * 0.3,  # Hip
            racket_timing * 0.5,  # Spine
            racket_timing * 0.7,  # Shoulder
            racket_timing * 0.85, # Elbow
            racket_timing * 0.95, # Wrist
            racket_timing         # Racket
        ]
        
        colors = plt.cm.RdYlBu(np.linspace(0, 1, len(segments)))
        bars = ax.barh(segments, dummy_times, color=colors, alpha=0.7)
        
        # Add note that this is estimated
        ax.text(0.02, 0.98, 'Estimated progression', transform=ax.transAxes, 
               fontsize=8, style='italic', va='top')
    else:
        # No kinetic chain data available
        ax.barh(segments, [0] * len(segments), alpha=0.3)
        ax.text(0.5, 0.5, 'Kinetic chain analysis not available', 
               transform=ax.transAxes, ha='center', va='center', 
               fontsize=10, color='gray')
    
    ax.set_xlabel('Activation Time (% of stroke)')
    ax.set_title('Kinetic Chain Sequence', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

def plot_velocity_profiles(ax, limb_velocities):
    """Plot velocity profiles for key body segments"""
    time_axis = np.linspace(0, 100, max(len(v) for v in limb_velocities.values() if isinstance(v, np.ndarray)))
    
    key_limbs = ['hand', 'forearm', 'upper_arm', 'racket']
    colors = ['red', 'blue', 'green', 'orange']
    
    for limb, color in zip(key_limbs, colors):
        if limb in limb_velocities:
            velocities = limb_velocities[limb]
            if isinstance(velocities, np.ndarray) and len(velocities) > 0:
                ax.plot(time_axis[:len(velocities)], velocities, 
                       label=limb.title(), color=color, linewidth=2)
    
    ax.set_xlabel('Stroke Progress (%)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Segmental Velocities', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_body_rotation(ax, body_rotation):
    """Visualize trunk rotation dynamics"""
    if 'trunk_rotation' in body_rotation and body_rotation['trunk_rotation'] is not None:
        rotation = body_rotation['trunk_rotation']
        angular_vel = body_rotation.get('trunk_angular_velocity', np.zeros_like(rotation))
        
        # Check if we have actual rotation data (not all zeros)
        if np.any(rotation != 0) or np.any(angular_vel != 0):
            time_axis = np.linspace(0, 100, len(rotation))
            
            ax.plot(time_axis, np.degrees(rotation), 'b-', linewidth=2, label='Trunk Rotation')
            ax.set_ylabel('Rotation (degrees)', color='blue')
            
            # Add angular velocity on secondary axis
            ax2 = ax.twinx()
            ax2.plot(time_axis, angular_vel, 'r--', linewidth=2, label='Angular Velocity')
            ax2.set_ylabel('Angular Velocity (rad/s)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        else:
            # All values are zero - show a message
            ax.text(0.5, 0.5, 'Body rotation calculation returned zero values\n(Possible missing trunk landmarks)', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=10, color='orange', style='italic')
            ax.set_title('Body Rotation Dynamics (No Movement Detected)', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Body rotation data not available', 
               transform=ax.transAxes, ha='center', va='center', 
               fontsize=10, color='gray')
    
    ax.set_xlabel('Stroke Progress (%)')
    ax.set_title('Body Rotation Dynamics', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)

def plot_power_generation(ax, power_generation):
    """Visualize power generation throughout stroke"""
    if 'kinetic_energy' in power_generation:
        ke = power_generation['kinetic_energy']
        time_axis = np.linspace(0, 100, len(ke))
        
        ax.fill_between(time_axis, ke, alpha=0.7, color='gold', label='Kinetic Energy')
        
        # Mark peak power
        peak_power = power_generation.get('peak_power', 0)
        peak_frame = np.argmax(ke) if len(ke) > 0 else 0
        ax.axvline(time_axis[peak_frame], color='red', linestyle='--', 
                  label=f'Peak Power: {peak_power:.2f}W')
    
    ax.set_xlabel('Stroke Progress (%)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Generation Profile', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_stroke_phases(ax, timing_features, racket_dynamics):
    """Visualize stroke phase segmentation"""
    duration = timing_features.get('stroke_duration', 100)
    
    # Phase boundaries
    prep_end = duration * 0.33
    exec_end = duration * 0.67
    
    phases = ['Preparation', 'Execution', 'Follow-through']
    phase_starts = [0, prep_end, exec_end]
    phase_lengths = [prep_end, exec_end - prep_end, duration - exec_end]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars = ax.barh(phases, phase_lengths, left=phase_starts, color=colors, alpha=0.7)
    
    # Mark impact point
    impact_frame = racket_dynamics.get('impact_frame', duration * 0.5)
    ax.axvline(impact_frame, color='red', linestyle='--', linewidth=3, label='Ball Impact')
    
    ax.set_xlabel('Time (frames)')
    ax.set_title('Stroke Phase Analysis', fontweight='bold')
    ax.legend()

def plot_feature_summary(ax, features):
    """Create heatmap summary of all extracted features"""
    feature_summary = {}
    
    for category, data in features.items():
        if isinstance(data, dict):
            feature_summary[category] = len([v for v in data.values() if v is not None])
        else:
            feature_summary[category] = 1 if data is not None else 0
    
    categories = list(feature_summary.keys())
    values = list(feature_summary.values())
    
    # Create heatmap-style bar chart
    colors = plt.cm.viridis(np.array(values) / max(values) if max(values) > 0 else [0])
    bars = ax.bar(categories, values, color=colors)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               str(value), ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Features Extracted')
    ax.set_title('Feature Extraction Summary', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

def create_stroke_animation(data, features, output_path="tennis_animation.gif"):
    """Create animated visualization of the tennis stroke"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Setup 3D plot for skeleton
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot skeleton at current frame
        plot_skeleton_frame(ax1, data, frame)
        
        # Plot velocity profile up to current frame
        plot_velocity_progress(ax2, features['limb_velocities'], frame)
        
        ax1.set_title(f'Tennis Stroke - Frame {frame}')
        ax2.set_title('Velocity Profile Progress')
    
    frames = min(len(data), 100)  # Limit animation length
    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
    anim.save(output_path, writer='pillow', fps=10)
    
    return anim

def plot_skeleton_frame(ax, data, frame):
    """Plot 3D skeleton for a specific frame"""
    # Define skeleton connections
    connections = [
        ('ShoulderLeft', 'ShoulderRight'),
        ('ShoulderRight', 'ElbowRight'),
        ('ElbowRight', 'WristRight'),
        ('WristRight', 'HandRight'),
        ('SpineBase', 'ShoulderCenter'),
        ('HipLeft', 'HipRight'),
        ('HipRight', 'KneeRight'),
        ('KneeRight', 'AnkleRight')
    ]
    
    # Plot connections
    for joint1, joint2 in connections:
        if f'{joint1}_X' in data.columns and f'{joint2}_X' in data.columns:
            x1, y1, z1 = data.iloc[frame][[f'{joint1}_X', f'{joint1}_Y', f'{joint1}_Z']]
            x2, y2, z2 = data.iloc[frame][[f'{joint2}_X', f'{joint2}_Y', f'{joint2}_Z']]
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=2)
    
    # Highlight racket (hand)
    if 'HandRight_X' in data.columns:
        x, y, z = data.iloc[frame][['HandRight_X', 'HandRight_Y', 'HandRight_Z']]
        ax.scatter(x, y, z, c='red', s=100, marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_velocity_progress(ax, limb_velocities, current_frame):
    """Plot velocity profiles up to current frame"""
    if 'hand' in limb_velocities:
        velocities = limb_velocities['hand'][:current_frame]
        ax.plot(velocities, 'r-', linewidth=2, label='Hand Velocity')
        ax.axvline(current_frame, color='black', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)