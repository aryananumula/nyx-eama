#test_llm_feedback.py
import sys
import os
import numpy as np

# Add src to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from feedback.llm_feedback import build_context_summary, generate_feedback

# Mock features from feature_extraction output
MOCK_FEATURES = {
    "joint_angles": {
        "shoulder_flexion": np.random.normal(62.11, 32.69, 100).tolist(),
        "elbow_flexion": np.random.normal(62.11, 32.69, 100).tolist(),
        "wrist_extension": np.random.normal(85.31, 38.09, 100).tolist(),
        "hip_rotation": np.random.normal(68.11, 37.27, 100).tolist(),
        "knee_flexion": np.random.normal(63.15, 36.28, 100).tolist(),
    },
    "limb_velocities": {
        "hand": np.random.normal(-0.02, 0.70, (100, 3)).tolist(),
        "racket_tip": np.random.normal(0.05, 0.71, (100, 3)).tolist(),
    },
    "racket_dynamics": {
        "speed": np.random.normal(1.12, 0.53, 100).tolist(),
        "acceleration_magnitude": np.random.normal(0.99, 0.50, 100).tolist(),
        "impact_frame": 99,
        "max_velocity": 3.92,
    },
    "timing_features": {
        "stroke_duration": 100,
        "preparation_phase": [0, 33],
        "execution_phase": [33, 66],
        "follow_through_phase": [66, 100],
    },
    "power_generation": {
        "kinetic_energy": np.random.normal(0.77, 0.87, 100).tolist(),
        "peak_power": 7.68,
        "average_power": 0.77,
    },
    "classification": {
        "type": "volley",
        "power_level": "low",
    },
}

def test_context_summary_generation_with_mock_features():
    print(">>> Formatted context summary:")
    summary = build_context_summary(MOCK_FEATURES)
    print(summary)
    print("====================")                      
    
def test_generate_feedback_with_mock_features():
    print(">>> Sending mocked features to LLM...")
    response = generate_feedback(MOCK_FEATURES)

    print("\n=== LLM Response ===")
    print(response)
    print("====================")

if __name__ == "__main__":
    test_generate_feedback_with_mock_features()
