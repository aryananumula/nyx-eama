import os
import sys
import numpy as np

# Force mock mode (no API calls) 
os.environ["MOCK_LLM"] = "True"
os.environ.pop("OPENAI_API_KEY", None)

# Make src/ importable
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "../src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feedback.llm_feedback import generate_feedback  # noqa: E402


def test_generate_feedback_minimal():
    """Smoke test: should return a non-empty string and mention the stroke."""
    payload = {
        "stroke_type": "forehand",
        "accuracy": 0.85,
        "power": 7.5,
        "spin": "topspin",
    }

    feedback = generate_feedback(payload)
    print("Generated Feedback (minimal):", feedback)

    assert isinstance(feedback, str)
    assert feedback.strip() != ""
    # Should mention the stroke (mock path may format as 'stroke=forehand')
    assert "forehand" in feedback.lower() or "stroke=forehand" in feedback.lower()
    # Should clearly be mock output (so we know no credits were burned)
    assert "(mock)" in feedback or "mock feedback" in feedback.lower()


def test_generate_feedback_with_biomech_signals():
    """Richer input: ensure mock feedback reacts to biomech stats without calling the API."""
    payload = {
        "stroke_type": "backhand",
        # arrays → summarized into mean/max/min
        "joint_angles": {
            "shoulder_flexion": np.array([35, 38, 40, 42]),  # low-ish mean
            "elbow_flexion": np.array([85, 90, 88, 92]),     # high mean → may trigger elbow advice
        },
        "body_rotation": {
            "torso_rotation_deg": np.array([25, 30, 33, 35]),  # < 40 → limited torso rotation
            "hip_rotation_deg": np.array([28, 34, 36, 39]),
        },
        "racket_dynamics": {
            "tip_speed": np.array([4.8, 5.1, 5.6]),  # max < 7 → low racket head speed
        },
        "kinetic_chain": {
            "timing_offsets": {"arm_to_racket": 0.08},  # > 0.06 → late wrist release
        },
    }

    feedback = generate_feedback(payload)
    print("Generated Feedback (biomech):", feedback)

    assert isinstance(feedback, str)
    assert feedback.strip() != ""
    assert "backhand" in feedback.lower() or "stroke=backhand" in feedback.lower()
    # Likely mentions at least one of the triggered issues
    possible_flags = ["torso rotation", "racket head speed", "wrist release", "elbow"]
    assert any(flag in feedback.lower() for flag in possible_flags)
    assert "(mock)" in feedback or "mock feedback" in feedback.lower()

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# import pytest
# from feedback.llm_feedback import generate_feedback

# def test_generate_feedback():
#     # Sample input data for testing
#     stroke_classification = {
#         'stroke_type': 'forehand',
#         'accuracy': 0.85,
#         'power': 7.5,
#         'spin': 'topspin'
#     }

#     feedback = generate_feedback(stroke_classification)
    
#     print("Generated Feedback:", feedback) 

#     assert isinstance(feedback, str), "Feedback should be a string."
#     assert len(feedback.strip()) > 0, "Feedback should not be empty."

# ____________________________________________________________________________________________________

# import pytest
# from src.feedback.llm_feedback import generate_feedback

# def test_generate_feedback():
#     # Sample input data for testing
#     stroke_classification = {
#         'stroke_type': 'forehand',
#         'accuracy': 0.85,
#         'power': 7.5,
#         'spin': 'topspin'
#     }
    
#     expected_feedback_keywords = ['forehand', 'accuracy', 'power', 'spin']
    
#     feedback = generate_feedback(stroke_classification)
    
#     assert isinstance(feedback, str), "Feedback should be a string."
#     assert all(keyword in feedback for keyword in expected_feedback_keywords), "Feedback should contain relevant keywords."