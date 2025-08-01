import pytest
from src.feedback.llm_feedback import generate_feedback

def test_generate_feedback():
    # Sample input data for testing
    stroke_classification = {
        'stroke_type': 'forehand',
        'accuracy': 0.85,
        'power': 7.5,
        'spin': 'topspin'
    }
    
    expected_feedback_keywords = ['forehand', 'accuracy', 'power', 'spin']
    
    feedback = generate_feedback(stroke_classification)
    
    assert isinstance(feedback, str), "Feedback should be a string."
    assert all(keyword in feedback for keyword in expected_feedback_keywords), "Feedback should contain relevant keywords."