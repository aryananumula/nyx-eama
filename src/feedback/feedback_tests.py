import re
from feedback.llm_feedback import compare_to_reference
from feedback.llm_feedback import REFERENCE_RANGES


def check_format(feedback: str) -> bool:
    """
    checks if feedback contains a numerical score and exactly 3 corrections
    """
    has_score = feedback.strip().startswith("Overall Score:")
    # checks if there are dashes, bullets, or numbers at the start of lines
    corrections = re.findall(r"^(?:[-•]|\d+\.)\s+.+", feedback, re.MULTILINE)
    return has_score and len(corrections) == 3


def check_reference(features, feedback: str):
    """
    check if features flagged as LOW/HIGH are accurately judged 
    returns percentage of correct flags
    """
    true_comp = compare_to_reference(features, features["classification"]["type"])
    key_map = {
        "Racket velocity (m/s)": "racket_velocity",
        "Peak power (W)": "power",
        "Rotation range (°)": "rotation",
        "Stroke duration (frames @60fps)": "duration",
        "Peak angular velocity (rad/s)": "angular_velocity",
        "Impact timing (%)": "timing"
    }

    truth = {}
    for metric in true_comp:
        for human_name, prog_key in key_map.items():
            if metric.startswith(human_name):
                if "HIGH" in metric:
                    truth[prog_key] = "HIGH"
                elif "LOW" in metric:
                    truth[prog_key] = "LOW"

    reported = {}
    for line in feedback.splitlines():
        line = line.strip()
        for human_name, prog_key in key_map.items():
            if human_name in line:
                if "HIGH" in line:
                    reported[prog_key] = "HIGH"
                elif "LOW" in line:
                    reported[prog_key] = "LOW"

    total_checks = len(truth)
    if total_checks == 0:
        return 100.0 
    correct = sum(1 for k, v in truth.items() if reported.get(k) == v)
    validity = correct / total_checks * 100

    return validity


def check_num_fidelity(feedback: str, features) -> bool:
    """
    checks for fabricated numbers outside of the features dict or reference ranges
    """
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b(?!\s*\.)', feedback)
    valid_numbers = set()
    for v in features.values():
        try:
            valid_numbers.add(f"{float(v):.0f}")
        except:
            pass
    for stroke, ranges in REFERENCE_RANGES.items():
        for lo, hi in ranges.values():
            valid_numbers.add(f"{lo:.0f}")
            valid_numbers.add(f"{hi:.0f}")
    for n in numbers:
        if n not in valid_numbers:
            return False
    return True

def check_error(feedback: str) -> bool:
    feedback_lower = feedback.lower()
    if "error" in feedback_lower:
        return True
    return False


def run_tests(features, feedback):
    print("running tests...")
    structure = check_format(feedback)
    reference = check_reference(features, feedback)
    fidelity = check_num_fidelity(feedback, features)
    errors = check_error(feedback)
    print(structure, reference, fidelity, errors)
    return structure, reference, fidelity, errors
