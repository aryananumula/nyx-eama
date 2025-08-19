import re
from feedback.llm_feedback import generate_feedback, compare_to_reference
from feedback.llm_feedback import REFERENCE_RANGES


def check_format(feedback: str) -> bool:
    """
    checks if feedback contains a numerical score and exactly 3 corrections
    """
    has_score = feedback.strip().startswith("Overall Score:")
    # checks if there are dashes, bullets, or numbers at the start of lines
    corrections = re.findall(r"(?:^[-•\d]+\s+.+)", feedback, re.MULTILINE)
    return has_score and len(corrections) == 3


def check_reference(features, feedback: str):
    """
    check if features flagged as LOW/HIGH are accurately judged 
    returns f1 score (0-1)
    """
    true_comp = compare_to_reference(features, features["predicted_stroke"])
    expected = 0 # to be figured out...!
    found = 0 # :D

    tp = len(expected & found) # true positives
    fp = len(found - expected) # false positives
    fn = len(expected - found) # false negatives

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def check_num_fidelity(feedback: str, features) -> bool:
    """
    checks for fabricated numbers outside of the features dict or reference ranges
    """
    numbers = re.findall(r"\d+(?:\.\d+)?", feedback)
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


def run_experiment(test_cases):
    results = []
    for features in test_cases:
        fb = generate_feedback(features)
        fmt_ok = check_format(fb)
        f1 = check_reference(features, fb)
        fidelity_ok = check_num_fidelity(fb, features)
        results.append((fmt_ok, f1, fidelity_ok))
        print("---")
        print(fb)
        print(
            f"Format: {fmt_ok}, F1 Score: {f1}, Fidelity: {fidelity_ok}"
        )
    return results
