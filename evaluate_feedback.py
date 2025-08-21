import os
import sys
import ezc3d
from pathlib import Path
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
EXTRACTED_DIR = BASE_DIR / "thetis_output" / "extracted"
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from models.feature_extraction import extract_tennis_biomechanical_features
from feedback.llm_feedback import generate_feedback
from feedback.feedback_tests import run_tests

results_list = []

def load_c3d_stroke(c3d_path: Path) -> pd.DataFrame:
    if not c3d_path.exists():
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    c3d_data = ezc3d.c3d(str(c3d_path))
    points = c3d_data['data']['points']
    labels = c3d_data['parameters']['POINT']['LABELS']['value']

    df_dict = {}
    for i, label in enumerate(labels):
        df_dict[f"{label}_x"] = points[0, i, :]
        df_dict[f"{label}_y"] = points[1, i, :]
        df_dict[f"{label}_z"] = points[2, i, :]

    return pd.DataFrame(df_dict)

def evaluate_stroke(c3d_path):
    stroke_df = load_c3d_stroke(c3d_path)
    features = extract_tennis_biomechanical_features(stroke_df)
    
    if "classification" not in features:
        features["classification"] = {
            "type": "Backhand (One-handed)",
            "power_level": None,
        }

    features_for_feedback = {}
    for k, v in features.items():
        if isinstance(v, dict):
            features_for_feedback[k] = v.copy()
        else:
            features_for_feedback[k] = v

    feedback = generate_feedback(features_for_feedback)
    results = run_tests(features, feedback)
    return results

for folder in tqdm(sorted(EXTRACTED_DIR.iterdir()), desc="Backhand folders"):
    if "bh" not in folder.name.lower() or not folder.is_dir():
        continue

    nested_folder = folder / folder.name
    if not nested_folder.exists():
        print(f"Warning: Nested folder not found for {folder.name}")
        continue

    for stroke_folder in sorted(nested_folder.iterdir()):
        if not stroke_folder.is_dir() or not stroke_folder.name.startswith("s"):
            continue

        c3d_files = list(stroke_folder.glob("*.c3d"))
        if not c3d_files:
            print(f"No C3D found in {stroke_folder}")
            continue

        c3d_path = c3d_files[0]
        try:
            test_results = evaluate_stroke(c3d_path)
            results_list.append({
                "c3d_file": c3d_path.name,
                "test_1": test_results[0],
                "test_2": test_results[1],
                "test_3": test_results[2],
            })
        except Exception as e:
            print(f"Error processing {c3d_path}: {e}")

results_df = pd.DataFrame(results_list)
results_df.to_csv(BASE_DIR / "stroke_evaluation_results.csv", index=False)
print(f"Results saved to {BASE_DIR / 'stroke_evaluation_results.csv'}")
