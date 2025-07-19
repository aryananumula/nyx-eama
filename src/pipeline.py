from pathlib import Path
import pandas as pd
from models.action_recognition import ActionRecognitionModel
from models.feature_extraction import extract_features
from feedback.llm_feedback import generate_feedback

def run_pipeline():
    # Load data
    data = load_data()
    if data is None:
        return

    # Extract features
    features = extract_features(data)
    
    # Classify strokes
    model = ActionRecognitionModel()
    predictions = model.classify(features)

    # Generate feedback
    feedback = generate_feedback(predictions)

    return predictions, feedback

def load_data():
    output_dir = Path("thetis_output")
    if not output_dir.exists():
        print("Output directory not found.")
        return None

    csv_files = list(output_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the thetis_output directory.")
        return None

    data = {}
    for csv_file in csv_files:
        key = csv_file.stem
        try:
            data[key] = pd.read_csv(csv_file)
            print(f"Loaded {key}: {len(data[key]):,} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    return data

if __name__ == "__main__":
    predictions, feedback = run_pipeline()
    print("Predictions:", predictions)
    print("Feedback:", feedback)