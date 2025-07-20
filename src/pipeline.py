from pathlib import Path
import pandas as pd
from models.action_recognition import create_vit_model, evaluate_model
from models.feature_extraction import extract_joint_angles, extract_limb_velocities
from feedback.llm_feedback import generate_feedback
import torch

def run_pipeline():
    # Load data
    data = load_data()
    if data is None:
        return None, None

    # Extract features
    features = {}
    for key, df in data.items():
        joint_angles = extract_joint_angles(df)
        limb_velocities = extract_limb_velocities(df)
        features[key] = {**joint_angles, **limb_velocities}

    # Classify strokes
    input_shape = (3, 224, 224)  # Example input shape, adjust as needed
    num_classes = 5  # Example number of classes, adjust as needed
    model = create_vit_model(input_shape, num_classes)

    # Load a pre-trained model or train the model here
    # For now, we'll assume the model is already trained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = {}
    for key, feature_set in features.items():
        # Convert features to a tensor (example, adjust as needed)
        feature_tensor = torch.tensor(list(feature_set.values()), dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(feature_tensor.unsqueeze(0))  # Add batch dimension
            _, predicted = torch.max(outputs, 1)
            predictions[key] = predicted.item()

    # Generate feedback
    feedback = {}
    for key, prediction in predictions.items():
        feedback[key] = generate_feedback(prediction)

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