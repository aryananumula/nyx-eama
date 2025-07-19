from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example preprocessing: fill missing values and normalize data
    df.fillna(method='ffill', inplace=True)
    df = (df - df.mean()) / df.std()
    return df

def visualize_data(df: pd.DataFrame, title: str = "Data Visualization") -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend(df.columns)
    plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }