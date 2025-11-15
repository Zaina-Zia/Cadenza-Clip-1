"""Evaluate predictions against ground truth correctness values."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

def rmse_score(x: np.ndarray, y: np.ndarray) -> float:
    """Root mean squared error."""
    return np.sqrt(np.mean((x - y) ** 2))

def std_err(x: np.ndarray, y: np.ndarray) -> float:
    """Standard error of prediction differences."""
    return np.std(x - y) / np.sqrt(len(x))

def ncc_score(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized cross correlation (Pearson correlation)."""
    return pearsonr(x, y)[0]

def kt_score(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau correlation."""
    return kendalltau(x, y)[0]

def compute_scores(predictions, labels) -> dict:
    """Compute all evaluation metrics."""
    return {
        "RMSE": rmse_score(predictions, labels),
        "Std": std_err(predictions, labels),
        "NCC": ncc_score(predictions, labels),
        "KT": kt_score(predictions, labels),
    }

def evaluate(predictions_file: str, metadata_file: str, output_file: str):
    """Evaluate predictions against metadata correctness."""
    predictions_file = Path(predictions_file)
    metadata_file = Path(metadata_file)
    output_file = Path(output_file)

    # Load metadata
    with open(metadata_file, encoding="utf-8") as f:
        records = json.load(f)
    record_index = {r["signal"]: r for r in records}

    # Load predictions
    df = pd.read_csv(predictions_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename columns if needed
    if "signal_id" in df.columns:
        df.rename(columns={"signal_id": "signal"}, inplace=True)
    if "intelligibility_score" in df.columns:
        df.rename(columns={"intelligibility_score": "prediction"}, inplace=True)

    if "signal" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"Predictions file must contain 'signal' and 'prediction' columns. Found: {df.columns.tolist()}")

    # Add ground-truth correctness
    df["correctness"] = [record_index[s]["correctness"] for s in df.signal]

    # Compute scores
    scores = compute_scores(df["prediction"].to_numpy(), df["correctness"].to_numpy())

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Evaluation completed. Scores:\n{json.dumps(scores, indent=2)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    data_root = r"C:\Rudaina University\Semester 3\Artificial Intelligence\Cadenza\a2\clarity_data\cad_icassp_2026\cadenza_data"
    predictions_file = Path(data_root) / "val_predictions.csv"
    metadata_file = Path(data_root) / "metadata" / "cadenza_data.trainval_val_metadata.json"
    output_file = Path(data_root) / "cnn_mfcc_val_evaluation.json"

    evaluate(predictions_file, metadata_file, output_file)
