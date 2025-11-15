"""Make intelligibility predictions from baseline features (e.g., STOI, Whisper) with train/val split."""

from __future__ import annotations
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    LogisticModel,
    load_dataset_with_score,
)

log = logging.getLogger(__name__)


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def predict_train_val(cfg: DictConfig):
    """
    Predict intelligibility for baselines with internal train/validation split.

    - Trains logistic regression on 80% of the train data.
    - Evaluates on 20% validation split (saves predictions and metadata for local evaluation).
    - Optionally predicts on eval set (if eval metadata & features exist).
    """

    # ------------------------------------------------------------
    # Load train data
    # ------------------------------------------------------------
    log.info("Loading dataset (train)...")
    records_train_df = load_dataset_with_score(cfg, "train")

    # ------------------------------------------------------------
    # Split into train/validation
    # ------------------------------------------------------------
    log.info("Splitting train data into train/validation (80/20)...")
    train_df, val_df = train_test_split(records_train_df, test_size=0.2, random_state=42)

    # ------------------------------------------------------------
    # Save metadata for validation subset
    # ------------------------------------------------------------
    metadata_dir = Path(cfg.data.cadenza_data_root) / cfg.data.dataset / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    val_meta_file = metadata_dir / f"{cfg.data.dataset}.trainval_val_metadata.json"
    val_df.to_json(val_meta_file, orient="records", lines=False)
    log.info(f"Validation metadata saved to {val_meta_file}")

    # ------------------------------------------------------------
    # Fit logistic regression model
    # ------------------------------------------------------------
    log.info("Training logistic regression model...")
    model = LogisticModel()
    model.fit(train_df[f"{cfg.baseline.system}"], train_df.correctness)

    # ------------------------------------------------------------
    # Predict on validation subset
    # ------------------------------------------------------------
    log.info("Predicting on validation subset...")
    val_df["predicted_correctness"] = model.predict(val_df[f"{cfg.baseline.system}"])

    val_output_file = Path(cfg.data.cadenza_data_root) / cfg.data.dataset / f"{cfg.data.dataset}.{cfg.baseline.system}.trainval.val.predict2.csv"
    val_df[["signal", "predicted_correctness"]].to_csv(
        val_output_file,
        index=False,
        header=["signal_ID", "intelligibility_score"],
        mode="w",
    )
    log.info(f"Validation predictions saved to {val_output_file}")

    # ------------------------------------------------------------
    # Optionally predict on eval set (if available)
    # ------------------------------------------------------------
    try:
        log.info("Attempting to load eval data...")
        records_eval_df = load_dataset_with_score(cfg, "eval")

        log.info("Predicting on eval set...")
        records_eval_df["predicted_correctness"] = model.predict(
            records_eval_df[f"{cfg.baseline.system}"]
        )

        eval_output_file = Path(cfg.data.cadenza_data_root) / cfg.data.dataset / f"{cfg.data.dataset}.{cfg.baseline.system}.trainval.eval.predict.csv"
        records_eval_df[["signal", "predicted_correctness"]].to_csv(
            eval_output_file,
            index=False,
            header=["signal_ID", "intelligibility_score"],
            mode="w",
        )
        log.info(f"Eval predictions saved to {eval_output_file}")

    except FileNotFoundError:
        log.warning("Eval metadata or features not found — skipping eval prediction.")

    log.info("Prediction pipeline completed successfully.")


if __name__ == "__main__":
    predict_train_val()
