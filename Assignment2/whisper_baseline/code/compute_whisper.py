"""Compute the Whisper correctness scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import whisper
from omegaconf import DictConfig
from torch.nn import Module
from tqdm import tqdm

from clarity.utils.file_io import write_jsonl, write_signal
from recipes.cad_icassp_2026.baseline.shared_predict_utils import load_mixture
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


def compute_asr_for_signal(
    cfg: DictConfig, record: dict, signal: np.ndarray, asr_model: Module
) -> float:
    """Compute the correctness score for a given signal."""
    reference = record["prompt"]

    score_left = compute_correctness(
        signal[:, 0],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )
    score_right = compute_correctness(
        signal[:, 1],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )

    return np.max([score_left, score_right])


def compute_correctness(
    signal: np.ndarray,
    sample_rate: int,
    reference: str,
    asr_model: Module,
    contraction_file: str,
) -> float:
    """Compute the correctness score for a given signal."""
    scorer = SentenceScorer(contraction_file)

    # temporary FLAC for Whisper input
    path_temp = Path("temp.flac")
    write_signal(
        filename=path_temp, signal=signal, sample_rate=sample_rate, floating_point=False
    )

    # run whisper
    hypothesis = asr_model.transcribe(
        str(path_temp), fp16=False, language="en", temperature=0.0
    )["text"]

    # score the transcription
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits

    path_temp.unlink()  # clean up

    return results.hits / total_words


def run_asr_from_mixture(
    dataroot: Path, records: list, asr_model: Module, cfg: DictConfig
) -> list[dict]:
    """Compute Whisper correctness for all records and return results."""
    all_results = []

    for record in tqdm(records):
        signal_name = record["signal"]
        if not signal_name.endswith("_unproc.flac"):
            signal_name = signal_name.replace(".flac", "_unproc.flac")
            record["signal"] = signal_name

        signal_to_whisper, _ = load_mixture(dataroot, record, cfg)
        correct = compute_asr_for_signal(cfg, record, signal_to_whisper, asr_model)

        all_results.append({"signal": signal_name, "whisper": correct})

    return all_results


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_whisper(cfg: DictConfig) -> None:
    """Run Whisper to compute correctness scores."""
    # --- FIX SYSTEM NAME TO 'whisper' ---
    cfg.baseline.system = "whisper"

    logger.info(f"Running {cfg.baseline.system} baseline on {cfg.split} set...")

    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset
    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    # fix file names
    for record in records:
        record["signal"] = record["signal"].replace(".flac", "_unproc.flac")

    total_records = len(records)
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    print("Sample record:", records[0] if records else "No records found")
    print("Dataset root:", dataroot)
    logger.info(f"Computing scores for {len(records)} out of {total_records} signals")

    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model(cfg.baseline.whisper_version, device=device)
    logger.info(f"Using device: {device}")

    # compute all scores
    all_results = run_asr_from_mixture(dataroot, records, asr_model, cfg)

    # ensure output directory
    output_dir = Path("baseline_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_str = (
        f".{cfg.baseline.batch}_{cfg.baseline.n_batches}"
        if cfg.baseline.n_batches > 1
        else ""
    )
    results_file = output_dir / f"{cfg.data.dataset}.{cfg.split}.whisper{batch_str}.jsonl"

    # save results
    write_jsonl(str(results_file), all_results)
    logger.info(f"✅ Results written to: {results_file.resolve()}")


if __name__ == "__main__":
    run_compute_whisper()
