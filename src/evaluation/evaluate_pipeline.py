"""
Step 5 of the pipeline: evaluates the hybrid BERT spellchecker (SardinianSpellchecker)
on the held-out test set, reporting Precision/Recall/F0.5, WER, CER, and average
inference latency.

The pipeline's threshold hyperparameters are NOT tuned here — see
evaluation/grid_search.py, which searches them on the validation set and writes
the winning combination to runs/pipeline_thresholds.json. This script reads that
file for its defaults (falling back to the last known-good values if grid_search.py
hasn't been run yet), so the numbers reported here are always a genuinely
held-out measurement, never inflated by tuning on the same sentences.

Requires the trained MLM + detector checkpoints (see pretrain_mlm.py / train_detector.py).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import jiwer
import torch
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from pipeline.pipeline import SardinianSpellchecker  # noqa: E402
from evaluation.common import build_test_set, compute_correction_metrics, save_record  # noqa: E402

DEFAULT_MLM_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-uncased")
DEFAULT_DETECTOR_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-spellchecker")
DEFAULT_DICT_PATHS = [
    os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries", "campidanese_standard.txt"),
    os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries", "campidanese_non_standard.txt"),
    os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries", "corpus_dict.txt"),
]
THRESHOLDS_JSON = os.path.join(SCRIPT_DIR, "..", "..", "runs", "pipeline_thresholds.json")

# Fallback only: used if runs/pipeline_thresholds.json doesn't exist yet (i.e.
# evaluation/grid_search.py has never been run against the current models).
FALLBACK_DETECTION_THRESHOLD = 0.95
FALLBACK_CORRECTION_THRESHOLD = -1.5
FALLBACK_MAX_LEVENSHTEIN_DISTANCE = 1
FALLBACK_MAX_CANDIDATES = 20


def load_threshold_defaults():
    if os.path.exists(THRESHOLDS_JSON):
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            t = json.load(f)
        print(f"Using thresholds from {THRESHOLDS_JSON} (grid search on the validation set, {t.get('timestamp', '?')}).")
        return (
            t["detection_threshold"], t["correction_threshold"],
            t["max_levenshtein_distance"], t.get("max_candidates", FALLBACK_MAX_CANDIDATES),
        )
    print(f"WARNING: {THRESHOLDS_JSON} not found — falling back to the last known-good thresholds. "
          f"Run `python -m src.evaluation.grid_search` first to tune them on the current models.")
    return (
        FALLBACK_DETECTION_THRESHOLD, FALLBACK_CORRECTION_THRESHOLD,
        FALLBACK_MAX_LEVENSHTEIN_DISTANCE, FALLBACK_MAX_CANDIDATES,
    )


def parse_args():
    default_det, default_corr, default_dist, default_cands = load_threshold_defaults()

    parser = argparse.ArgumentParser(description="Evaluates the hybrid BERT spellchecker pipeline on the held-out test set.")
    parser.add_argument("--mlm-checkpoint", default=DEFAULT_MLM_CHECKPOINT)
    parser.add_argument("--detector-checkpoint", default=DEFAULT_DETECTOR_CHECKPOINT)
    parser.add_argument("--dict-paths", nargs="+", default=DEFAULT_DICT_PATHS)
    parser.add_argument("--detection-threshold", type=float, default=default_det)
    parser.add_argument("--correction-threshold", type=float, default=default_corr)
    parser.add_argument("--max-levenshtein-distance", type=int, default=default_dist)
    parser.add_argument("--max-candidates", type=int, default=default_cands)
    parser.add_argument("--show-failures", action="store_true",
                         help="Print the sentences where the pipeline's output didn't match the ground truth.")
    return parser.parse_args()


def load_pipeline(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.mlm_checkpoint)
    mlm = AutoModelForMaskedLM.from_pretrained(args.mlm_checkpoint).to(device)
    detector = AutoModelForTokenClassification.from_pretrained(args.detector_checkpoint).to(device)

    vocabulary = set()
    for path in args.dict_paths:
        with open(path, "r", encoding="utf-8") as f:
            vocabulary.update(f.read().split())

    return SardinianSpellchecker(
        tokenizer=tokenizer, mlm=mlm, detector=detector, vocabulary=vocabulary, device=device,
        detection_threshold=args.detection_threshold,
        correction_threshold=args.correction_threshold,
        max_levenshtein_distance=args.max_levenshtein_distance,
        max_candidates=args.max_candidates,
        debug=False,
    )


def run_inference(spellchecker, df_test):
    predicted_texts, inference_times = [], []
    for text in df_test["corrupted_text"]:
        start_time = time.perf_counter()
        corrected = spellchecker.correct_text(text)
        end_time = time.perf_counter()
        predicted_texts.append(corrected)
        inference_times.append((end_time - start_time) * 1000)

    df_test = df_test.copy()
    df_test["predicted_text"] = predicted_texts
    df_test["latency_ms"] = inference_times
    return df_test


def print_report(metrics, wer_score, cer_score, baseline_wer, baseline_cer, avg_latency_ms):
    print("\n=== [END-TO-END METRICS] ===")
    print(f"Correction Precision: {metrics['Correction Precision']:.4f}")
    print(f"Correction Recall:    {metrics['Correction Recall']:.4f}")
    print(f"Correction F0.5:      {metrics['Correction F0.5']:.4f}\n")

    print("=== [CONFUSION MATRIX] ===")
    print(f"{'':<16} | {'Predicted: ERR':<14} | {'Predicted: CORR':<14} |")
    print("-" * 52)
    print(f"{'Ground truth: ERR':<16} | TP: {metrics['TP']:<10} | FN: {metrics['FN']:<10} |")
    print(f"{'Ground truth: CORR':<16} | FP: {metrics['FP']:<10} | TN: {metrics['TN']:<10} |\n")

    print(f"Word Error Rate (WER):      {wer_score:.4f} (Corrupted baseline: {baseline_wer:.4f})")
    print(f"Character Error Rate (CER): {cer_score:.4f} (Corrupted baseline: {baseline_cer:.4f})")
    print(f"Average inference latency:  {avg_latency_ms:.2f} ms/sentence")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pipeline models...")
    spellchecker = load_pipeline(args, device)

    print("Building the test set (160 sentences, seed=42, error_rate=0.15)...")
    df_test = build_test_set()

    df_test = run_inference(spellchecker, df_test)
    avg_latency_ms = df_test["latency_ms"].mean()

    metrics = compute_correction_metrics(df_test)
    references = df_test["original_text"].tolist()
    hypotheses = df_test["predicted_text"].tolist()
    corrupted = df_test["corrupted_text"].tolist()

    wer_score = jiwer.wer(references, hypotheses)
    cer_score = jiwer.cer(references, hypotheses)
    baseline_wer = jiwer.wer(references, corrupted)
    baseline_cer = jiwer.cer(references, corrupted)

    print_report(metrics, wer_score, cer_score, baseline_wer, baseline_cer, avg_latency_ms)

    if args.show_failures:
        df_test["success"] = df_test["original_text"] == df_test["predicted_text"]
        print("\n=== Failed examples (False Positives / False Negatives) ===")
        for idx, row in df_test[~df_test["success"]].iterrows():
            print(f"\nID: {idx}")
            print(f"Original:  {row['original_text']}")
            print(f"Corrupted: {row['corrupted_text']}")
            print(f"Predicted: {row['predicted_text']}")

    record = {
        "Timestamp": [datetime.now().strftime("%Y%m%d_%H%M%S")],
        "MLM Model": [os.path.basename(os.path.normpath(args.mlm_checkpoint))],
        "Spellchecker Model": [os.path.basename(os.path.normpath(args.detector_checkpoint))],
        "Avg Latency (ms)": [round(avg_latency_ms, 2)],
        "Correction F0.5": [round(metrics["Correction F0.5"], 4)],
        "Correction Precision": [round(metrics["Correction Precision"], 4)],
        "Correction Recall": [round(metrics["Correction Recall"], 4)],
        "WER": [round(wer_score, 4)],
        "CER": [round(cer_score, 4)],
        "Baseline WER": [round(baseline_wer, 4)],
        "Baseline CER": [round(baseline_cer, 4)],
        "TP": [metrics["TP"]], "FP": [metrics["FP"]], "FN": [metrics["FN"]], "TN": [metrics["TN"]],
    }
    csv_path = save_record(record, "testing_records.csv")
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    main()
