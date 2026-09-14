"""
Tunes the hybrid pipeline's inference-time hyperparameters — detection
threshold, correction threshold, max Levenshtein distance — by grid search on
the VALIDATION set (val_corpus.txt), never on the test set. This is what makes
evaluate_pipeline.py's final reported metrics an honest, held-out measurement:
the thresholds are chosen before test_corpus.txt is ever touched.

Writes:
  - runs/records/grid_search_records.csv : one summary row per run (best combo + score)
  - runs/details/grid_search_<timestamp>.csv : the full grid (every combination tried)
  - runs/pipeline_thresholds.json : the winning thresholds, read automatically as
    the defaults by evaluate_pipeline.py (and by anything else that wants the
    current best-known thresholds, e.g. the GUI)

Requires the trained MLM + detector checkpoints (see pretrain_mlm.py / train_detector.py).
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from pipeline.pipeline import SardinianSpellchecker  # noqa: E402
from evaluation.common import build_test_set, compute_correction_metrics  # noqa: E402

DEFAULT_MLM_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-uncased")
DEFAULT_DETECTOR_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-spellchecker")
DEFAULT_VAL_CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "val_corpus.txt")
DEFAULT_DICT_PATHS = [
    os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries", "campidanese_standard.txt"),
    os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries", "campidanese_non_standard.txt"),
    os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries", "corpus_dict.txt"),
]

RECORDS_CSV = os.path.join(SCRIPT_DIR, "..", "..", "runs", "records", "grid_search_records.csv")
DETAILS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "runs", "details")
THRESHOLDS_JSON = os.path.join(SCRIPT_DIR, "..", "..", "runs", "pipeline_thresholds.json")

DETECTION_THRESHOLDS = [0.75, 0.8, 0.85, 0.9, 0.95]
CORRECTION_THRESHOLDS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
MAX_LEVENSHTEIN_DISTANCES = [1, 2, 3]
MAX_CANDIDATES = 20  # not searched: kept fixed, same as evaluate_pipeline.py's default


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-searches the pipeline's thresholds on the validation set.")
    parser.add_argument("--mlm-checkpoint", default=DEFAULT_MLM_CHECKPOINT)
    parser.add_argument("--detector-checkpoint", default=DEFAULT_DETECTOR_CHECKPOINT)
    parser.add_argument("--dict-paths", nargs="+", default=DEFAULT_DICT_PATHS)
    parser.add_argument("--val-corpus", default=DEFAULT_VAL_CORPUS_PATH)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
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
        detection_threshold=DETECTION_THRESHOLDS[0],
        correction_threshold=CORRECTION_THRESHOLDS[0],
        max_levenshtein_distance=MAX_LEVENSHTEIN_DISTANCES[0],
        max_candidates=args.max_candidates,
        debug=False,
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pipeline models...")
    spellchecker = load_pipeline(args, device)

    print(f"Building the validation set from {args.val_corpus} (seed=42, error_rate=0.15)...")
    df_val = build_test_set(args.val_corpus)
    print(f"Validation sentences: {len(df_val)}")

    grid = list(itertools.product(DETECTION_THRESHOLDS, CORRECTION_THRESHOLDS, MAX_LEVENSHTEIN_DISTANCES))
    print(f"\nStarting grid search over {len(grid)} combinations on the VALIDATION set (never on test_corpus.txt)\n")

    results = []
    for det_thresh, corr_thresh, max_dist in grid:
        spellchecker.detection_threshold = det_thresh
        spellchecker.correction_threshold = corr_thresh
        spellchecker.max_levenshtein_distance = max_dist

        df_temp = df_val.copy()
        df_temp["predicted_text"] = [spellchecker.correct_text(t) for t in df_val["corrupted_text"]]

        metrics = compute_correction_metrics(df_temp)
        results.append({
            "detection_thresh": det_thresh, "correction_thresh": corr_thresh, "max_dist": max_dist,
            "F0.5": metrics["Correction F0.5"], "Precision": metrics["Correction Precision"],
            "Recall": metrics["Correction Recall"], "TP": metrics["TP"], "FP": metrics["FP"], "FN": metrics["FN"],
        })

    results.sort(key=lambda r: r["F0.5"], reverse=True)

    print("\n[TOP 5 GRID SEARCH RESULTS]")
    for r in results[:5]:
        print(r)

    best = results[0]
    print("\nOptimal parameters (on the validation set):")
    print(f"- Detection Threshold:      {best['detection_thresh']}")
    print(f"- Correction Threshold:     {best['correction_thresh']}")
    print(f"- Max Levenshtein Distance: {best['max_dist']}")
    print(f"- Validation F0.5 Score:    {best['F0.5']:.4f} (Precision: {best['Precision']:.4f}, Recall: {best['Recall']:.4f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full grid, for inspection/debugging.
    os.makedirs(DETAILS_DIR, exist_ok=True)
    details_path = os.path.join(DETAILS_DIR, f"grid_search_{timestamp}.csv")
    pd.DataFrame(results).to_csv(details_path, index=False)
    print(f"\nFull grid saved to: {details_path}")

    # One summary row, following the same append-only convention as the other stages.
    os.makedirs(os.path.dirname(RECORDS_CSV), exist_ok=True)
    record = {
        "Timestamp": [timestamp],
        "Val Set Size": [len(df_val)],
        "Combinations Tried": [len(grid)],
        "Best Detection Threshold": [best["detection_thresh"]],
        "Best Correction Threshold": [best["correction_thresh"]],
        "Best Max Levenshtein Distance": [best["max_dist"]],
        "Val F0.5": [round(best["F0.5"], 4)],
        "Val Precision": [round(best["Precision"], 4)],
        "Val Recall": [round(best["Recall"], 4)],
    }
    pd.DataFrame(record).to_csv(RECORDS_CSV, mode="a" if os.path.exists(RECORDS_CSV) else "w",
                                 header=not os.path.exists(RECORDS_CSV), index=False)
    print(f"Summary row saved to: {RECORDS_CSV}")

    # The contract evaluate_pipeline.py (and anything else) reads its defaults from.
    os.makedirs(os.path.dirname(THRESHOLDS_JSON), exist_ok=True)
    with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "detection_threshold": best["detection_thresh"],
            "correction_threshold": best["correction_thresh"],
            "max_levenshtein_distance": best["max_dist"],
            "max_candidates": args.max_candidates,
            "source": "grid_search.py",
            "timestamp": timestamp,
            "val_f0.5": round(best["F0.5"], 4),
        }, f, indent=2)
    print(f"Thresholds written to: {THRESHOLDS_JSON} (evaluate_pipeline.py will pick these up automatically)")


if __name__ == "__main__":
    main()
