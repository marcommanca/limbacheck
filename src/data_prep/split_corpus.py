"""
Splits the cleaned corpus into a training corpus, a validation set, and a
held-out test set.

Extracted from what used to be the first cells of modeling/tokenizer.ipynb: this
split is load-bearing for the whole project — every evaluation script (BERT
pipeline, LLM baselines, CROS comparison) reports metrics on the exact test set
produced here, and the pipeline's inference-time thresholds are tuned exclusively
on the validation set (see evaluation/grid_search.py), never on test — so the
seeds and split ratios must never change without re-running every downstream
stage.

The split is nested to keep the test set stable: test is carved off the full
corpus first (same logic/seed as the original train/test split), then val is
carved off what remains as train. This means TEST_SIZE alone controls how much
of the full corpus ends up in test, while VAL_SIZE controls how much of the
*remainder* (not of the full corpus) ends up in val.
"""

import argparse
import os
import re

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TXT = os.path.join(SCRIPT_DIR, "..", "..", "data", "datasets", "campidanese_cleaned.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "training")

TEST_SIZE = 0.05  # fraction of the full corpus
VAL_SIZE = 0.05  # fraction of the full corpus (taken out of the train remainder, i.e. ~0.0526 of it)
SPLIT_SEED = 42
MIN_SENTENCE_LEN = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Splits the cleaned corpus into train_corpus.txt / val_corpus.txt / test_corpus.txt (seed=42).")
    parser.add_argument("--input-txt", default=INPUT_TXT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                         help="WARNING: changing this invalidates every reported metric, which was computed on the default 5%% test split.")
    parser.add_argument("--val-size", type=float, default=VAL_SIZE,
                         help="Fraction of the FULL corpus to carve into val_corpus.txt (taken out of the train remainder, after test is set aside).")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED,
                         help="WARNING: changing this invalidates every reported metric, which was computed with seed=42.")
    return parser.parse_args()


def articles_to_sentences(dataset):
    sentences = []
    for entry in dataset:
        text = entry["text"]
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])", text) if len(s.strip()) > MIN_SENTENCE_LEN)
    return sentences


def main():
    args = parse_args()
    input_txt, output_dir = args.input_txt, args.output_dir

    print(f"[split_corpus] Loading {input_txt}")
    dataset = load_dataset("text", data_files=input_txt, split="train")

    # Step 1: carve off the test set first, exactly as before — this keeps
    # test_corpus.txt identical regardless of val_size, so it stays comparable
    # across runs and the model never influences its own final-report split.
    split_1 = dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    train_full_dataset = split_1["train"]
    test_dataset = split_1["test"]

    # Step 2: carve val out of the remaining train articles. val_size is
    # expressed as a fraction of the FULL corpus, so convert it to a fraction
    # of train_full_dataset before splitting.
    val_fraction_of_remainder = args.val_size / (1 - args.test_size)
    split_2 = train_full_dataset.train_test_split(test_size=val_fraction_of_remainder, seed=args.seed)
    train_dataset = split_2["train"]
    val_dataset = split_2["test"]

    train_path = os.path.join(output_dir, "train_corpus.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        for entry in train_dataset:
            f.write(entry["text"] + "\n")

    # val/test hold full articles/paragraphs; break them into individual
    # sentences so they match the granularity evaluated everywhere else.
    val_sentences = articles_to_sentences(val_dataset)
    val_path = os.path.join(output_dir, "val_corpus.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_sentences:
            f.write(s + "\n")

    test_sentences = articles_to_sentences(test_dataset)
    test_path = os.path.join(output_dir, "test_corpus.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        for s in test_sentences:
            f.write(s + "\n")

    print(f"[split_corpus] Training articles:   {len(train_dataset)} -> {train_path}")
    print(f"[split_corpus] Validation sentences: {len(val_sentences)} -> {val_path}")
    print(f"[split_corpus] Test sentences:       {len(test_sentences)} -> {test_path}")


if __name__ == "__main__":
    main()
