"""
Builds the train/val/test CSVs used by LMSpell to fine-tune the LLM baselines
(LLaMA, Gemma), in the "corrupted -> expected" format LMSpell's trainers expect.

The train/val split is corrupted once per row (data augmentation isn't dynamic
here, unlike train_detector.py, since LMSpell trains from a fixed CSV). The test
split reuses the exact same seed/logic as evaluation/common.py.build_test_set so
LLM baselines and the BERT pipeline are evaluated on identical corrupted sentences.
"""

import argparse
import os
import random
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from data_prep.synthetic_errors import SyntheticErrorGenerator  # noqa: E402

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "training")
TRAIN_CORPUS_PATH = os.path.join(DATA_DIR, "train_corpus.txt")
TEST_CORPUS_PATH = os.path.join(DATA_DIR, "test_corpus.txt")

VAL_SIZE = 0.10
ERROR_RATE = 0.15
SEED = 42


def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Builds the train/val/test CSVs used to fine-tune the LLM baselines via LMSpell.")
    parser.add_argument("--train-corpus-path", default=TRAIN_CORPUS_PATH)
    parser.add_argument("--test-corpus-path", default=TEST_CORPUS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_train = read_corpus(args.train_corpus_path)
    raw_test = read_corpus(args.test_corpus_path)
    print(f"Loaded clean training sentences: {len(raw_train)}")
    print(f"Loaded clean test sentences: {len(raw_test)}")

    random.seed(SEED)
    error_generator = SyntheticErrorGenerator(error_rate=ERROR_RATE)

    train_data = []
    for sentence in raw_train:
        res = error_generator.corrupt_sentence(sentence)
        train_data.append({"text": res["corrupted_text"], "expected": res["original_text"]})
    full_train_df = pd.DataFrame(train_data)
    train_df, val_df = train_test_split(full_train_df, test_size=VAL_SIZE, random_state=SEED)

    # Re-seed for full reproducibility with evaluation/common.py.build_test_set,
    # which uses the same "retry until corrupted" logic on the same test corpus.
    random.seed(SEED)
    test_data = []
    for sentence in raw_test:
        corrupted_text = sentence
        attempts = 0
        while corrupted_text == sentence and attempts < 20:
            res = error_generator.corrupt_sentence(sentence)
            corrupted_text = res["corrupted_text"]
            attempts += 1
        test_data.append({"text": corrupted_text, "expected": sentence})
    test_df = pd.DataFrame(test_data)

    train_df.to_csv(os.path.join(DATA_DIR, "trainLMSpell.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join(DATA_DIR, "valLMSpell.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(DATA_DIR, "testLMSpell.csv"), index=False, encoding="utf-8")

    print(f"CSVs generated. Final sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()
