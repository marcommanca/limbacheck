"""
Shared test-set construction and metric computation, used by every evaluation
script (BERT pipeline, LLM baselines, CROS comparison) so all systems are scored
on the exact same held-out sentences with the exact same methodology.
"""

import difflib
import os
import random
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from data_prep.synthetic_errors import SyntheticErrorGenerator  # noqa: E402

TEST_CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "test_corpus.txt")
RECORDS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "runs", "records")

# Same pattern used by SardinianSpellchecker.split_text and SyntheticErrorGenerator,
# so word counts line up across original/corrupted/predicted text everywhere.
WORD_PATTERN = re.compile(
    r"['\-]?[a-zA-Zàèìòùáéíóú]+(?:['\-][a-zA-Zàèìòùáéíóú]+)*['\-]?|\d+|[^a-zA-Z0-9àèìòùáéíóú\s]"
)

TEST_SET_ERROR_RATE = 0.15
TEST_SET_SEED = 42
TEST_SET_MAX_CORRUPTION_ATTEMPTS = 20


def split_text(text):
    return WORD_PATTERN.findall(text)


def build_test_set(test_corpus_path=TEST_CORPUS_PATH):
    """
    Deterministically rebuilds the 160-sentence corrupted test set (error_rate=0.15,
    seed=42) used to report every metric in the paper. Any evaluation script
    calling this with the same test_corpus_path gets byte-identical corrupted
    sentences, which is what makes cross-system comparisons meaningful.
    """
    with open(test_corpus_path, "r", encoding="utf-8") as f:
        clean_sentences = [line.strip() for line in f if line.strip()]

    corruptor = SyntheticErrorGenerator(error_rate=TEST_SET_ERROR_RATE)
    random.seed(TEST_SET_SEED)

    test_data = []
    for clean_text in clean_sentences:
        corrupted_text = clean_text
        attempts = 0
        while corrupted_text == clean_text and attempts < TEST_SET_MAX_CORRUPTION_ATTEMPTS:
            res = corruptor.corrupt_sentence(clean_text)
            corrupted_text = res["corrupted_text"]
            attempts += 1
        test_data.append({"original_text": clean_text, "corrupted_text": corrupted_text})

    return pd.DataFrame(test_data)


def compute_correction_metrics(df):
    """
    Word-level Precision/Recall/F0.5 between original_text, corrupted_text and
    predicted_text columns of df. Falls back to an edit-distance alignment when a
    system's output doesn't preserve the original word count (e.g. free-form LLM
    generations).
    """
    tp, fp, fn, tn = 0, 0, 0, 0

    for _, row in df.iterrows():
        orig_words = split_text(row["original_text"])
        corr_words = split_text(row["corrupted_text"])
        pred_words = split_text(row["predicted_text"])

        if len(orig_words) == len(corr_words) == len(pred_words):
            for o, c, p in zip(orig_words, corr_words, pred_words):
                if c != o:  # the word was an error
                    if p == o:
                        tp += 1  # correctly fixed
                    elif p != c:
                        fp += 1
                        fn += 1  # attempted a fix but got it wrong (implicitly p != o)
                    else:
                        fn += 1  # error not detected, or detected but discarded by threshold
                else:  # the word was NOT an error
                    if p != o:
                        fp += 1  # a correct word got altered (over-correction)
                    else:
                        tn += 1
        else:
            def edit_distance(a, b):
                n, m = len(a), len(b)
                dp = [[0] * (m + 1) for _ in range(n + 1)]
                for k in range(n + 1):
                    dp[k][0] = k
                for k in range(m + 1):
                    dp[0][k] = k
                for i_ in range(1, n + 1):
                    for j_ in range(1, m + 1):
                        if a[i_ - 1] == b[j_ - 1]:
                            dp[i_][j_] = dp[i_ - 1][j_ - 1]
                        else:
                            dp[i_][j_] = 1 + min(dp[i_ - 1][j_], dp[i_][j_ - 1], dp[i_ - 1][j_ - 1])
                return dp[n][m]

            matcher = difflib.SequenceMatcher(None, orig_words, pred_words)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    for i in range(i1, i2):
                        if i < len(corr_words) and corr_words[i] != orig_words[i]:
                            tp += 1
                        else:
                            tn += 1
                else:
                    for i in range(i1, i2):
                        if i < len(corr_words) and corr_words[i] != orig_words[i]:
                            fn += 1
                    sub_corr = corr_words[i1:i2] if i2 <= len(corr_words) else []
                    sub_pred = pred_words[j1:j2]
                    fp += edit_distance(sub_corr, sub_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1 + 0.5 ** 2) * precision * recall / ((0.5 ** 2 * precision) + recall) if (precision + recall) > 0 else 0

    return {
        "Correction Precision": precision,
        "Correction Recall": recall,
        "Correction F0.5": f05,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def save_record(record, csv_filename):
    """Appends a one-row summary record to runs/records/<csv_filename>, creating it if needed."""
    os.makedirs(RECORDS_DIR, exist_ok=True)
    csv_path = os.path.join(RECORDS_DIR, csv_filename)
    df_record = pd.DataFrame(record)
    df_record.to_csv(csv_path, mode="a" if os.path.exists(csv_path) else "w",
                      header=not os.path.exists(csv_path), index=False)
    return csv_path
