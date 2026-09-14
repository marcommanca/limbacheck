"""
Cleans and normalizes the raw bilingual Campidanese/Italian corpus, producing a
single-column plain-text file (one sentence per line) used downstream for
tokenizer training and train/test splitting.

Source: data/datasets/campidanese.tsv (bilingual corpus, "Po Cantu Biddanoa" by
Benvenuto Lobina), Sardinian column only.
"""

import argparse
import os
import re

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TSV = os.path.join(SCRIPT_DIR, "..", "..", "data", "datasets", "campidanese.tsv")
OUTPUT_TXT = os.path.join(SCRIPT_DIR, "..", "..", "data", "datasets", "campidanese_cleaned.txt")

MIN_WORD_COUNT = 3


def clean_text(text):
    """Lowercases and normalizes punctuation/whitespace for an uncased BERT pipeline."""
    text = str(text).lower()

    # Normalize curly quotes and backticks to straight ones
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("“", '"').replace("”", '"')

    # Normalize en-dash and ellipsis
    text = text.replace("–", "-")
    text = text.replace("…", "...")

    # Drop any other non-standard character
    text = re.sub(r"[^a-z0-9àèìòùáéíóú'\-.,!?;:()\" ]", " ", text)

    # Collapse multiple whitespace into one
    text = re.sub(r"\s+", " ", text).strip()

    # Drop unmatched quotes/parentheses
    if text.count('"') % 2 != 0:
        text = text.replace('"', "")
    if text.count("(") != text.count(")"):
        text = text.replace("(", "").replace(")", "")

    # Strip dots from acronyms (e.g. "a.n.c" -> "anc") so tokenization isn't fragmented
    text = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), text)

    # Tidy up spacing around punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([!?])\.", r"\1", text)
    text = re.sub(r'([!?])"\.', r'\1"', text)
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"^([\'\"(]*\s*)[.,!?;:\-]+\s*", r"\1", text)

    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Cleans the raw bilingual corpus into a single-column plain-text file.")
    parser.add_argument("--input-tsv", default=INPUT_TSV)
    parser.add_argument("--output-txt", default=OUTPUT_TXT)
    return parser.parse_args()


def main():
    args = parse_args()
    global INPUT_TSV, OUTPUT_TXT
    INPUT_TSV, OUTPUT_TXT = args.input_tsv, args.output_txt

    print(f"[clean_corpus] Loading {INPUT_TSV}")
    df = pd.read_csv(INPUT_TSV, sep="\t")

    if "Italiano" in df.columns:
        df = df.drop(columns=["Italiano"])
    df = df.rename(columns={"Sardo": "clean_text"})

    df = df.drop_duplicates(subset=["clean_text"])
    df["clean_text"] = df["clean_text"].apply(clean_text)

    # Keep only sentences with enough words to give the MLM sufficient context
    df["word_count"] = df["clean_text"].apply(lambda s: len(re.findall(r"\w+", str(s))))
    df = df[df["word_count"] >= MIN_WORD_COUNT]
    df = df.drop(columns=["word_count"])

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for text in df["clean_text"]:
            f.write(f"{text}\n")

    print(f"[clean_corpus] Saved {len(df)} sentences to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
