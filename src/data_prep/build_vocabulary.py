"""
Builds the corpus-derived vocabulary and reports coverage metrics across all
vocabulary sources used by the pipeline's candidate-generation module (SymSpell).

This script only extracts the vocabulary from the training corpus and analyzes
overlap with the other dictionaries already present in data/dictionaries/. To
(re)generate the scraped dictionaries themselves, see scrape_dictionaries.py.
"""

import argparse
import os
import re
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "train_corpus.txt")
DICT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "dictionaries")
CORPUS_DICT_OUTPUT = os.path.join(DICT_DIR, "corpus_dict.txt")

# Matches the same word pattern used by the spellchecker's tokenizer, so the
# vocabulary lines up exactly with what the pipeline will look up at inference time.
WORD_PATTERN = re.compile(r"['\-]?[a-zA-Zàèìòùáéíóú]+(?:['\-][a-zA-Zàèìòùáéíóú]+)*['\-]?")


def extract_corpus_vocabulary(file_path):
    vocab_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            words = WORD_PATTERN.findall(line.lower())
            vocab_set.update(w for w in words if w.replace("'", "").replace("-", "").isalpha())
    return vocab_set


def load_dict(filename):
    path = os.path.join(DICT_DIR, filename)
    if not os.path.exists(path):
        print(f"[build_vocabulary] WARNING: {path} not found, skipping (run scrape_dictionaries.py first)")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def strip_accents(word):
    nfkd_form = unicodedata.normalize("NFKD", word)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def parse_args():
    parser = argparse.ArgumentParser(description="Extracts the corpus vocabulary and reports dictionary coverage metrics.")
    parser.add_argument("--corpus-path", default=CORPUS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    print("[build_vocabulary] Extracting vocabulary from the training corpus...")
    corpus_vocab = extract_corpus_vocabulary(args.corpus_path)
    with open(CORPUS_DICT_OUTPUT, "w", encoding="utf-8") as f:
        for word in sorted(corpus_vocab):
            f.write(word + "\n")
    print(f"[build_vocabulary] Extracted {len(corpus_vocab)} unique words -> {CORPUS_DICT_OUTPUT}")

    # Coverage analysis against the other dictionary sources (informational only;
    # NOT used by the inference pipeline, which loads its own dict_paths directly)
    wiki_dict = load_dict("wikipedia_dict.txt")
    ditz_in_linia = load_dict("ditzionariu_in_linia.txt")
    camp_std_dict = load_dict("campidanese_standard.txt")
    camp_non_std_dict = load_dict("campidanese_non_standard.txt")

    if not wiki_dict:
        return

    official_dict = ditz_in_linia | camp_std_dict | camp_non_std_dict
    wiki_in_official = wiki_dict & official_dict
    wiki_not_in_official = wiki_dict - official_dict
    global_dict = wiki_dict | official_dict

    print("\n--- INDIVIDUAL DICTIONARY SIZES ---")
    print(f"Wikipedia:                {len(wiki_dict)} words")
    print(f"Ditzionariu in Linia:     {len(ditz_in_linia)} words")
    print(f"Campidanese Standard:     {len(camp_std_dict)} words")
    print(f"Campidanese Non-Standard: {len(camp_non_std_dict)} words\n")

    print("--- COMBINED METRICS ---")
    print(f"Total unique words across reference dictionaries: {len(official_dict)}")
    print(f"Total global vocabulary (all dictionaries merged): {len(global_dict)}\n")

    print("--- WIKIPEDIA COVERAGE ---")
    coverage_pct = (len(wiki_in_official) / len(wiki_dict)) * 100
    print(f"Wikipedia words present in reference dictionaries:     {len(wiki_in_official)} ({coverage_pct:.2f}%)")
    print(f"Wikipedia words NOT present in reference dictionaries: {len(wiki_not_in_official)}\n")

    wiki_dict_no_acc = {strip_accents(w) for w in wiki_dict}
    official_dict_no_acc = {strip_accents(w) for w in official_dict}
    wiki_in_official_no_acc = wiki_dict_no_acc & official_dict_no_acc

    print("--- ACCENT-INSENSITIVE COVERAGE ---")
    print(f"Unique Wikipedia words (no accents):        {len(wiki_dict_no_acc)}")
    print(f"Unique reference-dict words (no accents):   {len(official_dict_no_acc)}")
    coverage_no_acc_pct = (len(wiki_in_official_no_acc) / len(wiki_dict_no_acc)) * 100
    print(f"Wikipedia words present (no accents): {len(wiki_in_official_no_acc)} ({coverage_no_acc_pct:.2f}%)")


if __name__ == "__main__":
    main()
