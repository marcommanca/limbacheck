"""
OPTIONAL ablation experiment: trains a custom Unigram tokenizer on the Campidanese
corpus and reports its fertility / [UNK] rate on the held-out test set.

NOT used by the final pipeline: exploratory testing showed keeping the native
tokenizer of the base Italian BERT checkpoint (dbmdz/bert-base-italian-uncased)
outperformed a from-scratch Sardinian tokenizer, so the pipeline uses that native
tokenizer throughout. This script is kept for transparency/reproducibility of
that comparison, not as a required pipeline stage.
"""

import argparse
import os
import re

from tokenizers import Tokenizer, Regex, decoders, models, normalizers, pre_tokenizers, processors, trainers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "datasets", "campidanese_cleaned.txt")
DEFAULT_TRAIN_CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "train_corpus.txt")
DEFAULT_TEST_CORPUS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "test_corpus.txt")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "tokenizers", "unigram-tokenizer-12k-campidanese-uncased.json")

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]
WORD_PATTERN = re.compile(r"[a-zA-Zàèìòùáéíóú]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Trains and evaluates a custom Unigram tokenizer (ablation experiment, not used in the final pipeline).")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH,
                         help="Full cleaned corpus, used for the final full-corpus training run.")
    parser.add_argument("--train-corpus-path", default=DEFAULT_TRAIN_CORPUS_PATH,
                         help="Train-only split, used for the fertility/[UNK] evaluation run.")
    parser.add_argument("--test-corpus-path", default=DEFAULT_TEST_CORPUS_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--skip-eval", action="store_true",
                         help="Skip the fertility/[UNK] evaluation pass and go straight to the full-corpus training run.")
    return parser.parse_args()


def build_tokenizer(vocab_size):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Lowercase(),
        normalizers.Replace("`", "'"),
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(),
        pre_tokenizers.Punctuation(),
    ])
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, unk_token="[UNK]")
    return tokenizer, trainer


def finalize_tokenizer(tokenizer):
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()


def evaluate(tokenizer, test_corpus_path):
    with open(test_corpus_path, "r", encoding="utf-8") as f:
        test_sentences = [line.strip() for line in f if line.strip()]

    total_logical_words = 0
    total_subwords = 0
    unk_count = 0
    total_generated_tokens = 0

    for sentence in test_sentences:
        words = WORD_PATTERN.findall(sentence)
        total_logical_words += len(words)

        output = tokenizer.encode(sentence)
        total_generated_tokens += len(output.ids)

        for token in output.tokens:
            if token == "[UNK]":
                unk_count += 1
            elif token not in SPECIAL_TOKENS and re.search(r"[a-zA-Zàèìòùáéíóú]", token):
                # Fertility only counts alphabetic subwords
                total_subwords += 1

    fertility = total_subwords / total_logical_words
    unk_percentage = (unk_count / total_generated_tokens) * 100

    print("--- TOKENIZER METRICS ---")
    print(f"Logical words analyzed in the test set:        {total_logical_words}")
    print(f"Alphabetic tokens generated:                    {total_subwords}")
    print(f"Total tokens (incl. special/punctuation):       {total_generated_tokens}")
    print("--------------------------")
    print(f"FERTILITY (subwords / words): {fertility:.4f}  (optimal: 1.10 - 1.50)")
    print(f"[UNK] PERCENTAGE:              {unk_percentage:.4f}%  (optimal: < 0.1%)")


def main():
    args = parse_args()

    if not args.skip_eval:
        tokenizer, trainer = build_tokenizer(args.vocab_size)
        print(f"Training on the train-only split ({args.train_corpus_path}) for evaluation purposes...")
        tokenizer.train([args.train_corpus_path], trainer)
        finalize_tokenizer(tokenizer)
        evaluate(tokenizer, args.test_corpus_path)

    print(f"\nTraining final tokenizer on the full corpus ({args.dataset_path})...")
    tokenizer, trainer = build_tokenizer(args.vocab_size)
    tokenizer.train([args.dataset_path], trainer)
    finalize_tokenizer(tokenizer)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer.save(args.output_path)
    print(f"Tokenizer saved to: {args.output_path}")


if __name__ == "__main__":
    main()
