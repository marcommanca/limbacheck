"""
Evaluates a fine-tuned LLM baseline (LLaMA/Gemma, fine-tuned via train_llm.py) on
the same test set and with the same metrics as the BERT pipeline, for a direct
comparison. Loads the model directly with unsloth/transformers (no LMSpell
framework needed at inference time, since the LoRA weights are already trained
and saved locally).

Requires a GPU and the model checkpoint produced by train_llm.py.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import jiwer
import pandas as pd
import torch
from unsloth import FastLanguageModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from evaluation.common import compute_correction_metrics, save_record  # noqa: E402

TEST_CSV_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "testLMSpell.csv")

PROMPT_TEMPLATE = (
    "You are an expert Campidanese Sardinian spell corrector. Below is a sentence "
    "in Campidanese Sardinian language. It may or may not have a spelling mistake. "
    "Give the corrected output in Campidanese Sardinian.\n\n### Text:\n{}\n\n### Output:\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluates a fine-tuned LLM baseline against the BERT pipeline's test set.")
    parser.add_argument("--model-path", required=True,
                         help="Path to the fine-tuned model, e.g. ../../models/lmspell_models/sardinian_llama_3.1_8B")
    parser.add_argument("--test-csv", default=TEST_CSV_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def load_test_set(test_csv_path):
    df_raw = pd.read_csv(test_csv_path)
    # 'expected' = ground-truth corrected sentence, 'text' = corrupted sentence
    return pd.DataFrame({
        "original_text": df_raw["expected"],
        "corrupted_text": df_raw["text"],
    })


def make_corrector(model, tokenizer, max_new_tokens):
    def correct_sentence(sentence):
        prompt = PROMPT_TEMPLATE.format(sentence)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        if "### Output:" in result:
            return result.split("### Output:")[1].strip()
        return result.strip()
    return correct_sentence


def main():
    args = parse_args()

    df_test = load_test_set(args.test_csv)
    print(f"Loaded test set from {args.test_csv} with {len(df_test)} sentences.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # 2x faster inference mode
    print(f"Model {args.model_path} loaded successfully.")

    correct_sentence = make_corrector(model, tokenizer, args.max_new_tokens)

    predicted_texts, inference_times = [], []
    for text in df_test["corrupted_text"]:
        start_time = time.perf_counter()
        corrected = correct_sentence(text)
        end_time = time.perf_counter()
        predicted_texts.append(corrected)
        inference_times.append((end_time - start_time) * 1000)

    df_test["predicted_text"] = predicted_texts
    df_test["latency_ms"] = inference_times
    avg_latency_ms = sum(inference_times) / len(inference_times)
    print(f"Average inference latency: {avg_latency_ms:.2f} ms/sentence")

    metrics = compute_correction_metrics(df_test)
    references = df_test["original_text"].tolist()
    hypotheses = df_test["predicted_text"].tolist()
    corrupted = df_test["corrupted_text"].tolist()

    wer_score = jiwer.wer(references, hypotheses)
    cer_score = jiwer.cer(references, hypotheses)
    baseline_wer = jiwer.wer(references, corrupted)
    baseline_cer = jiwer.cer(references, corrupted)

    print("\n=== [END-TO-END METRICS] ===")
    print(f"Correction Precision: {metrics['Correction Precision']:.4f}")
    print(f"Correction Recall:    {metrics['Correction Recall']:.4f}")
    print(f"Correction F0.5:      {metrics['Correction F0.5']:.4f}\n")
    print(f"Word Error Rate (WER):      {wer_score:.4f} (Corrupted baseline: {baseline_wer:.4f})")
    print(f"Character Error Rate (CER): {cer_score:.4f} (Corrupted baseline: {baseline_cer:.4f})")

    model_name = os.path.basename(os.path.normpath(args.model_path))
    record = {
        "Timestamp": [datetime.now().strftime("%Y%m%d_%H%M%S")],
        "Model": [model_name],
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
    csv_path = save_record(record, "lmspell_testing_records.csv")
    print(f"\nMetrics saved to {csv_path}")


if __name__ == "__main__":
    main()
