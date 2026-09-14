"""
Step 3 of the pipeline: fine-tunes the MLM checkpoint (from pretrain_mlm.py) with
a Token Classification head, turning it into the error detector used at inference
time. Errors are injected dynamically at training time via SyntheticErrorGenerator
(a fresh corruption per batch, rather than a static corrupted dataset) — this was
found to generalize far better than a one-off static corpus: the static variant
reaches high validation scores but degrades markedly at test time, having
memorized the specific anomalies of the generated corpus rather than their typology.

Requires a GPU.
"""

import argparse
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
    EarlyStoppingCallback, Trainer, TrainingArguments,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from data_prep.synthetic_errors import SyntheticErrorGenerator  # noqa: E402

DEFAULT_TRAIN_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "train_corpus.txt")
DEFAULT_VAL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "val_corpus.txt")
DEFAULT_MODEL_CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-uncased")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-spellchecker")
RECORDS_CSV = os.path.join(SCRIPT_DIR, "..", "..", "runs", "records", "finetune_records.csv")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "runs", "plots", "finetune")

# How many times the (fixed, seeded) validation set is duplicated before
# corruption, so the eval set is large/varied enough for a stable metric
# despite each sentence only being corrupted once per copy. This set is used
# ONLY for early stopping / best-checkpoint selection here — test_corpus.txt
# is reserved exclusively for the final reported metrics (evaluate_pipeline.py).
VAL_SET_REPEATS = 6


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tunes the MLM checkpoint as a Token Classification error detector.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH,
                         help="Used only for early stopping / best-checkpoint selection during training. "
                              "Never the test set — see evaluation/grid_search.py and evaluate_pipeline.py "
                              "for the final, held-out evaluation on test_corpus.txt.")
    parser.add_argument("--model-checkpoint", default=DEFAULT_MODEL_CHECKPOINT,
                         help="Path to the continued-pretraining MLM checkpoint (see pretrain_mlm.py).")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--error-rate", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    return parser.parse_args()


def make_corruptor_and_tokenizer(tokenizer, corruptor):
    def dynamic_corruptor_and_tokenizer(batch):
        corrupted_words = []
        error_labels = []

        for text in batch["text"]:
            res = corruptor.corrupt_sentence(text)
            corrupted_words.append(res["corrupted_words_list"])
            error_labels.append(res["error_labels"])

        tokenized_inputs = tokenizer(
            corrupted_words, truncation=True, is_split_into_words=True, max_length=512,
        )

        labels = []
        for i, label_list in enumerate(error_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_idx is None else label_list[word_idx] for word_idx in word_ids]
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return dynamic_corruptor_and_tokenizer


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)

    true_labels = [[l for l in label if l != -100] for label in labels]
    true_predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    flat_true = [item for sublist in true_labels for item in sublist]
    flat_preds = [item for sublist in true_predictions for item in sublist]

    acc = accuracy_score(flat_true, flat_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(flat_true, flat_preds, average="binary", zero_division=0)
    # F0.5 weights precision more heavily than recall
    f0_5 = fbeta_score(flat_true, flat_preds, beta=0.5, average="binary", zero_division=0)

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "f0.5": f0_5}


def main():
    args = parse_args()
    print(f"Model will be saved to: {args.output_path}")

    raw_datasets = load_dataset("text", data_files={"train": args.train_path, "val": args.val_path})
    print(f"Training rows: {len(raw_datasets['train'])}")
    print(f"Validation rows: {len(raw_datasets['val'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    corruptor = SyntheticErrorGenerator(error_rate=args.error_rate)
    corrupt_and_tokenize = make_corruptor_and_tokenizer(tokenizer, corruptor)

    # Errors are re-generated on the fly for every training batch (data augmentation)
    raw_datasets["train"].set_transform(corrupt_and_tokenize)

    # The eval set is corrupted ONCE, with a fixed seed, so results are reproducible
    # and comparable across training runs.
    random.seed(42)
    expanded_val_dataset = concatenate_datasets([raw_datasets["val"]] * VAL_SET_REPEATS)
    static_val_dataset = expanded_val_dataset.map(
        corrupt_and_tokenize, batched=True, remove_columns=["text"],
    )

    id2label = {0: "CORRECT", 1: "ERROR"}
    label2id = {"CORRECT": 0, "ERROR": 1}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f0.5",
        bf16=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=static_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Starting fine-tuning:")
    trainer.train()

    # --- Metrics extraction, plot, and record-keeping ---
    history = trainer.state.log_history
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    eval_f05 = [x["eval_f0.5"] for x in history if "eval_f0.5" in x]
    eval_precision = [x["eval_precision"] for x in history if "eval_precision" in x]
    eval_recall = [x["eval_recall"] for x in history if "eval_recall" in x]

    actual_epochs = len(eval_loss)
    best_f05 = max(eval_f05)
    best_idx = eval_f05.index(best_f05)
    best_epoch = best_idx + 1

    epochs_x = [x["epoch"] for x in history if "loss" in x]
    eval_epochs_x = [x["epoch"] for x in history if "eval_loss" in x]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.normpath(args.model_checkpoint))
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"finetune_{timestamp}.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_x, train_loss, label="Training Loss")
    ax1.plot(eval_epochs_x, eval_loss, label="Validation Loss", marker="o")
    ax1.set_title("Fine-tune Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(eval_epochs_x, eval_f05, label="F0.5 Score", marker="o", color="green")
    ax2.plot(eval_epochs_x, eval_precision, label="Precision", marker="x", color="blue")
    ax2.plot(eval_epochs_x, eval_recall, label="Recall", marker="s", color="orange")
    ax2.set_title("Performance Metrics")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    record = {
        "Timestamp": [timestamp],
        "Base Model Name": [model_name],
        "Max Epochs": [args.epochs],
        "Stopped Epoch": [actual_epochs],
        "Best Epoch": [best_epoch],
        "Batch Size": [args.batch_size],
        "Grad Accumulation": [args.grad_accumulation],
        "Learning Rate": [args.learning_rate],
        "Final Train Loss": [train_loss[-1] if train_loss else None],
        "Best F0.5 Score": [best_f05],
        "Associated Precision": [eval_precision[best_idx]],
        "Associated Recall": [eval_recall[best_idx]],
        "Associated Val Loss": [eval_loss[best_idx]],
        "Plot Path": [plot_path],
    }
    df = pd.DataFrame(record)
    os.makedirs(os.path.dirname(RECORDS_CSV), exist_ok=True)
    df.to_csv(RECORDS_CSV, mode="a" if os.path.exists(RECORDS_CSV) else "w",
              header=not os.path.exists(RECORDS_CSV), index=False)
    print(f"Metrics saved to {RECORDS_CSV}")

    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Model and tokenizer saved to: {args.output_path}")


if __name__ == "__main__":
    main()
