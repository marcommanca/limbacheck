"""
Step 1 of the pipeline: continued pre-training of a Masked Language Model on the
clean Campidanese corpus (transfer learning from an Italian BERT checkpoint).

Requires a GPU. Produces the MLM checkpoint used both as the error-detection
backbone (after Token Classification fine-tuning, see train_detector.py) and as
the contextual candidate ranker in the inference pipeline.
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "train_corpus.txt")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "..", "models", "bert-sardinian-uncased")
DEFAULT_MODEL_CHECKPOINT = "dbmdz/bert-base-italian-uncased"
RECORDS_CSV = os.path.join(SCRIPT_DIR, "..", "..", "runs", "records", "pretraining_records.csv")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "runs", "plots", "pretrain")


def parse_args():
    parser = argparse.ArgumentParser(description="Continued MLM pre-training on the Campidanese corpus.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-checkpoint", default=DEFAULT_MODEL_CHECKPOINT,
                         help="Base pretrained checkpoint to continue pre-training from.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    return parser.parse_args()


def prepare_datasets(dataset_path, tokenizer, max_seq_length):
    raw_datasets = load_dataset("text", data_files=dataset_path, split="train")
    split_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    def group_texts(examples):
        # Concatenate all texts and split them into fixed-size blocks so the
        # trailing "short" chunk doesn't waste a padded batch.
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        return {
            key: [tokens[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for key, tokens in concatenated.items()
        }

    tokenized_datasets = split_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets.map(group_texts, batched=True, num_proc=4)


def main():
    args = parse_args()
    print(f"Model will be saved to: {args.output_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    print(f"Loaded tokenizer vocabulary: {len(tokenizer)} tokens")

    lm_datasets = prepare_datasets(args.dataset_path, tokenizer, args.max_seq_length)
    print(f"Train samples: {len(lm_datasets['train'])}")
    print(f"Validation samples: {len(lm_datasets['test'])}")

    model = AutoModelForMaskedLM.from_pretrained(args.model_checkpoint)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
    )

    # Warmup is needed because the pre-trained backbone benefits from a gentle
    # ramp-up when adapting to a new domain.
    total_train_steps = (len(lm_datasets["train"]) // (args.batch_size * args.grad_accumulation)) * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    print(f"Estimated total steps: {total_train_steps}")
    print(f"Warmup steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        bf16=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Starting pre-training:")
    trainer.train()

    # --- Metrics extraction, plot, and record-keeping ---
    history = trainer.state.log_history
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    epochs_x = [x["epoch"] for x in history if "loss" in x]
    eval_epochs_x = [x["epoch"] for x in history if "eval_loss" in x]

    actual_epochs = len(eval_loss)
    best_val_loss = min(eval_loss)
    best_epoch = eval_loss.index(best_val_loss) + 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_checkpoint.split("/")[-1]
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, f"pretrain_loss_{timestamp}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_loss, label="Training Loss")
    plt.plot(eval_epochs_x, eval_loss, label="Validation Loss", marker="o")
    plt.title(f"{model_name} - {timestamp}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
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
        "MLM Probability": [args.mlm_probability],
        "Final Train Loss": [train_loss[-1]],
        "Best Val Loss": [best_val_loss],
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
