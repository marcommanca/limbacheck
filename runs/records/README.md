# Result records

Each CSV here holds only the **canonical** result row(s) — the ones actually
reported in the paper — for one experiment type:

| File | What it records | Written by |
|---|---|---|
| `pretraining_records.csv` | MLM continued-pretraining run (Step 1) | `src/modeling/pretrain_mlm.py` |
| `finetune_records.csv` | Detector fine-tuning run (Step 3), early-stopped on `val_corpus.txt` | `src/modeling/train_detector.py` |
| `grid_search_records.csv` | Best pipeline thresholds found on `val_corpus.txt` (Step 4) — the audit trail for the thresholds evaluate_pipeline.py uses | `src/evaluation/grid_search.py` |
| `testing_records.csv` | Hybrid BERT pipeline evaluation on the held-out `test_corpus.txt` (Step 5) | `src/evaluation/evaluate_pipeline.py` |
| `lmspell_testing_records.csv` | LLM baseline evaluations (one row per checkpoint) | `src/llm_baselines/evaluate_llm.py` |
| `cros_testing_records.csv` | CROS comparison | `src/evaluation/evaluate_cros.py` |

Running any of those scripts **appends** a new row here. Full experimental
history (discarded base-model trials, earlier training runs) is intentionally
not kept here — see `old/results_history/` (gitignored, local only) for that.

Thresholds are tuned exclusively on `val_corpus.txt` (never on `test_corpus.txt`)
— see `data/training/` and `src/data_prep/split_corpus.py` for the train/val/test
split, load-bearing for every number above.

`src/evaluation/report.py` (the `report` stage of `run_experiments.py`) reads
these files and produces `runs/comparison.{csv,md,tex}`.
