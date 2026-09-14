# Model checkpoints

This directory is not versioned (see `.gitignore`) — it holds trained model
weights, which are too large for the repository. Running the training stages of
`run_experiments.py` populates it with the following expected layout:

```
models/
├── bert-sardinian-uncased/          # output of `pretrain-mlm` (Step 1)
├── bert-sardinian-spellchecker/     # output of `train-detector` (Step 3),
│                                     # built on top of bert-sardinian-uncased/
├── tokenizers/
│   └── unigram-tokenizer-12k-campidanese-uncased.json  # output of the optional
│                                                         # `train-tokenizer` ablation
│                                                         # (NOT used by the final
│                                                         # pipeline — underperformed)
└── lmspell_models/
    ├── sardinian_llama_3.2_1B/
    ├── sardinian_llama_3.2_1B_Instruct/
    ├── sardinian_llama_3.2_3B/
    ├── sardinian_llama_3.2_3B_Instruct/
    ├── sardinian_llama_3.1_8B/
    ├── sardinian_llama_3.1_8B_Instruct/
    ├── sardinian_gemma_2_2B/
    ├── sardinian_gemma_2_2B_Instruct/
    ├── sardinian_gemma_2_9B/
    └── sardinian_gemma_2_9B_Instruct/      # outputs of `train-llm --model <shortcut>`
```

`src/evaluation/evaluate_pipeline.py` and `src/evaluation/grid_search.py` expect
the first two directories at those exact names by default (override with
`--mlm-checkpoint` / `--detector-checkpoint` if you used different names).
