<p align="center">
  <img src="https://img.shields.io/badge/Language-Campidanese%20Sardinian-blue?style=for-the-badge" alt="Language">
  <img src="https://img.shields.io/badge/Model-BERT%20%2B%20SymSpell-orange?style=for-the-badge" alt="Model">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge&logo=python" alt="Python">
</p>

<h1 align="center">Limbacheck</h1>
<p align="center"><b>A Context-Aware Neural Spell Checker for Campidanese Sardinian in Low-Resource Settings</b></p>

<p align="center">
Salvatore Carta · Filippo Concas · Alessandro Giuliani · Marco Manolo Manca · Mirko Marras · Alessandro Sebastian Podda<br>
University of Cagliari — GenAI4SG 2026 (Generative AI for Social Good and Vulnerable Populations)
</p>

---

## Overview

Sardinian is the largest minority language in Italy and is classified by UNESCO as *definitely endangered*.
Speakers writing in it currently have no context-aware support: the tool distributed by the regional
administration, [CROS](https://cros.nor-web.eu/), is dictionary-based and therefore blind to real-word errors —
orthographically valid forms used in the wrong context.

**Limbacheck** is a prototype context-aware spell checker for **Campidanese Sardinian**, built around the
constraints that actually govern deployment for a minority language: no parallel error-correction corpora,
no dedicated accelerators, and a latency budget tight enough for interactive use. It combines:

- a **BERT encoder**, adapted to Campidanese through continued pre-training, that flags anomalous tokens
  (detection) and scores candidates in context (ranking);
- **SymSpell** to generate correction candidates algorithmically from a validated lexicon — the neural
  model never generates text, so no form outside the lexicon can appear in the output;
- a **synthetic error generator** whose confusion sets are derived from the *Official Graphemic Repertory of
  the Sardinian Language*, used to synthesize training pairs in the absence of parallel data.

On a held-out test set, the system reaches an **F₀.₅ of 0.7477** — the highest of a twelve-system comparison
that includes ten fine-tuned generative baselines (LLaMA 3.1/3.2, Gemma 2, 1B–9B, Base and Instruct) and the
institutional tool — while running at **20.88 ms/sentence**, roughly two orders of magnitude faster than the
generative baselines.

Full method and results: see the paper (citation below).

---

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │           STAGE 1 — Encoder adaptation          │
                        │  Campidanese corpus → continued pre-training →  │
                        │        Sardinian BERT (MLM, shared encoder)     │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │       STAGE 2 — Synthetic error generation      │
                        │  Official Graphemic Repertory → confusion sets  │
                        │      (39 patterns, 69 substitutions) → noise    │
                        └──────────────────────┬──────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │         STAGE 3 — Detector fine-tuning          │
                        │  Dynamically corrupted corpus → token           │
                        │  classification head on the shared encoder      │
                        └──────────────────────┬──────────────────────────┘
                                               │
              ┌────────────────────────────────▼────────────────────────────────────┐
              │                    STAGE 4 — Inference                              │
              │                                                                     │
              │  Input text                                                        │
              │      │                                                             │
              │      ├── [Detector] token classification, p_i > τ_d ──► anomalies  │
              │      │                                             │               │
              │      └── [SymSpell] candidates within d_min+1 ◄────┘               │
              │                           │                                        │
              │        [MLM ranker] batched contextual score ℓ(·) ──► corrected    │
              │        substitute iff ℓ(c*) − ℓ(w_i) > τ_c                         │
              └─────────────────────────────────────────────────────────────────────┘
                                               │
                        ┌──────────────────────▼──────────────────────────┐
                        │              STAGE 5 — Evaluation               │
                        │  F0.5 · WER · CER · latency · non-harmfulness   │
                        │  (WER_out < WER_in) on a held-out test set      │
                        └─────────────────────────────────────────────────┘
```

Candidates are ranked lexicographically on `(damerau_levenshtein_distance, -mlm_score)`: orthographic
proximity has absolute priority, and the contextual MLM score breaks ties. All candidate variants for a
flagged word are batched into a single forward pass, so latency is virtually independent of how many
candidates are generated.

Detection/correction thresholds (τ_d, τ_c) and the maximum edit distance (k) are tuned by exhaustive grid
search **on a validation split, never on the test set** — see [`src/evaluation/grid_search.py`](src/evaluation/grid_search.py).

---

## Results

Evaluated on a 160-sentence held-out test set (see the paper, Section 4–5, for the full protocol):

| System | F₀.₅ | Precision | Recall | WER | CER | Latency (ms) |
|---|---|---|---|---|---|---|
| *Uncorrected input* | – | – | – | 0.1174 | 0.0226 | – |
| **Hybrid Pipeline (ours)** | **0.7477** | **0.8027** | 0.5868 | 0.0502 | **0.0108** | **20.88** |
| Gemma 2 9B | 0.7348 | 0.7310 | 0.7506 | **0.0442** | 0.0122 | 2145.15 |
| LLaMA 3.1 8B | 0.7132 | 0.7168 | 0.6993 | 0.0499 | 0.0154 | 1384.52 |
| LLaMA 3.1 8B-Instruct | 0.7084 | 0.7180 | 0.6724 | 0.0505 | 0.0143 | 1439.31 |
| Gemma 2 9B-Instruct | 0.6965 | 0.6929 | 0.7115 | 0.0508 | 0.0148 | 2197.96 |
| LLaMA 3.2 3B | 0.6337 | 0.6423 | 0.6015 | 0.0634 | 0.0176 | 1180.77 |
| Gemma 2 2B | 0.6157 | 0.6234 | 0.5868 | 0.0640 | 0.0210 | 1285.83 |
| LLaMA 3.2 3B-Instruct | 0.6071 | 0.6117 | 0.5892 | 0.0663 | 0.0222 | 1284.94 |
| LLaMA 3.2 1B-Instruct | 0.5197 | 0.5259 | 0.4963 | 0.0746 | 0.0233 | 716.68 |
| LLaMA 3.2 1B | 0.5144 | 0.5192 | 0.4963 | 0.0861 | 0.0346 | 729.89 |
| Gemma 2 2B-Instruct ⚠ | 0.1909 | 0.1636 | 0.5721 | 0.3343 | 0.2725 | 1700.70 |
| CROS (VAR) ⚠ | 0.1432 | 0.1321 | 0.2152 | 0.2149 | 0.0584 | 1229.74¹ |

⚠ Fails the **non-harmfulness** criterion (WER<sub>out</sub> ≥ WER<sub>in</sub> = 0.1174): these two systems
leave the text more corrupted, on average, than they received it.
¹ CROS latency includes a network round-trip and is not comparable to the local GPU latency of the other systems.

Optimal inference hyperparameters (found by grid search on the validation split): τ_d = 0.75, τ_c = −1.5,
k = 1, m = 20 candidates — see `runs/pipeline_thresholds.json`.

---

## Repository structure

```
limbacheck/
│
├── data/
│   ├── datasets/            # Raw bilingual corpus + cleaned Sardinian-only text
│   ├── dictionaries/        # Validated lexicon (corpus-derived + Campidanese standard/non-standard)
│   └── training/            # train/val/test splits (corpus + LMSpell CSVs), regenerated by data-prep stages
│
├── models/                  # Trained checkpoints (gitignored — see models/README.md to regenerate)
│   ├── bert-sardinian-uncased/       # continued-pretrained MLM (Stage 1)
│   ├── bert-sardinian-spellchecker/  # fine-tuned detector (Stage 3)
│   └── lmspell_models/               # 10 fine-tuned LLM baselines (comparison)
│
├── src/
│   ├── data_prep/            # Corpus cleaning, train/val/test split, synthetic error generator
│   ├── modeling/             # Continued MLM pre-training, detector fine-tuning
│   ├── pipeline/pipeline.py  # Core: SardinianSpellchecker (detector + SymSpell + MLM ranker)
│   ├── evaluation/           # grid_search.py, evaluate_pipeline.py, evaluate_cros.py, report.py
│   └── llm_baselines/        # Data prep, fine-tuning and evaluation for the generative baselines
│
├── LMSpell/                  # Vendored toolkit used to fine-tune/evaluate the LLM baselines
├── runs/                     # Result records (CSV), plots, pipeline_thresholds.json, comparison tables
├── docs/                     # Official Graphemic Repertory of the Sardinian Language
├── paper_plot.py             # Reproduces the paper's figures from runs/comparison.csv
├── run_experiments.py        # Single entrypoint to (re)run any/all stages — see below
└── requirements.txt
```

---

## Installation

> **Requirements:** Python 3.10+, CUDA 12.x, a GPU with at least 8GB VRAM (16GB+ recommended for the 8–9B
> LLM baselines).

```bash
git clone https://github.com/<org>/limbacheck.git
cd limbacheck

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> `torch`, `xformers`, and `bitsandbytes` depend on the installed CUDA version. If installation fails, check:
> - https://pytorch.org/get-started/locally/
> - https://github.com/TimDettmers/bitsandbytes
>
> If your driver caps out below the CUDA version PyPI ships by default, install a matching build explicitly,
> e.g. `pip install torch==<version> --index-url https://download.pytorch.org/whl/cu126`.

---

## Reproducing the experiments

`run_experiments.py` is a single entrypoint over every stage of the pipeline, in dependency order:

```bash
python run_experiments.py --list-stages     # see every stage and its flags
python run_experiments.py --stage all       # runs the full pipeline end to end
```

Stage order (`--stage all`): `clean-corpus → split-corpus → build-vocabulary → prepare-llm-data →
pretrain-mlm → train-detector → grid-search → evaluate-pipeline → train-llm-all → evaluate-llm-all →
evaluate-cros → report`.

A few stages worth calling out:

- **`split-corpus`** produces `train_corpus.txt` / `val_corpus.txt` / `test_corpus.txt`. The test split is
  carved off first and is never touched again; `val_corpus.txt` is used exclusively for early stopping
  (`train-detector`) and threshold tuning (`grid-search`), so the numbers in the Results table above are a
  genuinely held-out measurement.
- **`grid-search`** (`src/evaluation/grid_search.py`) searches τ_d × τ_c × k on the validation split and
  writes the winner to `runs/pipeline_thresholds.json`, which `evaluate-pipeline` then reads automatically.
- **`train-llm-all`** / **`evaluate-llm-all`** each run one model per subprocess (not in-process), which
  matters for correctness: LMSpell's model-wrapper classes are monkey-patched per checkpoint, and reusing
  one interpreter across checkpoints would silently reuse the first checkpoint's base model for the rest.

Each stage can also be run standalone, e.g. `python -m src.evaluation.grid_search --help` for its own
argument list.

---

## Citation

```bibtex
@inproceedings{carta2026limbacheck,
  author    = {Carta, Salvatore and Concas, Filippo and Giuliani, Alessandro and Manca, Marco Manolo and Marras, Mirko and Podda, Alessandro Sebastian},
  title     = {A Context-Aware Neural Spell Checker for Campidanese Sardinian in Low-Resource Settings},
  booktitle = {Proceedings of Generative AI for Social Good and Vulnerable Populations (GenAI4SG 2026)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  pages     = {8},
  doi       = {XXXXXXX.XXXXXXX}
}
```

<!-- TODO: replace the DOI/ISBN placeholders above once the camera-ready is assigned one. -->

---

## License

No license has been chosen for this repository yet. Until one is added, all rights are reserved by the
authors; reach out before reusing the code.

---

## Acknowledgments

Developed at the [University of Cagliari](https://www.unica.it/). The Official Graphemic Repertory of the
Sardinian Language and CROS are maintained by the Autonomous Region of Sardinia.
