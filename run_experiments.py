#!/usr/bin/env python
"""
Single entrypoint to (re)run every experiment in the Limbacheck project, stage by
stage, in the right order. Each stage is a thin wrapper around the corresponding
module under src/ — see that module's docstring for details and additional flags
(run e.g. `python -m src.evaluation.evaluate_pipeline --help` directly for the
full argument list of a given stage).

Stages requiring a GPU and/or pre-trained checkpoints are marked as such below;
they cannot run without the model weights referenced in models/README.md.

Usage:
    python run_experiments.py --stage <stage-name> [extra args passed through]
    python run_experiments.py --stage all              # runs every stage in order
    python run_experiments.py --list-stages
"""

import os
import runpy
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STAGES = {
    # --- Data preparation (CPU only) ---
    "clean-corpus": {
        "module": "src.data_prep.clean_corpus",
        "help": "Cleans the raw bilingual corpus into campidanese_cleaned.txt.",
        "gpu": False,
    },
    "split-corpus": {
        "module": "src.data_prep.split_corpus",
        "help": "Splits the cleaned corpus into train_corpus.txt / val_corpus.txt / test_corpus.txt (seed=42). "
                "Load-bearing: every evaluation stage depends on this exact split. val_corpus.txt is used only for "
                "grid-search and for train-detector's early stopping; test_corpus.txt is reserved for final metrics.",
        "gpu": False,
    },
    "scrape-dictionaries": {
        "module": "src.data_prep.scrape_dictionaries",
        "help": "OPTIONAL/manual: re-scrapes the online Sardinian dictionaries (slow, hits external sites).",
        "gpu": False,
        "default": False,
    },
    "build-vocabulary": {
        "module": "src.data_prep.build_vocabulary",
        "help": "Extracts the corpus vocabulary and reports dictionary coverage metrics.",
        "gpu": False,
    },
    "prepare-llm-data": {
        "module": "src.llm_baselines.prepare_llm_data",
        "help": "Builds the train/val/test CSVs used to fine-tune the LLM baselines.",
        "gpu": False,
    },

    # --- Model training (GPU required) ---
    "pretrain-mlm": {
        "module": "src.modeling.pretrain_mlm",
        "help": "Step 1: continued MLM pre-training on the Campidanese corpus.",
        "gpu": True,
    },
    "train-detector": {
        "module": "src.modeling.train_detector",
        "help": "Step 3: fine-tunes the MLM as a Token Classification error detector.",
        "gpu": True,
    },
    "train-tokenizer": {
        "module": "src.modeling.train_tokenizer",
        "help": "OPTIONAL ablation: trains a custom Unigram tokenizer (not used by the final pipeline).",
        "gpu": False,
        "default": False,
    },
    "train-llm": {
        "module": "src.llm_baselines.train_llm",
        "help": "Fine-tunes one LLM baseline via LMSpell. Pass --model <shortcut>, e.g. --model gemma-2-9b.",
        "gpu": True,
    },
    "train-llm-all": {
        "meta": True,
        "help": "Trains all 10 LLM baselines from src/llm_baselines/train_llm.py's MODEL_REGISTRY, one "
                "after another. Skips checkpoints that already exist unless --force is passed.",
        "gpu": True,
    },

    # --- Evaluation (GPU required unless noted) ---
    "grid-search": {
        "module": "src.evaluation.grid_search",
        "help": "Tunes the pipeline's detection/correction thresholds and max edit distance on val_corpus.txt "
                "(never on test_corpus.txt) and writes the winner to runs/pipeline_thresholds.json, which "
                "evaluate-pipeline then picks up automatically. Requires pretrain-mlm + train-detector to be done. "
                "Can be run standalone at any time to inspect the current optimum before running the rest.",
        "gpu": True,
    },
    "evaluate-pipeline": {
        "module": "src.evaluation.evaluate_pipeline",
        "help": "Step 5: evaluates the hybrid BERT pipeline on the held-out test set (F0.5/WER/CER/latency), "
                "using the thresholds from runs/pipeline_thresholds.json (see grid-search).",
        "gpu": True,
    },
    "evaluate-llm": {
        "module": "src.llm_baselines.evaluate_llm",
        "help": "Evaluates one fine-tuned LLM baseline. Pass --model-path <path>.",
        "gpu": True,
    },
    "evaluate-llm-all": {
        "meta": True,
        "help": "Evaluates every LLM baseline checkpoint found under models/lmspell_models/, skipping "
                "any MODEL_REGISTRY entry that hasn't been trained yet.",
        "gpu": True,
    },
    "evaluate-cros": {
        "module": "src.evaluation.evaluate_cros",
        "help": "Evaluates the third-party CROS spellchecker over its public web API (network only, no GPU).",
        "gpu": False,
    },

    # --- Reporting (CPU only) ---
    "report": {
        "module": "src.evaluation.report",
        "help": "Builds the final comparison table (CSV + Markdown + LaTeX) from runs/records/.",
        "gpu": False,
    },
}

# Order used by --stage all. Manual/optional stages (scrape-dictionaries,
# train-tokenizer) are intentionally excluded from the default full run.
ALL_ORDER = [
    "clean-corpus", "split-corpus", "build-vocabulary", "prepare-llm-data",
    "pretrain-mlm", "train-detector", "grid-search", "evaluate-pipeline",
    "train-llm-all", "evaluate-llm-all",
    "evaluate-cros",
    "report",
]


def print_stage_list():
    print("Available stages:\n")
    for name, spec in STAGES.items():
        gpu_tag = " [GPU]" if spec["gpu"] else ""
        manual_tag = " [manual/optional]" if spec.get("default") is False else ""
        print(f"  {name:<20}{gpu_tag}{manual_tag}\n      {spec['help']}\n")
    print("  all                  runs every non-optional stage in order:")
    print("      " + " -> ".join(ALL_ORDER))


def run_stage(stage_name, extra_args):
    spec = STAGES[stage_name]
    if spec.get("meta"):
        if stage_name == "train-llm-all":
            run_train_llm_all(extra_args)
        elif stage_name == "evaluate-llm-all":
            run_evaluate_llm_all(extra_args)
        return

    sys.argv = [spec["module"]] + extra_args
    runpy.run_module(spec["module"], run_name="__main__")


def run_stage_subprocess(stage_name, extra_args):
    """Runs a stage in a fresh Python subprocess instead of in-process.

    Required for train-llm/evaluate-llm when looped over multiple models
    (train-llm-all / evaluate-llm-all): LMSpell's Llama/Gemma wrapper classes
    are monkey-patched at runtime by train_llm.apply_unsloth_patches(), and
    that patch is applied once per class per process (guarded by a
    `_is_patched` flag that's never reset). Looping in-process via
    runpy.run_module (as run_stage does) makes every model after the first in
    a shared wrapper class (e.g. all non-"-8b" Llama checkpoints, or any
    Gemma checkpoint) silently reuse the first model's base checkpoint. A
    subprocess per model gives each one a clean, unpatched interpreter.
    """
    spec = STAGES[stage_name]
    cmd = [sys.executable, "-m", spec["module"]] + extra_args
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[run_experiments] Stage '{stage_name}' {extra_args} failed with exit code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def run_train_llm_all(extra_args):
    """Trains every checkpoint in train_llm.MODEL_REGISTRY, skipping ones that
    already exist unless --force is passed (imported lazily: this pulls in
    unsloth/LMSpell, which we don't want to load for unrelated stages)."""
    from src.llm_baselines.train_llm import MODEL_REGISTRY, DEFAULT_MODELS_DIR

    force = "--force" in extra_args
    passthrough_args = [a for a in extra_args if a != "--force"]

    for key, (_, out_name) in MODEL_REGISTRY.items():
        output_path = os.path.join(DEFAULT_MODELS_DIR, out_name)
        if not force and os.path.isdir(output_path) and os.listdir(output_path):
            print(f"[run_experiments] Skipping train-llm --model {key}: {output_path} already exists (pass --force to retrain).")
            continue
        print(f"\n{'=' * 60}\n[run_experiments] train-llm --model {key}\n{'=' * 60}")
        run_stage_subprocess("train-llm", ["--model", key] + passthrough_args)


def run_evaluate_llm_all(extra_args):
    """Evaluates every checkpoint in train_llm.MODEL_REGISTRY that has actually
    been trained, skipping the rest with a warning."""
    from src.llm_baselines.train_llm import MODEL_REGISTRY, DEFAULT_MODELS_DIR

    for key, (_, out_name) in MODEL_REGISTRY.items():
        model_path = os.path.join(DEFAULT_MODELS_DIR, out_name)
        if not os.path.isdir(model_path):
            print(f"[run_experiments] Skipping evaluate-llm for {key}: {model_path} not found (train it first with --stage train-llm --model {key}).")
            continue
        print(f"\n{'=' * 60}\n[run_experiments] evaluate-llm --model-path {model_path}\n{'=' * 60}")
        run_stage_subprocess("evaluate-llm", ["--model-path", model_path] + extra_args)


def main():
    # --stage is parsed manually (rather than via argparse) so that a trailing
    # --help/-h is NOT swallowed by this outer CLI and instead reaches the
    # target stage's own argparse, printing that stage's actual options.
    argv = sys.argv[1:]

    if not argv or "--list-stages" in argv:
        print(__doc__)
        print_stage_list()
        return

    if "--stage" not in argv:
        print("Error: --stage <name> is required (or use --list-stages).\n", file=sys.stderr)
        print_stage_list()
        sys.exit(2)

    idx = argv.index("--stage")
    if idx + 1 >= len(argv):
        print("Error: --stage requires a value. Use --list-stages to see available stages.", file=sys.stderr)
        sys.exit(2)

    stage_name = argv[idx + 1]
    extra_args = argv[:idx] + argv[idx + 2:]

    if stage_name != "all" and stage_name not in STAGES:
        print(f"Error: unknown stage '{stage_name}'. Use --list-stages to see available stages.", file=sys.stderr)
        sys.exit(2)

    if stage_name == "all":
        if extra_args:
            print("Note: extra arguments are ignored when running --stage all (each stage uses its own defaults).")
        for name in ALL_ORDER:
            print(f"\n{'=' * 60}\n[run_experiments] Stage: {name}\n{'=' * 60}")
            run_stage(name, [])
        return

    run_stage(stage_name, extra_args)


if __name__ == "__main__":
    main()
