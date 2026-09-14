"""
Fine-tunes one LLM baseline (LLaMA or Gemma, 4-bit + LoRA via unsloth) on the
Campidanese spell-correction task, through the vendored LMSpell toolkit
(../../LMSpell). Used to produce the generative baselines compared against the
hybrid BERT pipeline.

Requires a GPU. One call trains one checkpoint — see MODEL_REGISTRY below for the
--model shortcuts matching the checkpoints reported in the paper.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "LMSpell")))

import unsloth  # noqa: E402
from LMSpell import LMSpell  # noqa: E402
from ModelEnum import ModelEnum  # noqa: E402
from Models.Llama.LlamaAbstractModel import LlamaAbstractModel  # noqa: E402
from Models.Gemma.GemmaAbstractModel import GemmaAbstractModel  # noqa: E402
from Trainer.UnslothTrainer import UnslothTrainer  # noqa: E402

DEFAULT_TRAIN_CSV = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "trainLMSpell.csv")
DEFAULT_VAL_CSV = os.path.join(SCRIPT_DIR, "..", "..", "data", "training", "valLMSpell.csv")
DEFAULT_MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "models", "lmspell_models")

# --model shortcuts -> (unsloth checkpoint, output directory name), matching the
# ten checkpoints evaluated in the paper.
MODEL_REGISTRY = {
    "llama-3.2-1b":          ("unsloth/Llama-3.2-1B-bnb-4bit",              "sardinian_llama_3.2_1B"),
    "llama-3.2-1b-instruct": ("unsloth/Llama-3.2-1B-Instruct-bnb-4bit",     "sardinian_llama_3.2_1B_Instruct"),
    "llama-3.2-3b":          ("unsloth/Llama-3.2-3B-bnb-4bit",              "sardinian_llama_3.2_3B"),
    "llama-3.2-3b-instruct": ("unsloth/Llama-3.2-3B-Instruct-bnb-4bit",     "sardinian_llama_3.2_3B_Instruct"),
    "llama-3.1-8b":          ("unsloth/Meta-Llama-3.1-8B-bnb-4bit",         "sardinian_llama_3.1_8B"),
    "llama-3.1-8b-instruct": ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "sardinian_llama_3.1_8B_Instruct"),
    "gemma-2-2b":            ("unsloth/gemma-2-2b-bnb-4bit",                "sardinian_gemma_2_2B"),
    "gemma-2-2b-instruct":   ("unsloth/gemma-2-2b-it-bnb-4bit",             "sardinian_gemma_2_2B_Instruct"),
    "gemma-2-9b":            ("unsloth/gemma-2-9b-bnb-4bit",                "sardinian_gemma_2_9B"),
    "gemma-2-9b-instruct":   ("unsloth/gemma-2-9b-it-bnb-4bit",             "sardinian_gemma_2_9B_Instruct"),
}

SARDINIAN_LANGUAGE = "Campidanese Sardinian"
PROMPT_TEMPLATE = (
    "You are an expert {lang} spell corrector. Below is a sentence in {lang} "
    "language. It may or may not have a spelling mistake. Give the corrected "
    "output in {lang}.\n\n### Text:\n{{}}\n\n### Output:\n{{}}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tunes an LLM baseline for Campidanese spell correction via LMSpell.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()),
                         help="Shortcut for one of the checkpoints reported in the paper. "
                              "Overridden by --model-checkpoint/--output-path if those are set.")
    parser.add_argument("--model-checkpoint", default=None, help="Explicit unsloth checkpoint, e.g. unsloth/gemma-2-9b-bnb-4bit.")
    parser.add_argument("--output-path", default=None, help="Explicit output directory for the fine-tuned model.")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", default=DEFAULT_VAL_CSV)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=64)
    args = parser.parse_args()

    if not args.model and not (args.model_checkpoint and args.output_path):
        parser.error("either --model <shortcut> or both --model-checkpoint and --output-path are required")

    if args.model:
        checkpoint, out_name = MODEL_REGISTRY[args.model]
        args.model_checkpoint = args.model_checkpoint or checkpoint
        args.output_path = args.output_path or os.path.join(DEFAULT_MODELS_DIR, out_name)

    return args


def apply_unsloth_patches(model_checkpoint, lora_rank):
    """
    unsloth's LoRA config and LMSpell's Llama/Gemma wrappers need small runtime
    patches to line up (LoRA rank isn't otherwise configurable through LMSpell,
    and the abstract model classes don't always resolve model_label on their own
    when bypassing unsloth's own loader).
    """
    if not getattr(unsloth.FastLanguageModel, "_is_peft_patched", False):
        original_peft = unsloth.FastLanguageModel.get_peft_model

        def patched_peft(*p_args, **p_kwargs):
            p_kwargs["r"] = lora_rank
            p_kwargs["lora_alpha"] = lora_rank * 2
            return original_peft(*p_args, **p_kwargs)

        unsloth.FastLanguageModel.get_peft_model = patched_peft
        unsloth.FastLanguageModel._is_peft_patched = True

    def patch_model_label(model_class):
        if getattr(model_class, "_is_patched", False):
            return
        original_init = model_class.__init__

        def patched_init(self):
            for model_enum in ModelEnum:
                if model_enum.value == model_checkpoint:
                    self.model_label = model_enum
                    break
            original_init(self)
            self.do_sample = False

        model_class.__init__ = patched_init
        model_class._is_patched = True

    if "llama" in model_checkpoint.lower():
        patch_model_label(LlamaAbstractModel)
    if "gemma" in model_checkpoint.lower():
        patch_model_label(GemmaAbstractModel)


def main():
    args = parse_args()

    # If your CUDA toolkit is in a non-standard location, point
    # LIMBACHECK_CUDA_LIB_PATH at it (e.g. .venv/lib/python*/site-packages/nvidia/*/lib)
    # so bitsandbytes can find it. Not needed on most setups.
    cuda_lib_path = os.environ.get("LIMBACHECK_CUDA_LIB_PATH")
    if cuda_lib_path:
        os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    print("=========================================")
    print(" Starting LMSpell Training")
    print(f" Model:  {args.model_checkpoint}")
    print(f" Output: {args.output_path}")
    print("=========================================")

    apply_unsloth_patches(args.model_checkpoint, args.lora_rank)

    spell_model = LMSpell(args.model_checkpoint)
    spell_model.model.do_sample = False  # Force greedy decoding
    spell_model.language = SARDINIAN_LANGUAGE
    spell_model.PROMPT = PROMPT_TEMPLATE.format(lang=SARDINIAN_LANGUAGE)

    trainer = UnslothTrainer(
        model_instance=spell_model,
        train_path=args.train_csv,
        val_path=args.val_csv,
        exp_name=args.output_path,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
    )

    print(f"\n[TRAINING] Starting fine-tuning ({args.epochs} epochs, LR {args.learning_rate}, LoRA rank {args.lora_rank})...")
    trainer.train()

    print(f"\n[DONE] Model successfully saved to: {args.output_path}")


if __name__ == "__main__":
    main()
