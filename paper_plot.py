"""
Generates the two comparison figures for the paper from runs/comparison.csv,
sized for a single LaTeX column (figsize=(3.4, 2.1), tight-cropped):

  1. wer_by_system      — horizontal bar chart of WER per system, with a
                           vertical reference line at the "do nothing" baseline
                           WER (corrupted text vs ground truth) and the region
                           worse than that baseline shaded.
  2. latency_vs_f05     — scatter of inference latency (log scale) vs F0.5,
                           excluding CROS (its latency includes a network
                           round-trip and isn't comparable to local GPU latency).

Systems are grouped into three families (Hybrid Pipeline / LLaMA / Gemma,
plus CROS for the WER plot) and colored with the first slots of the
project's validated categorical palette (see the dataviz skill's
references/palette.md) — order is fixed, never cycled, and color always
follows the family, never the rank.

No in-figure title: at column width the title ate too much of the 2.1in
height for what the LaTeX \\caption{} already says — add the caption there.

Usage:
    python paper_plot.py [--comparison-csv path/to/comparison.csv] [--output-dir paper_plots]
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COMPARISON_CSV = os.path.join(SCRIPT_DIR, "runs", "comparison.csv")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "paper_plots")

BASELINE_WER = 0.1174  # WER of the uncorrected/corrupted text vs ground truth (see evaluation/common.py)

FIGSIZE = (3.4, 2.1)
SAVE_KWARGS = dict(dpi=400, bbox_inches="tight", pad_inches=0.02)

# Validated categorical palette (see dataviz skill references/palette.md) —
# fixed order, never cycled. First 3 slots are validated for all-pairs
# comparisons (the scatter); first 4 for adjacent comparisons (the bar chart).
COLOR_PIPELINE = "#2a78d6"  # slot 1, blue
COLOR_LLAMA = "#eb6834"     # slot 2, orange
COLOR_GEMMA = "#1baf7a"     # slot 3, aqua
COLOR_CROS = "#eda100"      # slot 4, yellow

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d3"

plt.rcParams.update({
    "font.size": 6.5,
    "font.family": "serif",
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "legend.fontsize": 5.5,
    "axes.grid": False,
    "svg.fonttype": "none",
})


def classify(system):
    if system.startswith("Hybrid Pipeline"):
        return "Hybrid Pipeline (ours)", COLOR_PIPELINE
    if system.startswith("CROS"):
        return "CROS (external)", COLOR_CROS
    if "llama" in system.lower():
        return "LLaMA baseline", COLOR_LLAMA
    if "gemma" in system.lower():
        return "Gemma baseline", COLOR_GEMMA
    return "Other", TEXT_SECONDARY


def short_label(system):
    """Shortens raw system names for compact axis/point labels."""
    if system.startswith("Hybrid Pipeline"):
        return "BERT (ours)"
    if system.startswith("CROS"):
        return "CROS"
    name = system.replace("sardinian_", "")
    name = name.replace("llama_3.2_", "LLaMA ").replace("llama_3.1_", "LLaMA ")
    name = name.replace("gemma_2_", "Gemma ")
    name = name.replace("_Instruct", "-Inst").replace("_", " ")
    return name


def plot_wer_by_system(df, output_dir):
    df = df.copy()
    df["family"], df["color"] = zip(*df["System"].map(classify))
    df["label"] = df["System"].map(short_label)
    df = df.sort_values("WER", ascending=False).reset_index(drop=True)  # worst at top->bottom becomes best-at-top after barh's bottom-up draw order

    fig, ax = plt.subplots(figsize=FIGSIZE)

    y_pos = range(len(df))
    bars = ax.barh(y_pos, df["WER"], color=df["color"], height=0.68, zorder=3)

    x_max = max(df["WER"].max(), BASELINE_WER) * 1.22
    ax.axvspan(BASELINE_WER, x_max, color="#e34948", alpha=0.07, zorder=1)
    ax.axvline(BASELINE_WER, color="#e34948", linewidth=1.0, linestyle="--", zorder=2)
    ax.text(BASELINE_WER + x_max * 0.012, len(df) - 0.4, f"baseline={BASELINE_WER:.3f}",
            color="#e34948", fontsize=5, va="top", ha="left")

    for bar, wer in zip(bars, df["WER"]):
        ax.text(bar.get_width() + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{wer:.3f}", va="center", ha="left", fontsize=5, color=TEXT_PRIMARY)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Word Error Rate (WER)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.6, len(df) - 0.4)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", pad=1.5)

    fig.savefig(os.path.join(output_dir, "wer_by_system.pdf"), **SAVE_KWARGS)
    fig.savefig(os.path.join(output_dir, "wer_by_system.png"), **SAVE_KWARGS)
    print(f"Saved: {os.path.join(output_dir, 'wer_by_system.pdf')}")
    print(f"Saved: {os.path.join(output_dir, 'wer_by_system.png')}")
    plt.close(fig)


def plot_latency_vs_f05(df, output_dir):
    df = df[~df["System"].str.startswith("CROS")].copy()
    df["family"], df["color"] = zip(*df["System"].map(classify))
    df["label"] = df["System"].map(short_label)

    # Compact mnemonic codes instead of a numbered key box: at column width
    # there's no room for an 11-line legend, and short codes ("L8i") sit
    # close to their marker without a lookup table.
    code_of = {
        "BERT (ours)": "BERT", "LLaMA 1B": "L1", "LLaMA 1B-Inst": "L1i",
        "LLaMA 3B": "L3", "LLaMA 3B-Inst": "L3i", "LLaMA 8B": "L8", "LLaMA 8B-Inst": "L8i",
        "Gemma 2B": "G2", "Gemma 2B-Inst": "G2i", "Gemma 9B": "G9", "Gemma 9B-Inst": "G9i",
    }
    df["code"] = df["label"].map(code_of)

    markers = {"Hybrid Pipeline (ours)": "o", "LLaMA baseline": "s", "Gemma baseline": "^"}

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for family, group in df.groupby("family", sort=False):
        ax.scatter(
            group["Latency (ms)"], group["F0.5 Score"],
            s=20, color=group["color"].iloc[0], marker=markers.get(family, "o"),
            edgecolor="white", linewidth=0.4, zorder=3, label=family,
        )

    # Small per-code nudges (points) for the two tight clusters (3B-scale
    # trio, 8B/9B-scale quartet) so adjacent codes don't touch.
    nudge = {
        "L1": (-3, -3), "L1i": (3, 3),
        "L3": (-4, 3), "L3i": (-4, -4), "G2": (4, 0),
        "L8": (0, 4), "L8i": (0, -4.5),
        "G9": (0, 4), "G9i": (0, -4.5),
    }
    for _, row in df.iterrows():
        dx, dy = nudge.get(row["code"], (3, 2.5))
        ax.annotate(
            row["code"], (row["Latency (ms)"], row["F0.5 Score"]),
            textcoords="offset points", xytext=(dx, dy),
            ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
            va="center", fontsize=4.6, color=row["color"], zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(10, 6000)
    ax.set_ylim(0.13, 0.80)
    ax.set_xlabel("Latency (ms, log scale)")
    ax.set_ylabel("Correction F0.5")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.3, zorder=0, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)

    ax.legend(loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3)

    fig.savefig(os.path.join(output_dir, "latency_vs_f05.pdf"), **SAVE_KWARGS)
    fig.savefig(os.path.join(output_dir, "latency_vs_f05.png"), **SAVE_KWARGS)
    print(f"Saved: {os.path.join(output_dir, 'latency_vs_f05.pdf')}")
    print(f"Saved: {os.path.join(output_dir, 'latency_vs_f05.png')}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generates the paper's WER and latency-vs-F0.5 comparison figures.")
    parser.add_argument("--comparison-csv", default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.comparison_csv)
    print(f"Loaded {len(df)} systems from {args.comparison_csv}")

    plot_wer_by_system(df, args.output_dir)
    plot_latency_vs_f05(df, args.output_dir)


if __name__ == "__main__":
    main()
