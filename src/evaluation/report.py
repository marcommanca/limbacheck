"""
Builds the final comparison table (Pipeline vs LLM baselines vs CROS) from the
canonical result records in runs/records/, in CSV, Markdown, and LaTeX. This is
the "report" stage of run_experiments.py, meant to be re-run any time a record
file changes, without re-running any experiment.

Expects each records CSV to already contain only the canonical/final rows (see
runs/records/README.md) — intermediate/discarded runs live in old/results_history/,
not here. The threshold grid search itself (grid_search.py) has its own record,
runs/records/grid_search_records.csv, kept as the audit trail for how the
thresholds used below were chosen; it isn't part of this comparison table.
"""

import argparse
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "runs", "records")
OUTPUT_BASENAME = os.path.join(SCRIPT_DIR, "..", "..", "runs", "comparison")

PIPELINE_RECORDS_CSV = os.path.join(RECORDS_DIR, "testing_records.csv")
LLM_RECORDS_CSV = os.path.join(RECORDS_DIR, "lmspell_testing_records.csv")
CROS_RECORDS_CSV = os.path.join(RECORDS_DIR, "cros_testing_records.csv")

COLUMNS = ["System", "F0.5 Score", "Precision", "Recall", "WER", "CER", "Latency (ms)"]


def _row(system, r, f05_key="Correction F0.5", prec_key="Correction Precision",
         rec_key="Correction Recall", wer_key="WER", cer_key="CER", lat_key="Avg Latency (ms)"):
    return {
        "System": system,
        "F0.5 Score": float(r[f05_key]),
        "Precision": float(r[prec_key]),
        "Recall": float(r[rec_key]),
        "WER": float(r[wer_key]),
        "CER": float(r[cer_key]),
        "Latency (ms)": float(r[lat_key]),
    }


def build_comparison_table():
    rows = []

    if os.path.exists(PIPELINE_RECORDS_CSV):
        df_pipe = pd.read_csv(PIPELINE_RECORDS_CSV)
        for _, r in df_pipe.iterrows():
            rows.append(_row("Hybrid Pipeline (BERT)", r))

    if os.path.exists(LLM_RECORDS_CSV):
        df_llm = pd.read_csv(LLM_RECORDS_CSV)
        for _, r in df_llm.iterrows():
            rows.append(_row(r["Model"], r))

    if os.path.exists(CROS_RECORDS_CSV):
        df_cros = pd.read_csv(CROS_RECORDS_CSV)
        for _, r in df_cros.iterrows():
            rows.append(_row(r["System"], r))

    if not rows:
        raise SystemExit(
            f"No record files found in {RECORDS_DIR}. Run the evaluate-* stages first, "
            "or check runs/records/ was populated from old/results_history/."
        )

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(by="F0.5 Score", ascending=False).reset_index(drop=True)
    return df


NOTE_TEXT = (
    "All systems were evaluated on the same 160-sentence test split "
    "(error_rate=0.15, seed=42; see src/evaluation/common.py). For CROS, "
    "\"correction\" is our own reconstruction (first suggestion accepted for "
    "each flagged word) since CROS has no native full-sentence correction or "
    "contextual disambiguation, unlike the proposed MLM-ranked pipeline. CROS "
    "latency includes the network round-trip to a remote public server and is "
    "not directly comparable to the local (GPU) latency of the BERT pipeline "
    "and the LLM baselines."
)


def write_markdown(df, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Comparison: Hybrid Pipeline vs LLM Baselines vs CROS\n\n")
        f.write(NOTE_TEXT + "\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")


def write_latex(df, path):
    latex_table = df.to_latex(
        index=False, float_format="%.4f", escape=True,
        caption="Comparison of the hybrid BERT pipeline, the LLM baselines, and CROS on the "
                "160-sentence Campidanese test set.",
        label="tab:comparison",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_table)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Builds runs/comparison.{csv,md,tex} from the canonical records in runs/records/."
    )
    return parser.parse_args()


def main():
    parse_args()
    df = build_comparison_table()

    csv_path = f"{OUTPUT_BASENAME}.csv"
    md_path = f"{OUTPUT_BASENAME}.md"
    tex_path = f"{OUTPUT_BASENAME}.tex"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    write_markdown(df, md_path)
    write_latex(df, tex_path)

    print(df.to_string(index=False))
    print(f"\nComparison files written:\n  - {csv_path}\n  - {md_path}\n  - {tex_path}")


if __name__ == "__main__":
    main()
