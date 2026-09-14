"""
Compares the hybrid BERT pipeline with the regional spellchecker "CROS -
Curretore Regionale Ortogràficu Sardu in lìnia v.2.0" (Condaghes / Regione
Autonoma della Sardegna), publicly available at https://cros.nor-web.eu/.

Methodology (to stay directly comparable with the numbers already reported for
the BERT pipeline and the LLM baselines):

1. The exact same 160-sentence test set and synthetic corruption is rebuilt
   deterministically (see evaluation/common.py: SyntheticErrorGenerator,
   error_rate=0.15, seed=42). The resulting baseline WER/CER is printed so it can
   be checked against the historical value (0.1174 / 0.0226).

2. CROS does not expose a "correct the whole sentence" endpoint: its web
   interface flags unknown words (action=spellcheck) and, only on request for one
   specific word, returns a list of suggestions (action=suggest, no sentence
   context). To get an output comparable to the BERT pipeline and the LLMs, every
   flagged word is replaced with the first suggestion CROS returns (if any) — the
   same behavior as a user who always accepts the top suggestion.

3. Metrics (Precision/Recall/F0.5, WER, CER, average latency) are computed with
   the exact same logic used everywhere else (see evaluation/common.py), to stay
   1:1 comparable.

Methodological caveats to report in the paper:
 - CROS targets standardized Sardinian (LSC) and its varieties (VAR), not
   Campidanese specifically: it can flag valid dialectal forms or proper nouns
   absent from its dictionary as false positives.
 - The "full correction" measured here is our own reconstruction (first
   suggestion accepted), not a native CROS feature.
 - Latency includes the network round-trip to a remote public server, so it is
   not directly comparable to the local (GPU) latency measured for the BERT
   pipeline and the LLMs.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime

import jiwer
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from evaluation.common import build_test_set, compute_correction_metrics, save_record  # noqa: E402

RUNS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "runs")
DETAILS_DIR = os.path.join(RUNS_DIR, "details")

CROS_ENDPOINT = "https://cros.nor-web.eu/cros_inlinia.php"
CROS_USER_AGENT = (
    "Mozilla/5.0 (compatible; Limbacheck-Benchmark/1.0; "
    "academic research, UniCA - Sardinian spellchecker comparison)"
)
REQUEST_DELAY_S = 0.2
REQUEST_TIMEOUT_S = 15
MAX_RETRIES = 3

# CROS flags both plain spelling errors (<span>) and grammar/style anomalies
# (e.g. <span class="sc_misspelled_gram">) through the same client-side click
# mechanism (see cros_inlinia.js: newDiv.getElements('span').each(...) applies to
# ANY span, regardless of class). The regexes must therefore accept spans with or
# without attributes.
SPAN_SPLIT_PATTERN = re.compile(r"(<span[^>]*>.*?</span>)")
SPAN_MATCH_PATTERN = re.compile(r"^<span[^>]*>(.*?)</span>$")


def _post_with_retry(session, data):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(CROS_ENDPOINT, data=data, timeout=REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.5 * attempt)
    raise last_exc


def cros_spellcheck(session, text, language):
    txt = _post_with_retry(session, {"action": "spellcheck", "language": language, "spellText": text})
    return txt.strip()


def cros_suggest(session, word, language):
    """
    Returns the ordered list of suggestions for a word.
    For grammar/homophone-type anomalies (e.g. "che" as pronoun vs. conjunction),
    CROS returns multiple <div class="suggestion"> blocks with category labels
    interleaved as plain text (e.g. "pronùmene: <div>...</div>"); the class
    attribute may appear in a different order than in a purely orthographic
    suggestion (e.g. with a style attribute first), so the pattern doesn't assume
    a fixed attribute order.
    """
    txt = _post_with_retry(session, {"action": "suggest", "language": language, "suggestionText": word})
    if txt.strip() == "0" or not txt.strip():
        return []
    return re.findall(r'<div[^>]*class="suggestion"[^>]*>(.*?)</div>', txt)


def parse_spellcheck_response(html_text):
    """Returns a list of (segment, is_flagged), preserving original spacing/punctuation."""
    if html_text.strip() == "0":
        return []
    parts = SPAN_SPLIT_PATTERN.split(html_text)
    segments = []
    for part in parts:
        if part == "":
            continue
        m = SPAN_MATCH_PATTERN.match(part)
        if m:
            segments.append((m.group(1), True))
        else:
            segments.append((part, False))
    return segments


def correct_with_cros(session, text, language, stats):
    """
    Runs detection + (reconstructed) correction on a single sentence via CROS.
    The measured latency is the sum of only the HTTP request/response times
    (spellcheck + the suggest calls needed), excluding the REQUEST_DELAY_S we
    insert between calls to avoid overloading the public service: that delay is
    our own throttling artifact, not a property of CROS itself, and including it
    would artificially inflate its latency.
    """
    total_request_time = 0.0

    t0 = time.perf_counter()
    raw = cros_spellcheck(session, text, language)
    total_request_time += time.perf_counter() - t0
    time.sleep(REQUEST_DELAY_S)

    segments = parse_spellcheck_response(raw)
    if not segments:
        return text, 0, total_request_time * 1000

    sentence_cache = {}
    flagged_count = 0
    rebuilt = []
    for seg_text, is_flagged in segments:
        if not is_flagged:
            rebuilt.append(seg_text)
            continue

        flagged_count += 1
        key = seg_text.lower()
        if key not in sentence_cache:
            try:
                t1 = time.perf_counter()
                suggestions = cros_suggest(session, seg_text, language)
                total_request_time += time.perf_counter() - t1
            except requests.RequestException:
                stats["failed_requests"] += 1
                suggestions = []
            time.sleep(REQUEST_DELAY_S)
            sentence_cache[key] = suggestions

        suggestions = sentence_cache[key]
        rebuilt.append(suggestions[0] if suggestions else seg_text)

    return "".join(rebuilt), flagged_count, total_request_time * 1000


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluates CROS on the Limbacheck test set.")
    parser.add_argument("--language", choices=["VAR", "LSC"], default="VAR",
                         help="CROS language mode: VAR (varieties, the homepage default) or LSC (standard).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of sentences (useful for quick tests).")
    return parser.parse_args()


def main():
    args = parse_args()
    language, limit = args.language, args.limit

    print("[1/4] Rebuilding the test set (160 sentences, seed=42, error_rate=0.15)...")
    df_test = build_test_set()
    if limit:
        df_test = df_test.head(limit).reset_index(drop=True)

    baseline_wer = jiwer.wer(df_test["original_text"].tolist(), df_test["corrupted_text"].tolist())
    baseline_cer = jiwer.cer(df_test["original_text"].tolist(), df_test["corrupted_text"].tolist())
    print(f"      Baseline WER/CER (corrupted text vs ground truth): {baseline_wer:.4f} / {baseline_cer:.4f}")
    print("      (Expected: 0.1174 / 0.0226 — matches the paper's reported baseline if the split is unchanged)")

    print(f"[2/4] Querying CROS ({CROS_ENDPOINT}, language={language}) on {len(df_test)} sentences...")
    session = requests.Session()
    session.headers.update({"User-Agent": CROS_USER_AGENT})
    stats = {"failed_requests": 0}

    predicted_texts, flagged_counts, latencies = [], [], []
    for idx, row in df_test.iterrows():
        try:
            pred, n_flagged, lat = correct_with_cros(session, row["corrupted_text"], language, stats)
        except requests.RequestException as exc:
            print(f"      [WARNING] Sentence {idx} failed ({exc}); keeping the corrupted text unchanged.")
            pred, n_flagged, lat = row["corrupted_text"], 0, 0.0
            stats["failed_requests"] += 1

        predicted_texts.append(pred)
        flagged_counts.append(n_flagged)
        latencies.append(lat)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(df_test):
            print(f"      ... {idx + 1}/{len(df_test)} sentences processed")

    df_test["predicted_text"] = predicted_texts
    df_test["flagged_words"] = flagged_counts
    df_test["latency_ms"] = latencies

    print("[3/4] Computing metrics...")
    metrics = compute_correction_metrics(df_test)
    wer_score = jiwer.wer(df_test["original_text"].tolist(), df_test["predicted_text"].tolist())
    cer_score = jiwer.cer(df_test["original_text"].tolist(), df_test["predicted_text"].tolist())
    avg_latency_ms = sum(latencies) / len(latencies)

    print("\n=== [CROS - END-TO-END METRICS] ===")
    print(f"Language mode:         {language}")
    print(f"Correction Precision:  {metrics['Correction Precision']:.4f}")
    print(f"Correction Recall:     {metrics['Correction Recall']:.4f}")
    print(f"Correction F0.5:       {metrics['Correction F0.5']:.4f}")
    print(f"WER:                   {wer_score:.4f} (Baseline: {baseline_wer:.4f})")
    print(f"CER:                   {cer_score:.4f} (Baseline: {baseline_cer:.4f})")
    print(f"Average latency:       {avg_latency_ms:.2f} ms/sentence (network round-trip included)")
    print(f"Failed requests:       {stats['failed_requests']}")

    os.makedirs(DETAILS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = os.path.join(DETAILS_DIR, f"cros_{language}_{timestamp}.csv")
    df_test.to_csv(details_path, index=False, encoding="utf-8")
    print(f"\n[4/4] Per-sentence detail saved to: {details_path}")

    record = {
        "Timestamp": [timestamp],
        "System": [f"CROS ({language})"],
        "Avg Latency (ms)": [round(avg_latency_ms, 2)],
        "Correction F0.5": [round(metrics["Correction F0.5"], 4)],
        "Correction Precision": [round(metrics["Correction Precision"], 4)],
        "Correction Recall": [round(metrics["Correction Recall"], 4)],
        "WER": [round(wer_score, 4)],
        "CER": [round(cer_score, 4)],
        "Baseline WER": [round(baseline_wer, 4)],
        "Baseline CER": [round(baseline_cer, 4)],
        "TP": [metrics["TP"]], "FP": [metrics["FP"]], "FN": [metrics["FN"]], "TN": [metrics["TN"]],
        "Failed Requests": [stats["failed_requests"]],
    }
    csv_path = save_record(record, "cros_testing_records.csv")
    print(f"Summary row appended to: {csv_path}")


if __name__ == "__main__":
    main()
