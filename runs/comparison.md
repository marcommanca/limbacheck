# Comparison: Hybrid Pipeline vs LLM Baselines vs CROS

All systems were evaluated on the same 160-sentence test split (error_rate=0.15, seed=42; see src/evaluation/common.py). For CROS, "correction" is our own reconstruction (first suggestion accepted for each flagged word) since CROS has no native full-sentence correction or contextual disambiguation, unlike the proposed MLM-ranked pipeline. CROS latency includes the network round-trip to a remote public server and is not directly comparable to the local (GPU) latency of the BERT pipeline and the LLM baselines.

| System                          |   F0.5 Score |   Precision |   Recall |    WER |    CER |   Latency (ms) |
|:--------------------------------|-------------:|------------:|---------:|-------:|-------:|---------------:|
| Hybrid Pipeline (BERT)          |       0.7477 |      0.8027 |   0.5868 | 0.0502 | 0.0108 |        20.8800 |
| sardinian_gemma_2_9B            |       0.7348 |      0.7310 |   0.7506 | 0.0442 | 0.0122 |      2145.1500 |
| sardinian_llama_3.1_8B          |       0.7132 |      0.7168 |   0.6993 | 0.0499 | 0.0154 |      1384.5200 |
| sardinian_llama_3.1_8B_Instruct |       0.7084 |      0.7180 |   0.6724 | 0.0505 | 0.0143 |      1439.3100 |
| sardinian_gemma_2_9B_Instruct   |       0.6965 |      0.6929 |   0.7115 | 0.0508 | 0.0148 |      2197.9600 |
| sardinian_llama_3.2_3B          |       0.6337 |      0.6423 |   0.6015 | 0.0634 | 0.0176 |      1180.7700 |
| sardinian_gemma_2_2B            |       0.6157 |      0.6234 |   0.5868 | 0.0640 | 0.0210 |      1285.8300 |
| sardinian_llama_3.2_3B_Instruct |       0.6071 |      0.6117 |   0.5892 | 0.0663 | 0.0222 |      1284.9400 |
| sardinian_llama_3.2_1B_Instruct |       0.5197 |      0.5259 |   0.4963 | 0.0746 | 0.0233 |       716.6800 |
| sardinian_llama_3.2_1B          |       0.5144 |      0.5192 |   0.4963 | 0.0861 | 0.0346 |       729.8900 |
| sardinian_gemma_2_2B_Instruct   |       0.1909 |      0.1636 |   0.5721 | 0.3343 | 0.2725 |      1700.7000 |
| CROS (VAR)                      |       0.1432 |      0.1321 |   0.2152 | 0.2149 | 0.0584 |      1229.7400 |
