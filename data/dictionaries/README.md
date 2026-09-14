# Dictionaries

Vocabulary sources used by the pipeline's SymSpell candidate generator.

| File | Source | Used by the final pipeline? |
|---|---|---|
| `corpus_dict.txt` | Extracted from the training corpus (`build_vocabulary.py`) | Yes |
| `campidanese_standard.txt` | Scraped from sardu.wiki, standard Campidanese lemmas (`scrape_dictionaries.py`) | Yes |
| `campidanese_non_standard.txt` | Scraped from sardu.wiki, non-standard Campidanese lemmas (`scrape_dictionaries.py`) | Yes |
| `wikipedia_dict.txt` | Extracted from the `wikimedia/wikipedia` (`sc`) Hugging Face dataset (`build_vocabulary.py`) | No — kept for the coverage analysis only; too noisy in practice (LSC/mixed varieties, proper nouns, typos) |
| `ditzionariu_in_linia.txt` | Scraped from ditzionariu.nor-web.eu (`scrape_dictionaries.py`) | No — kept for provenance/comparison only; mixes LSC and other varieties outside the Campidanese domain |

See the paper for the full rationale. The three "Yes" dictionaries are the
default `--dict-paths` in `src/evaluation/evaluate_pipeline.py` and
`src/evaluation/grid_search.py`.
