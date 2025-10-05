# Preprocessing and feature pipeline

This pipeline prepares novel dialogue data into cleaned quotes, paired utterances, and linguistic features or LIWC input.

## Files

| File                            | Description                                                                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`preprocess_data.py`**        | Cleans `quotation_info.csv`, keeping main fields (`quoteID`, `quoteText`, `quoteBy`, `quoteTo`), normalizing text and saving `cleaned_quotes.csv`.     |
| **`build_pairs.py`**            | Merges contiguous turns by the same speaker and builds dialogue pairs (adjacent, non-adjacent, randomized). Adds gender and category metadata. |
| **`features.py`**               | Extracts spaCy-based token family counts (articles, pronouns, verbs, etc.) for A/B sides of pairs. Outputs `pairs_features*.csv`.                      |
| **`prepare_liwc.py`**           | Converts pairs into LIWC-compatible input (`uid`, `pair_id`, `side`, `text`).                                                                          |
| **`preprocessing_pipeline.py`** | Main orchestrator that runs all steps for one or multiple novels, producing either LIWC or spaCy outputs.                                              |

## Usage

### One Novel

```bash
bash preprocess_one.sh
```

### All Novels

```bash
bash preprocess_all.sh
```

