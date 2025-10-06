# LREC Conversational Alignment Pipeline

## Overview
This repository builds dialogue-alignment datasets and analyses from the Project Dialogism Novel Corpus (PDNC). The workflow has two stages:
1. **Preprocessing** cleans PDNC quotation exports, merges contiguous speaker turns, constructs conversational pairs, and either prepares LIWC-ready CSVs or extracts spaCy-based feature counts.
2. **Analysis** combines LIWC outputs or spaCy features to compute convergence metrics, summary statistics, graph views, and plots.

Use `main.py` to run both stages with a shared command-line interface.

## Directory Layout
- `analysis/` – convergence scoring, statistical testing, visualization, and graph utilities used during the analysis stage.
- `outputs/` – default location for generated artifacts. Organized by corpus name, processing mode (`liwc` or `spacy`), and novel.
- `preprocessing/` – quote cleaning (`preprocess_data.py`), turn pairing (`build_pairs.py`), feature extraction (`features.py`), LIWC preparation (`prepare_liwc.py`), and orchestration helpers (`preprocessing_pipeline.py`).
- `analysis.sh`, `preprocess_one.sh`, `preprocess_all.sh` – bash helpers that wrap `main.py` for common pipelines.
- `main.py` – single entry point that exposes the two pipeline stages.

## Environment Setup
1. Create and activate a virtual environment (example uses `python` on PATH):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # PowerShell
   # or source .venv/bin/activate on Unix shells
   ```
2. Install Python dependencies (minimum):
   ```bash
   pip install pandas numpy spacy scipy matplotlib networkx
   ```
3. Download the spaCy English model required for feature extraction:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. (LIWC mode only) Prepare LIWC Desktop 2022 and ensure you can export CSVs for each generated LIWC input file.

## Data Requirements
- Place the PDNC novel folders inside `project-dialogism-novel-corpus/data/` (the helper scripts assume this path).
- Each novel directory must include `quotation_info.csv`. If `character_info.csv` is present, the pipeline attaches gender/category metadata to each speaker pair.
- For corpus-level runs, ensure every listed novel folder exists before launching the pipeline.

## Running the Pipeline
All commands are issued from the repository root.

### Preprocess a Single Novel
```bash
python main.py \
  --stage preprocess \
  --mode liwc \                    # or spacy
  --novel_dir project-dialogism-novel-corpus/data/Emma \
  --out_dir outputs/Emma/liwc \
  --character_info project-dialogism-novel-corpus/data/Emma/character_info.csv \
  --lowercase                      # optional flag
```
Outputs (e.g., `cleaned_quotes.csv`, `pairs_A_to_B.csv`, `liwc_input.csv`) are written under the chosen `--out_dir`.

### Preprocess Multiple Novels into a Corpus
```bash
python main.py \
  --stage preprocess \
  --mode spacy \
  --novel_dir project-dialogism-novel-corpus/data/Emma \
              project-dialogism-novel-corpus/data/MansfieldPark \
  --out_dir outputs/austen_corpus/spacy \
  --character_info "project-dialogism-novel-corpus/data/{novel}/character_info.csv"
```
Passing multiple `--novel_dir` values triggers corpus aggregation and writes merged outputs to `outputs/austen_corpus/spacy/All_Novels/`.

### Run Analysis (spaCy mode)
```bash
python main.py \
  --stage analysis \
  --mode spacy \
  --novel_dir outputs/austen_corpus/spacy/Emma \
  --out_dir outputs/austen_corpus/spacy/Emma \
  --min_support 5
```
This step reads `pairs_features*.csv`, computes convergence scores, plots, graphs, and a stats summary in `analysis_results/`.

### Run Analysis (LIWC mode)
After exporting LIWC results for each `liwc_input*.csv`, supply their paths (wildcards like `{novel}` are expanded automatically):
```bash
python main.py \
  --stage analysis \
  --mode liwc \
  --novel_dir outputs/austen_corpus/liwc/Emma \
  --out_dir outputs/austen_corpus/liwc/Emma \
  --min_support 5 \
  --liwc_results "outputs/austen_corpus/liwc/Emma/LIWC-22 Results - liwc_input - LIWC Analysis.csv" \
  --liwc_results_nonadj "outputs/austen_corpus/liwc/Emma/LIWC-22 Results - liwc_input_nonadjacent - LIWC Analysis.csv" \
  --liwc_results_rand "outputs/austen_corpus/liwc/Emma/LIWC-22 Results - liwc_input_randomized - LIWC Analysis.csv"
```
The analysis stage merges LIWC metrics with the pairs data, produces convergence tables, statistical comparisons, plots, and graph exports under `analysis_results/`.

### Helper Scripts
- `preprocess_one.sh` / `preprocess_all.sh` mirror the commands above and accept environment variables (e.g., `MODE`, `OUT_DIR`, `LOWERCASE`). Run them from a bash-compatible shell.
- `analysis.sh` wrap the analysis stage for a single novel. Override defaults via environment variables (`MODE`, `TITLE`, `PREP_DIR`, `MIN_SUPPORT`, `LIWC_*`).

## Output Overview
Each novel processed under `outputs/<corpus>/<mode>/<NovelName>/` contains:
- `cleaned_quotes.csv` – normalized quotation records.
- `merged_turns.csv` – contiguous speaker turns with merge metadata.
- `pairs_A_to_B*.csv` – adjacent, nonadjacent, and randomized conversational pairs (with metadata when available).
- `liwc_input*.csv` or `pairs_features*.csv` – final artifacts for LIWC Desktop or spaCy convergence analysis.
- `analysis_results/` (after running the analysis stage):
  - `convergence_results/` – per-pair and per-dyad convergence summaries.
  - `plots/` – PNG visualizations of convergence patterns.
  - `graph_results/` – network layouts and edge lists.
  - `stats_results/` – bootstrap summaries and paired tests comparing conditions.

Corpus runs additionally produce `outputs/<corpus>/<mode>/All_Novels/` with concatenated pairs (and features/LIWC inputs) spanning every novel.

## Troubleshooting
- If spaCy raises a model loading error, confirm `en_core_web_sm` is installed in the active environment.
- LIWC analysis requires matching LIWC CSV exports. Re-run LIWC Desktop if the analysis stage reports missing files.
- The helper bash scripts expect Git Bash, WSL, or another Unix-style shell on Windows.
