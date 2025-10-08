#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
#MODE="${MODE:-liwc}"                              # spacy | liwc
#TITLE="TheGambler"
#PREP_DIR="${PREP_DIR:-outputs/${TITLE}/${MODE}}"  # folder produced by *preprocess* stage
#OUT_DIR="${OUT_DIR:-$PREP_DIR}"                   # where analysis_results/ will be written
#MIN_SUPPORT="${MIN_SUPPORT:-3}"

MODE="${MODE:-liwc}"                              # spacy | liwc
TITLE="All_Novels"
CORPUS_NAME="austen_corpus"
PREP_DIR="${PREP_DIR:-outputs/${CORPUS_NAME}/${MODE}}/${TITLE}"
OUT_DIR="${OUT_DIR:-$PREP_DIR}"
MIN_SUPPORT="${MIN_SUPPORT:-3}"

# LIWC exports (only used if MODE=liwc). Accept plain paths or {novel} pattern.
LIWC_ADJ_DEFAULT="$PREP_DIR/LIWC-22 Results - liwc_input - LIWC Analysis.csv"
LIWC_NON_DEFAULT="$PREP_DIR/LIWC-22 Results - liwc_input_nonadjacent - LIWC Analysis.csv"
LIWC_RND_DEFAULT="$PREP_DIR/LIWC-22 Results - liwc_input_randomized - LIWC Analysis.csv"

LIWC_ADJ="${LIWC_ADJ:-$LIWC_ADJ_DEFAULT}"
LIWC_NON="${LIWC_NON:-$LIWC_NON_DEFAULT}"
LIWC_RND="${LIWC_RND:-$LIWC_RND_DEFAULT}"

# -------- Run --------
COMMON_ARGS=(
  --stage analysis
  --mode "$MODE"
  --novel_dir "$PREP_DIR"
  --out_dir "$OUT_DIR"
  --min_support "$MIN_SUPPORT"
)

if [[ "$MODE" == "liwc" ]]; then
  python main.py \
    "${COMMON_ARGS[@]}" \
    --liwc_results "$LIWC_ADJ" \
    --liwc_results_nonadj "$LIWC_NON" \
    --liwc_results_rand "$LIWC_RND"
else
  python main.py "${COMMON_ARGS[@]}"
fi
