#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
MODE="${MODE:-liwc}"                              # spacy | liwc
MIN_SUPPORT="${MIN_SUPPORT:-3}"

# Full list of novels
# CORPUS_NAME="austen_corpus"
# NOVELS=(
#   "All_Novels"
#   "Emma"
#   "MansfieldPark"
#   "NorthangerAbbey"
#   "Persuasion"
#   "PrideAndPrejudice"
#   "SenseAndSensibility"
# )

# CORPUS_NAME="forster_corpus"
# NOVELS=(
#  "All_Novels"
#  "APassageToIndia"
#  "ARoomWithAView"
#  "HowardsEnd"
#  "WhereAngelsFearToTread"
# )

CORPUS_NAME="dickens_corpus"
NOVELS=(
 "All_Novels"
 "HardTimes"
 "OliverTwist"
)

# -------- Loop over all novels --------
for TITLE in "${NOVELS[@]}"; do
  echo "=== Running analysis for: $TITLE ==="

  NOVEL="$TITLE"
  PREP_DIR="outputs/${CORPUS_NAME}/${MODE}/${TITLE}"
  OUT_DIR="$PREP_DIR"

  # LIWC exports (only used if MODE=liwc)
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
    --title "$NOVEL"
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

  echo "=== Finished: $TITLE ==="
  echo
done

echo "✅ All novels processed successfully."
