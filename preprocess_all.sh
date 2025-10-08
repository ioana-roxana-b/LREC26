#!/usr/bin/env bash
set -euo pipefail

# Mode: spacy | liwc
MODE="${MODE:-liwc}"

# List of PDNC novel directories

# Jane Austen
NOVELS=(
  "project-dialogism-novel-corpus/data/Emma"
  "project-dialogism-novel-corpus/data/MansfieldPark"
  "project-dialogism-novel-corpus/data/NorthangerAbbey"
  "project-dialogism-novel-corpus/data/Persuasion"
  "project-dialogism-novel-corpus/data/PrideAndPrejudice"
  "project-dialogism-novel-corpus/data/SenseAndSensibility"
)
CORPUS_NAME="austen_corpus"

# Edward Morgan Forster
#NOVELS=(
#  "project-dialogism-novel-corpus/data/APassageToIndia"
#  "project-dialogism-novel-corpus/data/ARoomWithAView"
#  "project-dialogism-novel-corpus/data/HowardsEnd"
#  "project-dialogism-novel-corpus/data/WhereAngelsFearToTread"
#)
#CORPUS_NAME=forster_corpus

#NOVELS=(
#  "project-dialogism-novel-corpus/data/HardTimes"
#  "project-dialogism-novel-corpus/data/OliverTwist"
#)
#CORPUS_NAME=dickens_corpus

# Output directory for the combined corpus
OUT_DIR="${OUT_DIR:-outputs/${CORPUS_NAME}/${MODE}}"

# Character info pattern (if available per novel)
CHAR_INFO_PATTERN="project-dialogism-novel-corpus/data/{novel}/character_info.csv"

# Lowercase flag: 1 = lowercase all text
LOWERCASE="${LOWERCASE:-1}"

# Run
python main.py \
  --stage preprocess \
  --mode "$MODE" \
  --novel_dir "${NOVELS[@]}" \
  --out_dir "$OUT_DIR" \
  --character_info "$CHAR_INFO_PATTERN" \
  $( [ "$LOWERCASE" = "1" ] && echo --lowercase )
