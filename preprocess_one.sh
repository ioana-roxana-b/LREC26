#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-liwc}"  # spacy | liwc
NOVEL_DIR="${NOVEL_DIR:-project-dialogism-novel-corpus/data/Emma}"
TITLE="Emma"
OUT_DIR="${OUT_DIR:-outputs/${TITLE}/${MODE}}"
CHAR_INFO="${CHAR_INFO:-${NOVEL_DIR}/character_info.csv}"
LOWERCASE="${LOWERCASE:-0}"

python main.py \
  --stage preprocess \
  --mode "$MODE" \
  --novel_dir "$NOVEL_DIR" \
  --out_dir "$OUT_DIR" \
  --character_info "$CHAR_INFO" \
  $( [ "$LOWERCASE" = "1" ] && echo --lowercase )
