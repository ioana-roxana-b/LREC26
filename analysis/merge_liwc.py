from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Canonical LIWC family names we care about and their common aliases in exports
ALIASES: Dict[str, str] = {
    "article": "articles",
    "auxverb": "auxiliary_verbs",
    "conj": "conjunctions",
    "adverb": "adverbs",
    "ipron": "impersonal_pronouns",
    "negate": "negations",
    "ppron": "personal_pronouns",
    "prep": "prepositions",
    "quant": "quantifiers",
}


def merge_liwc_back(
    pairs_csv: Path,
    liwc_results_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """
    Merge a LIWC desktop export back onto the A→B pairs and emit binary presence flags
    for the nine target families as a_* and b_* columns. Also carries A_* and B_* metadata
    from the pairs file through to the output.

    Params:
        pairs_csv: Path to the pairs file produced in preprocessing (must include columns A and B).
        liwc_results_csv: Path to the LIWC-22 export CSV for this condition.
        out_csv: Destination CSV path for the merged dataset with a_* and b_* binary flags.

    Returns:
        A pandas DataFrame containing the merged rows that were also written to out_csv.
    """
    # 1) Load and normalize pairs
    pairs = pd.read_csv(pairs_csv, encoding="utf-8-sig")
    if not {"A", "B"}.issubset(pairs.columns):
        raise ValueError("pairs file must contain 'A' and 'B' speaker columns")
    pairs = pairs.rename(columns={"A": "a_speaker", "B": "b_speaker"})

    # 2) Load LIWC export
    liwc = pd.read_csv(liwc_results_csv, encoding="utf-8-sig", engine="python")

    # 3) Determine UID columns (pair_id and side) for the LIWC rows
    #    Try explicit columns first, else try to infer from a likely UID column,
    #    else fall back to row order relative to pairs.
    def _find_uid_column_local(df: pd.DataFrame) -> Optional[str]:
        norm_cols = {c: str(c).strip() for c in df.columns}
        df = df.rename(columns=norm_cols)
        candidates = {"uid", "segment", "segment id", "segmentid", "filename", "file", "id"}
        for c in df.columns:
            if str(c).lower().strip() in candidates:
                return c
        for c in df.columns:
            s = df[c].astype(str)
            if s.str.contains(r"\bp\d{5}_[AB]\b", na=False).any():
                return c
        return None

    def _recover_by_row_order_local(liwc_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        pairs_local = pd.read_csv(pairs_csv, encoding="utf-8-sig")
        expected = 2 * len(pairs_local)
        n = len(liwc_df)
        if n != expected:
            raise ValueError(
                f"LIWC export has {n} rows but expected {expected} based on pairs. "
                f"Cannot recover pair_id and side by row order."
            )
        idx = pd.RangeIndex(n)
        pair_id = (idx // 2).astype(int).astype(str).str.zfill(5)
        side = idx.map(lambda k: "A" if k % 2 == 0 else "B")
        return pair_id, side

    if {"pair_id", "side"}.issubset(liwc.columns):
        liwc["pair_id"] = liwc["pair_id"].astype(str).str.zfill(5)
        liwc["side"] = liwc["side"].astype(str).str.upper().str[0]
    else:
        uid_col = _find_uid_column_local(liwc)
        if uid_col is not None:
            parsed = liwc[uid_col].astype(str).str.extract(r"p(?P<pair_id>\d{5})_(?P<side>[AB])")
            if parsed.isnull().values.any():
                # Fallback to row order if pattern is inconsistent
                pair_by_row, side_by_row = _recover_by_row_order_local(liwc)
                liwc["pair_id"] = pair_by_row
                liwc["side"] = side_by_row
            else:
                liwc["pair_id"] = parsed["pair_id"]
                liwc["side"] = parsed["side"]
        else:
            # No UID column found, recover by row order
            pair_by_row, side_by_row = _recover_by_row_order_local(liwc)
            liwc["pair_id"] = pair_by_row
            liwc["side"] = side_by_row

    # 4) Map LIWC columns to our canonical nine families
    def _map_liwc_columns_local(liwc_df: pd.DataFrame) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        cols = list(liwc_df.columns)
        for short, long_name in ALIASES.items():
            hit = None
            # strict word-boundary search
            pat = rf"\b{re.escape(short)}\b"
            for c in cols:
                if re.search(pat, str(c).lower()):
                    hit = c
                    break
            # permissive prefix search if strict failed
            if not hit:
                for c in cols:
                    if str(c).lower().startswith(short):
                        hit = c
                        break
            if not hit:
                preview = ", ".join(map(str, cols[:12]))
                raise ValueError(
                    f"Could not find LIWC column for '{short}'. Some headers were: {preview} ..."
                )
            mapping[long_name] = hit
        return mapping

    cat_map = _map_liwc_columns_local(liwc)

    # Keep only what we need from LIWC and coerce to binary flags
    keep = ["pair_id", "side"] + list(cat_map.values())
    liwc = liwc[keep].copy()
    for long_name, c in cat_map.items():
        liwc[c] = (pd.to_numeric(liwc[c], errors="coerce").fillna(0) > 0).astype(int)

    # 5) Split A and B, rename to a_* and b_*, then pivot to one row per pair_id
    a = liwc[liwc["side"] == "A"].copy()
    b = liwc[liwc["side"] == "B"].copy()
    a = a.rename(columns={v: f"a_{k}" for k, v in cat_map.items()})
    b = b.rename(columns={v: f"b_{k}" for k, v in cat_map.items()})

    merged_flags = a[["pair_id"] + [f"a_{k}" for k in cat_map]].merge(
        b[["pair_id"] + [f"b_{k}" for k in cat_map]],
        on="pair_id",
        how="inner",
    )

    # 6) Join flags back onto pairs using stable numeric index as pair_id
    pairs = pairs.reset_index().rename(columns={"index": "pair_id"})
    pairs["pair_id"] = pairs["pair_id"].astype(str).str.zfill(5)

    out = pairs.merge(merged_flags, on="pair_id", how="inner")

    # 7) Write
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows with LIWC features to {out_csv}")
    return out


def run_for_condition_merge(
    pairs_csv: Path,
    liwc_results_csv: Path,
    out_dir: Path,
    label: str,  # "adjacent" | "nonadjacent" | "randomized"
) -> Path:
    """
    Merge LIWC features for a given condition and write a standard filename.

    Params:
        pairs_csv: Path to the pairs file for this condition.
        liwc_results_csv: Path to the LIWC export corresponding to the same condition.
        out_dir: Folder where the merged file should be written.
        label: Condition label. One of "adjacent", "nonadjacent", or "randomized".

    Returns:
        Path to the written merged CSV that includes the nine a_* and b_* binary flags.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / "pairs_liwc9_features.csv"
        if label == "adjacent"
        else out_dir / f"pairs_liwc9_features_{label}.csv"
    )
    merge_liwc_back(pairs_csv=pairs_csv, liwc_results_csv=liwc_results_csv, out_csv=out_path)
    return out_path
