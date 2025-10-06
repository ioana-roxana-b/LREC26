import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def compute_convergence_spacy(
    pairs_features_path: Path,
    out_conv_pair_feature: Path,
    out_conv_edge: Path,
    min_support: int = 10,
    family_rename: Dict[str, str] | None = None,
    allowed_families: Optional[set] = None,
    min_resp_support: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Compute CAT style convergence from a spaCy features table and write per pair per family rows and per edge summaries.

    Params:
        pairs_features_path: CSV with columns A and B and for each raw family F the flags F_i_present and F_j_present
                             values are 0 or 1 or numerics that will be coerced to 0 or 1
        out_conv_pair_feature: Output path for the per pair per family convergence table with counts and p0 p1 conv
                               plus any available a_* and b_* metadata
        out_conv_edge: Output path for the per edge summary table with mean_conv and mean_conv_w
                       plus one copy of a_* and b_* metadata
        min_support: Minimum number of A triggers within a dyad required to include that family
        family_rename: Optional mapping from raw family names to normalized names
        allowed_families: Optional set of normalized family names to keep
        min_resp_support: Optional minimum number of B positives within the triggered set required to keep a row

    Returns:
        A tuple with three items
        First is the per pair per family convergence data frame
        Second is the per edge summary data frame
        Third is the list of normalized family names used
    """
    # 1. Load
    df = pd.read_csv(pairs_features_path, encoding="utf-8-sig")

    # 2. Detect families from *_i_present
    fams_raw = sorted({c[:-10] for c in df.columns if c.endswith("_i_present")})
    if not fams_raw:
        raise ValueError("No *_i_present columns found in the features table.")

    # 3. Normalize family names inline
    raw_to_norm: Dict[str, str] = {}
    for raw in fams_raw:
        base = raw.strip()
        if family_rename and raw in family_rename:
            raw_to_norm[raw] = family_rename[raw]
            continue
        low = base.lower()
        low = re.sub(r'^(liwc|spacy|feat|feature|cat|category|ling|lex|style)_+', '', low)
        if family_rename and low in family_rename:
            raw_to_norm[raw] = family_rename[low]
        else:
            raw_to_norm[raw] = low

    fams_norm = sorted(set(raw_to_norm.values()))
    if allowed_families is not None:
        fams_norm = [f for f in fams_norm if f in allowed_families]
        if not fams_norm:
            raise ValueError("After applying allowed_families there are no families left.")

    # 4. Standardize speaker columns and OR any duplicate raw families into one normalized family
    work = df.rename(columns={"A": "a_speaker", "B": "b_speaker"}).copy()

    for fam in fams_norm:
        raws_for_fam = [r for r, n in raw_to_norm.items() if n == fam]
        i_cols = [f"{r}_i_present" for r in raws_for_fam if f"{r}_i_present" in work.columns]
        j_cols = [f"{r}_j_present" for r in raws_for_fam if f"{r}_j_present" in work.columns]

        if i_cols:
            tmp = work[i_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            work[f"{fam}_i_present"] = tmp.max(axis=1) if len(i_cols) > 1 else tmp.iloc[:, 0]
        if j_cols:
            tmp = work[j_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            work[f"{fam}_j_present"] = tmp.max(axis=1) if len(j_cols) > 1 else tmp.iloc[:, 0]

    # 5. Coerce to clean binary values 0 or 1
    bin_cols = [f"{f}_i_present" for f in fams_norm] + [f"{f}_j_present" for f in fams_norm]
    for c in bin_cols:
        if c in work.columns:
            s = pd.to_numeric(work[c], errors="coerce").fillna(0).astype(int)
            work[c] = s.clip(0, 1)

    # 6. Keep normalized presence columns and metadata
    keep_cols = ["a_speaker", "b_speaker"] + bin_cols
    meta_cols_A = [c for c in df.columns if c.startswith("A_")]
    meta_cols_B = [c for c in df.columns if c.startswith("B_")]
    keep_cols += [c for c in meta_cols_A + meta_cols_B if c in work.columns]
    work = work[keep_cols].copy()

    # 7. Build per edge metadata once and for all
    edge_meta = None
    if meta_cols_A or meta_cols_B:
        meta_renames = {c: "a_" + c[2:] for c in meta_cols_A}
        meta_renames.update({c: "b_" + c[2:] for c in meta_cols_B})
        wm = work[["a_speaker", "b_speaker"]].copy()
        for c in meta_cols_A + meta_cols_B:
            if c in work.columns:
                wm[meta_renames[c]] = work[c]
        edge_meta = (wm.groupby(["a_speaker", "b_speaker"], as_index=False)
                       .agg({col: "first" for col in wm.columns if col not in {"a_speaker", "b_speaker"}}))

    # 8. Per A B family convergence rows
    rows: List[Dict[str, object]] = []
    for (A, B), g in work.groupby(["a_speaker", "b_speaker"], dropna=False):
        n_edge = len(g)
        for fam in fams_norm:
            ip = f"{fam}_i_present"
            jp = f"{fam}_j_present"
            if ip not in g.columns or jp not in g.columns:
                continue

            p0 = float(g[jp].mean())
            g_trig = g[g[ip] == 1]
            n_trig = int(len(g_trig))
            if n_trig < min_support:
                continue

            p1 = float(g_trig[jp].mean())
            n_b_pos = int(g_trig[jp].sum())
            if min_resp_support is not None and n_b_pos < min_resp_support:
                continue

            rows.append({
                "a_speaker": A,
                "b_speaker": B,
                "family": fam,
                "n_pairs": n_edge,
                "n_triggers": n_trig,
                "n_b_positives_given_trigger": n_b_pos,
                "p1": p1,
                "p0": p0,
                "conv": p1 - p0,
            })

    conv_pf = pd.DataFrame(rows)
    if not conv_pf.empty:
        conv_pf = conv_pf.sort_values(["a_speaker", "b_speaker", "family"]).reset_index(drop=True)

    if edge_meta is not None and not conv_pf.empty:
        conv_pf = conv_pf.merge(edge_meta, on=["a_speaker", "b_speaker"], how="left")

    out_conv_pair_feature.parent.mkdir(parents=True, exist_ok=True)
    conv_pf.to_csv(out_conv_pair_feature, index=False, encoding="utf-8-sig")

    # 9. Edge summary with plain and weighted means
    if not conv_pf.empty:
        def _edge_agg(g: pd.DataFrame) -> pd.Series:
            weights = g["n_triggers"].to_numpy(dtype=float)
            vals = g["conv"].to_numpy(dtype=float)
            w_sum = weights.sum()
            mean_w = float(np.average(vals, weights=weights)) if w_sum > 0 else float(vals.mean())
            return pd.Series({
                "n_features": int(g["family"].nunique()),
                "n_pairs": int(g["n_pairs"].max()),
                "n_triggers_total": int(weights.sum()),
                "mean_conv": float(vals.mean()),
                "mean_conv_w": mean_w,
            })

        edges = conv_pf.groupby(["a_speaker", "b_speaker"], as_index=False).apply(_edge_agg)
        if not {"a_speaker", "b_speaker"}.issubset(edges.columns):
            edges = edges.reset_index()
        if edge_meta is not None:
            edges = edges.merge(edge_meta, on=["a_speaker", "b_speaker"], how="left")
    else:
        edges = pd.DataFrame(columns=[
            "a_speaker", "b_speaker", "n_features", "n_pairs",
            "n_triggers_total", "mean_conv", "mean_conv_w"
        ])

    edges.to_csv(out_conv_edge, index=False, encoding="utf-8-sig")

    print(f"Saved pair-feature convergence → {out_conv_pair_feature} rows={len(conv_pf)}")
    print(f"Saved edge summary             → {out_conv_edge} rows={len(edges)}")

    return conv_pf, edges, fams_norm
