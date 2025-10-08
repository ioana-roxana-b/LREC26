from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

ALLOWED_FAMILIES = {
    "articles",
    "auxiliary_verbs",
    "conjunctions",
    "adverbs",
    "impersonal_pronouns",
    "negations",
    "personal_pronouns",
    "prepositions",
    "quantifiers",
}


def compute_convergence_local(
    pairs_with_liwc_csv: Path,
    out_conv_pair_feature: Path,
    out_conv_edge: Path,
    min_support: int = 5,
    min_resp_support: Optional[int] = None,
    allowed_families: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute CAT style convergence for A to B dialogue pairs that include LIWC presence flags.
    The function writes per pair per family rows and per edge summaries and preserves A and B metadata.

    Params:
        pairs_with_liwc_csv: Input CSV that contains a_speaker and b_speaker plus a_<family> and b_<family> flags
                             and optional A_* and B_* metadata columns.
        out_conv_pair_feature: Output path for the per pair per family table.
        out_conv_edge: Output path for the per edge summary table.
        min_support: Minimum number of A triggers within a dyad required to include that family.
        min_resp_support: Optional minimum number of B positives within the triggered set required to keep a row.
        allowed_families: Optional set of family names to keep after matching a_* and b_*.

    Returns:
        A tuple with two data frames
        First is the per pair per family convergence table
        Second is the per edge summary table
    """
    df = pd.read_csv(pairs_with_liwc_csv, encoding="utf-8-sig")

    if not {"a_speaker", "b_speaker"}.issubset(df.columns):
        raise ValueError("pairs file must contain columns a_speaker and b_speaker")

    allowed = set(allowed_families) if allowed_families is not None else set(ALLOWED_FAMILIES)

    a_fams = {c[2:] for c in df.columns if c.startswith("a_")}
    b_fams = {c[2:] for c in df.columns if c.startswith("b_")}
    fams = sorted((a_fams & b_fams) & allowed)
    if not fams:
        raise ValueError(
            "No overlapping LIWC families found. "
            f"Available a_*: {sorted(a_fams)}; b_*: {sorted(b_fams)}; allowed: {sorted(allowed)}"
        )

    feature_cols = [f"a_{f}" for f in fams] + [f"b_{f}" for f in fams]
    for c in feature_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0)
            s = s.where(s <= 0, 1).astype(int)
            df[c] = s

    meta_cols_A = [c for c in df.columns if c.startswith("A_")]
    meta_cols_B = [c for c in df.columns if c.startswith("B_")]
    meta_renames: Dict[str, str] = {c: "a_" + c[2:] for c in meta_cols_A}
    meta_renames.update({c: "b_" + c[2:] for c in meta_cols_B})

    edge_meta = None
    if meta_renames:
        tmp = df[["a_speaker", "b_speaker"]].copy()
        for c in meta_cols_A + meta_cols_B:
            tmp[meta_renames[c]] = df[c]
        agg_map = {col: "first" for col in tmp.columns if col not in {"a_speaker", "b_speaker"}}
        edge_meta = tmp.groupby(["a_speaker", "b_speaker"], as_index=False).agg(agg_map)

    rows: List[Dict[str, object]] = []
    for (A, B), g in df.groupby(["a_speaker", "b_speaker"], dropna=False):
        n_pairs_edge = len(g)
        for fam in fams:
            a_col = f"a_{fam}"
            b_col = f"b_{fam}"
            p0 = float(g[b_col].mean()) if b_col in g else 0.0

            if a_col not in g:
                continue
            g_trig = g[g[a_col] == 1]
            n_trig = int(len(g_trig))
            if n_trig < min_support:
                continue

            p1 = float(g_trig[b_col].mean()) if b_col in g_trig else 0.0
            n_b_pos = int(g_trig[b_col].sum()) if b_col in g_trig else 0

            if min_resp_support is not None and n_b_pos < min_resp_support:
                continue

            rows.append({
                "a_speaker": A,
                "b_speaker": B,
                "family": fam,
                "n_pairs": n_pairs_edge,
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

    if not conv_pf.empty:
        edges = (
            conv_pf.groupby(["a_speaker", "b_speaker"], as_index=False)
            .agg(
                n_features=("family", "count"),
                n_pairs=("n_pairs", "max"),
                n_triggers_total=("n_triggers", "sum"),
                mean_conv=("conv", "mean"),
                mean_conv_w=("conv", lambda s: float(np.average(s, weights=conv_pf.loc[s.index, "n_triggers"]))),
            )
        )
        if edge_meta is not None:
            edges = edges.merge(edge_meta, on=["a_speaker", "b_speaker"], how="left")
    else:
        edges = pd.DataFrame(columns=[
            "a_speaker", "b_speaker", "n_features", "n_pairs",
            "n_triggers_total", "mean_conv", "mean_conv_w"
        ])

    edges.to_csv(out_conv_edge, index=False, encoding="utf-8-sig")

    print(f"Saved pair-feature convergence → {out_conv_pair_feature}  rows={len(conv_pf)}")
    print(f"Saved edge summary          → {out_conv_edge}           rows={len(edges)}")
    if fams:
        print(f"Families used: {', '.join(fams)}")

    return conv_pf, edges


def run_for_condition_convergence(
    pairs_with_liwc_csv: Path,
    out_dir: Path,
    label: str,   # "adjacent" | "nonadjacent" | "randomized"
    min_support: int = 10,
    min_resp_support: Optional[int] = None,
) -> None:
    """
    Compute convergence for one condition, write standard conv files, and also
    emit summaries split by gender/importance (family-level) and two-mode edge tables.

    Writes:
      conv_by_pair_feature_liwc{_*}.csv
      conv_edge_summary_liwc{_*}.csv
      conv_family_by_gender_responder_{label}.csv
      conv_family_by_gender_initiator_{label}.csv
      conv_family_by_importance_responder_{label}.csv
      conv_family_by_importance_initiator_{label}.csv
      edge_mean_by_gender_{label}.csv
      edge_mean_by_category_{label}.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if label == "adjacent":
        conv_by_pair = out_dir / "conv_by_pair_feature_liwc.csv"
        conv_edge    = out_dir / "conv_edge_summary_liwc.csv"
    elif label in {"nonadjacent", "randomized"}:
        conv_by_pair = out_dir / f"conv_by_pair_feature_liwc_{label}.csv"
        conv_edge    = out_dir / f"conv_edge_summary_liwc_{label}.csv"
    else:
        raise ValueError("label must be one of: 'adjacent', 'nonadjacent', 'randomized'")

    conv_pf, edges = compute_convergence_local(
        pairs_with_liwc_csv=pairs_with_liwc_csv,
        out_conv_pair_feature=conv_by_pair,
        out_conv_edge=conv_edge,
        min_support=min_support,
        min_resp_support=min_resp_support,
    )

    if conv_pf is None or conv_pf.empty or edges is None or edges.empty:
        return

    # GENDER: responder (B) and initiator (A)
    if "b_gender" in conv_pf.columns:
        (conv_pf.groupby(["family", "b_gender"])["conv"]
               .mean()
               .reset_index()
               .rename(columns={"b_gender": "gender", "conv": "mean_conv"})
               .to_csv(out_dir / f"conv_family_by_gender_responder_{label}.csv",
                       index=False, encoding="utf-8-sig"))

    if "a_gender" in conv_pf.columns:
        (conv_pf.groupby(["family", "a_gender"])["conv"]
               .mean()
               .reset_index()
               .rename(columns={"a_gender": "gender", "conv": "mean_conv"})
               .to_csv(out_dir / f"conv_family_by_gender_initiator_{label}.csv",
                       index=False, encoding="utf-8-sig"))

    if "b_category" in conv_pf.columns:
        (conv_pf.groupby(["family", "b_category"])["conv"]
               .mean()
               .reset_index()
               .rename(columns={"b_category": "importance", "conv": "mean_conv"})
               .to_csv(out_dir / f"conv_family_by_importance_responder_{label}.csv",
                       index=False, encoding="utf-8-sig"))

    if "a_category" in conv_pf.columns:
        (conv_pf.groupby(["family", "a_category"])["conv"]
               .mean()
               .reset_index()
               .rename(columns={"a_category": "importance", "conv": "mean_conv"})
               .to_csv(out_dir / f"conv_family_by_importance_initiator_{label}.csv",
                       index=False, encoding="utf-8-sig"))

    weight_col = "mean_conv_w" if "mean_conv_w" in edges.columns else (
        "mean_conv" if "mean_conv" in edges.columns else None
    )

    if weight_col is not None:

        # ---------- (A_gender, B_gender) ----------
        if {"a_gender", "b_gender"}.issubset(edges.columns):
            gtab = edges.copy()
            gtab["a_gender"] = gtab["a_gender"].astype(str).str.strip().str.upper()
            gtab["b_gender"] = gtab["b_gender"].astype(str).str.strip().str.upper()

            # keep only F/M
            mask_g = (gtab["a_gender"].isin(["F", "M"])) & (gtab["b_gender"].isin(["F", "M"]))
            gtab = gtab.loc[mask_g].copy()

            if not gtab.empty:
                (gtab.groupby(["a_gender", "b_gender"])[weight_col]
                 .agg(mean_weight="mean", count="size")
                 .reset_index()
                 .to_csv(out_dir / f"edge_mean_by_gender_{label}.csv",
                         index=False, encoding="utf-8-sig"))

        # ---------- (A_category, B_category) ----------
        if {"a_category", "b_category"}.issubset(edges.columns):
            ctab = edges.copy()
            ctab["a_category"] = ctab["a_category"].astype(str).str.strip().str.lower()
            ctab["b_category"] = ctab["b_category"].astype(str).str.strip().str.lower()

            # normalize a few common aliases (optional)
            norm = {"maj": "major", "main": "major", "primary": "major",
                    "int": "intermediate", "mid": "intermediate",
                    "min": "minor"}
            ctab["a_category"] = ctab["a_category"].replace(norm)
            ctab["b_category"] = ctab["b_category"].replace(norm)

            # drop blanks/NaNs
            mask_c = (ctab["a_category"] != "") & (ctab["b_category"] != "") \
                     & ctab["a_category"].notna() & ctab["b_category"].notna()
            ctab = ctab.loc[mask_c].copy()

            if not ctab.empty:
                (ctab.groupby(["a_category", "b_category"])[weight_col]
                 .agg(mean_weight="mean", count="size")
                 .reset_index()
                 .to_csv(out_dir / f"edge_mean_by_category_{label}.csv",
                         index=False, encoding="utf-8-sig"))
