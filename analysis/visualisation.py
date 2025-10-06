from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# =========================
# Color-blind-friendly style
# =========================
# Okabe–Ito palette
OI_BLUE   = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN  = "#009E73"
OI_RED    = "#D55E00"
OI_PURPLE = "#CC79A7"
OI_BLACK  = "#000000"
OI_GREY   = "#7F7F7F"

plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# =========================
# Family- and edge-level plots
# =========================
def plot_family_conv_bars(pf: pd.DataFrame, out_png: Path, title: str) -> None:
    """
    Family averages of linguistic convergence.

    description:
        Computes the average convergence Conv(t) for each feature family and
        renders a horizontal bar chart. Conv(t) = P(B uses t | A uses t) − P(B uses t).
        Higher bars mean responders (B) match initiators (A) more on that family.

        Saves a CSV next to the PNG with the values shown in the plot.

    Params:
        pf: DataFrame with at least 'family' and 'conv' columns; typically one row per (A,B,family).
        out_png: Output path for the chart image (PNG). A CSV with the same basename is also written.
        title: Short, human-readable chart title (e.g., "Average convergence by feature family").

    Returns:
        None. Writes PNG and CSV to disk.
    """
    if pf is None or pf.empty or not {"family", "conv"}.issubset(pf.columns):
        return

    g = (pf.groupby("family", dropna=False)
           .agg(mean_conv=("conv", "mean"),
                median_conv=("conv", "median"),
                n=("conv", "size"))
           .reset_index()
           .sort_values("mean_conv", ascending=False))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(3.8, 0.42 * len(g))))
    bar_color = plt.get_cmap("cividis")(0.55)
    plt.barh(g["family"][::-1], g["mean_conv"][::-1], alpha=0.95,
             color=bar_color, edgecolor=OI_BLACK, linewidth=0.6)

    plt.xlabel("Mean convergence Conv(t) (higher = more matching)")
    plt.ylabel("Feature family")
    plt.title(title)
    plt.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.4, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    g.to_csv(out_png.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def plot_edge_distribution(edges: pd.DataFrame, out_png: Path, title: Optional[str] = None) -> None:
    """
    Distribution of per-pair (A→B) convergence.

    description:
        Plots a histogram of the per-edge convergence score, where an edge is a directed pair A→B.
        The score summarizes how strongly B adapts to A across all turns/features.
        Uses the most informative column available (in order): 'mean_conv_w' (weighted by #triggers),
        else 'weight_work', else 'mean_conv'. A dashed vertical line marks 0 (no adaptation);
        solid/dotted lines mark the mean and median of the distribution.

    Params:
        edges: Edge summary table with 'a_speaker','b_speaker' and at least one of
               'mean_conv_w' | 'weight_work' | 'mean_conv'.
        out_png: Output path for the PNG.
        title: Optional custom title. If None, a descriptive default is used.

    Returns:
        None. Writes PNG.
    """
    if edges is None or edges.empty:
        return

    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    vals = pd.to_numeric(edges[weight_col], errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.8, 5.4))
    bins = max(12, int(np.sqrt(len(vals))))

    # Bars
    plt.hist(
        vals, bins=bins, alpha=0.9,
        color=plt.get_cmap("cividis")(0.6),
        edgecolor=OI_BLACK, linewidth=0.6
    )

    # Reference lines
    plt.axvline(0, color=OI_BLACK, linewidth=1.2, linestyle="--", alpha=0.9, label="No adaptation (0)")
    mean_v = float(np.mean(vals))
    med_v  = float(np.median(vals))
    plt.axvline(mean_v, color=OI_BLUE,   linestyle="-",  linewidth=1.6, label=f"Mean = {mean_v:.3f}")
    plt.axvline(med_v,  color=OI_ORANGE, linestyle=":", linewidth=1.8, label=f"Median = {med_v:.3f}")

    # Labels & title (clear, self-contained)
    plt.xlabel("Per-pair convergence score (Conv; > 0 = B adapts to A, < 0 = diverges)")
    plt.ylabel("Number of A→B pairs")
    plt.title(title or "How strongly each pair (A→B) adapts overall")
    plt.legend(frameon=False, loc="upper left")
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()


def plot_top_edges(edges: pd.DataFrame, out_png: Path, title: str, top_k: int = 20) -> None:
    """
    Top A→B pairs by mean convergence.

    description:
        Ranks directed pairs (A→B) by mean convergence using the most informative column available:
        'mean_conv_w' (preferred), else 'weight_work', else 'mean_conv'. Shows the top_k pairs.
        Saves a CSV with the rows shown in the figure.

    Params:
        edges: Edge summary table with speakers and a convergence column (see above).
        out_png: Output path for the PNG; a CSV with plotted rows is also written.
        title: Chart title (e.g., "Top converging pairs (A→B)").
        top_k: Number of pairs to display (default 20).

    Returns:
        None. Writes PNG and CSV.
    """
    if edges is None or edges.empty:
        return
    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    work = edges.copy()
    work["__w__"] = pd.to_numeric(work[weight_col], errors="coerce")
    top = work.dropna(subset=["__w__"]).sort_values("__w__", ascending=False).head(top_k)
    if top.empty:
        return

    labels = [f"{a} → {b}" for a, b in zip(top["a_speaker"], top["b_speaker"])]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.4, max(3.6, 0.44 * len(top))))
    bar_color = plt.get_cmap("cividis")(0.55)
    plt.barh(labels[::-1], top["__w__"].to_numpy()[::-1], alpha=0.95,
             color=bar_color, edgecolor=OI_BLACK, linewidth=0.6)

    plt.xlabel("Mean edge convergence")
    plt.ylabel("Speaker pair")
    plt.title(title)
    plt.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.4, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    (top.drop(columns=["__w__"])
        .to_csv(out_png.with_suffix(".csv"), index=False, encoding="utf-8-sig"))


def plot_heatmap(pf: pd.DataFrame, out_png: Path, title: str,
                 min_support: int = 10, max_edges: int = 50) -> None:
    """
    Convergence matrix by pair and feature family.

    description:
        Creates an A→B by family matrix of average Conv(t) values and renders it
        with a diverging, color-blind-safe colormap. Missing values are shown as light
        gray with hatch marks so sparse families aren’t invisible.
        If many pairs exist, the plot keeps those with the widest family coverage to stay legible.

    Params:
        pf: Per-(A,B,family) table with 'a_speaker','b_speaker','family','conv'
            and (optionally) 'n_triggers' to filter low-evidence cells.
        out_png: Output path for the PNG; a CSV pivot is also written.
        title: Chart title (e.g., "Convergence by pair and feature").
        min_support: Minimum number of A-triggers to keep a (pair,family) cell.
        max_edges: Max number of pairs (rows) to show; picks those with most non-missing families.

    Returns:
        None. Writes PNG and CSV.
    """
    if pf is None or pf.empty:
        return

    df = pf.copy()
    if "n_triggers" in df.columns:
        df = df[df["n_triggers"] >= min_support]
    if df.empty:
        return

    df["pair"] = df["a_speaker"].astype(str) + "→" + df["b_speaker"].astype(str)
    pivot = (df.pivot_table(index="pair", columns="family", values="conv", aggfunc="mean")
               .dropna(how="all"))
    if pivot.empty:
        return

    if len(pivot) > max_edges:
        coverage = pivot.notna().sum(axis=1).sort_values(ascending=False)
        pivot = pivot.loc[coverage.head(max_edges).index]

    M = pivot.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if M.size else 0.0
    vlim = vmax if vmax > 0 else 0.1

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, max(4.8, 0.40 * len(pivot))))
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)

    # grid lines for readability
    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # hatch missing cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.6, alpha=0.6))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Convergence Conv(t)")
    ax.set_title(title)
    ax.set_xlabel("Feature family")
    ax.set_ylabel("Speaker pair (A→B)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    pivot.to_csv(out_png.with_suffix(".csv"), encoding="utf-8-sig")

def plot_adaptability_distribution(pf: pd.DataFrame, out_png: Path, title: str) -> None:
    """
    Distribution of initiators’ adaptability.

    description:
        For each initiator A, computes their average Conv(t) across all partners and families,
        then plots a histogram of these per-speaker means. A vertical dashed line at 0 reflects
        the “no adaptation” baseline.
        Saves a CSV with the per-speaker averages.

    Params:
        pf: Per-(A,B,family) table with 'a_speaker' and 'conv'.
        out_png: Output PNG path; a CSV with the computed per-speaker values is also written.
        title: Chart title (e.g., "Distribution of speaker adaptability (A)").

    Returns:
        None. Writes PNG and CSV.
    """
    if pf is None or pf.empty or not {"a_speaker", "conv"}.issubset(pf.columns):
        return

    g = (pf.groupby("a_speaker", dropna=False)
           .agg(adapt_mean=("conv", "mean"), n=("conv", "size"))
           .reset_index()
           .sort_values("adapt_mean", ascending=False))
    if g.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.4, 5.0))
    plt.hist(g["adapt_mean"], bins=max(10, int(np.sqrt(len(g)))), alpha=0.9,
             color=plt.get_cmap("cividis")(0.6), edgecolor=OI_BLACK, linewidth=0.5)
    plt.axvline(0, color=OI_BLACK, linewidth=1.2, linestyle="--", alpha=0.8)
    plt.xlabel("Initiator’s mean Conv(t) across partners and families")
    plt.ylabel("Number of speakers (A)")
    plt.title(title)
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()

# =========================
# Gender & importance effects
# =========================
def plot_gender_levels(pf: pd.DataFrame, out_png: Path, who: str = "responder",
                       n_perm: int = 2000) -> None:
    """
    Convergence by gender and feature (+ two simple difference charts).

    description:
        Makes THREE charts:
          1) Grouped bars: average Conv(t) for each feature family, split by gender.
          2) Female − Male (F−M): positive bar => women show higher convergence.
          3) Male − Female (M−F): positive bar => men show higher convergence.

        Also saves two CSVs:
          - <out_png>.csv           : table used in the grouped bars (per family, mean by gender)
          - <out_png>.diffs.csv     : per-family differences (F−M and M−F) + a simple permutation p-value.

        Notes:
          • Use who="responder" to compare B’s gender (who replies), or who="initiator" for A’s gender.
          • Expected gender codes are 'F' and 'M'. Other/empty codes are ignored.
          • Conv(t) is the convergence score per family. We average it within each family.

    Params:
        pf:  Per-(A,B,family) table with columns: 'family', 'conv', and a gender column:
             'b_gender' if who == 'responder', else 'a_gender'.
        out_png:  Base PNG path for the grouped bars. Two extra files are saved:
                  *_diff_F_minus_M.png and *_diff_M_minus_F.png.
        who:  'responder' (default) or 'initiator'.
        n_perm:  Number of label permutations for the simple two-sided p-value (default 2000).

    Returns:
        None. Writes 3 PNGs and 2 CSVs.
    """
    # --- guard rails ---
    if pf is None or pf.empty:
        return

    side_col = "b_gender" if who == "responder" else "a_gender"
    if side_col not in pf.columns:
        return

    # Keep only rows with an F/M label
    df = pf.dropna(subset=[side_col]).copy()
    df[side_col] = df[side_col].astype(str).str.upper().str[0]
    df = df[df[side_col].isin(["F", "M"])]
    if df.empty:
        return

    # --- grouped means by family and gender ---
    g = (df.groupby(["family", side_col])["conv"].mean().unstack(side_col))
    if g is None or g.empty:
        return

    # Ensure columns order (F then M) if present
    cols = [c for c in ["F", "M"] if c in g.columns]
    if not cols:
        return

    # Sort families by overall mean for readability
    g_plot = g[cols].fillna(0.0).reindex(index=g.mean(axis=1).sort_values(ascending=False).index)

    # === Chart 1: grouped bars by gender ===
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.2, max(3.9, 0.44 * len(g_plot))))
    x = np.arange(len(g_plot.index))
    w = 0.4 if len(cols) == 2 else 0.6

    # Colors: consistent, color-blind friendly (Okabe–Ito palette expected in scope)
    gender_labels = {"F": "Female", "M": "Male"}
    gender_colors = {"F": OI_PURPLE, "M": OI_ORANGE}

    for i, c in enumerate(cols):
        plt.bar(
            x + i * w,
            g_plot[c].to_numpy(),
            width=w,
            label=gender_labels.get(c, c),
            color=gender_colors.get(c, plt.get_cmap("cividis")(0.6)),
            edgecolor=OI_BLACK,
            linewidth=0.6,
            alpha=0.95,
        )

    plt.xticks(x + w * (len(cols) - 1) / 2, g_plot.index, rotation=25, ha="right")
    plt.ylabel("Average convergence, Conv(t)")
    if who == "responder":
        plt.title("Gender difference in adaptation when replying (Female − Male)")
    else:
        plt.title("How others adapt when replying to men vs. women (by feature family)")
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.legend(title="Gender", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()

    # Save CSV for grouped bars
    g_plot.reset_index().rename(columns={"index": "family"}).to_csv(
        out_png.with_suffix(".csv"), index=False, encoding="utf-8-sig"
    )

    # === Differences per family ===
    fam_order = g_plot.index.tolist()
    diffs_FM = []
    pvals = []

    rng = np.random.default_rng(42)

    for fam in fam_order:
        fam_rows = df[df["family"] == fam]
        # observed difference: Female − Male
        f_vals = fam_rows[fam_rows[side_col] == "F"]["conv"].to_numpy()
        m_vals = fam_rows[fam_rows[side_col] == "M"]["conv"].to_numpy()
        if f_vals.size == 0 and m_vals.size == 0:
            diffs_FM.append(np.nan)
            pvals.append(np.nan)
            continue

        d_obs = float(np.nanmean(f_vals) - np.nanmean(m_vals))
        diffs_FM.append(d_obs)

        # permutation test (simple label exchange within family)
        labels = fam_rows[side_col].to_numpy()
        vals = fam_rows["conv"].to_numpy()
        mask = np.isin(labels, ["F", "M"]) & np.isfinite(vals)
        labels = labels[mask]
        vals = vals[mask]

        if labels.size == 0 or (labels == "F").sum() == 0 or (labels == "M").sum() == 0:
            pvals.append(np.nan)
        else:
            nF = int((labels == "F").sum())
            count = 0
            for _ in range(n_perm):
                perm = rng.permutation(labels.size)
                f_idx = perm[:nF]
                m_idx = perm[nF:]
                diff_perm = vals[f_idx].mean() - vals[m_idx].mean()
                if abs(diff_perm) >= abs(d_obs):
                    count += 1
            pvals.append((count + 1) / (n_perm + 1))

    # Build differences table (include both directions for clarity)
    diff_df = pd.DataFrame({
        "family": fam_order,
        "diff_F_minus_M": diffs_FM,
        "diff_M_minus_F": [-d if np.isfinite(d) else np.nan for d in diffs_FM],
        "perm_p_two_sided": pvals,
    })

    # === Chart 2: Female − Male ===
    diff_out_png_FM = out_png.with_name(out_png.stem + "_diff_F_minus_M.png")
    plt.figure(figsize=(10.2, max(3.9, 0.44 * len(diff_df))))
    plt.barh(
        diff_df["family"][::-1],
        diff_df["diff_F_minus_M"].to_numpy()[::-1],
        alpha=0.95,
        color=OI_GREEN,
        edgecolor=OI_BLACK,
        linewidth=0.6,
    )
    plt.axvline(0, color=OI_BLACK, linewidth=1.2, linestyle="--", alpha=0.8)
    plt.xlabel("Difference in convergence (Female − Male)")
    plt.title("Gender difference in convergence (F−M): positive = women adapt more")
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(diff_out_png_FM, bbox_inches="tight", dpi=220)
    plt.close()

    # === Chart 3: Male − Female ===
    diff_out_png_MF = out_png.with_name(out_png.stem + "_diff_M_minus_F.png")
    plt.figure(figsize=(10.2, max(3.9, 0.44 * len(diff_df))))
    plt.barh(
        diff_df["family"][::-1],
        diff_df["diff_M_minus_F"].to_numpy()[::-1],
        alpha=0.95,
        color=OI_BLUE,
        edgecolor=OI_BLACK,
        linewidth=0.6,
    )
    plt.axvline(0, color=OI_BLACK, linewidth=1.2, linestyle="--", alpha=0.8)
    plt.xlabel("Difference in convergence (Male − Female)")
    plt.title("Gender difference in convergence (M−F): positive = men adapt more")
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(diff_out_png_MF, bbox_inches="tight", dpi=220)
    plt.close()

    # Save differences CSV
    diff_df.to_csv(out_png.with_suffix(".diffs.csv"), index=False, encoding="utf-8-sig")



def plot_importance_levels(pf: pd.DataFrame, out_png: Path, who: str = "initiator") -> None:
    """
    Convergence by character importance (major / intermediate / minor).

    description:
        Shows mean Conv(t) by feature family across importance groups.
        For who='initiator', groups refer to A (how much others adjust to them).
        For who='responder', groups are B (who adapts when replying).
        Labels are normalized to 'major'/'intermediate'/'minor'.

    Params:
        pf: Per-(A,B,family) table with 'family','conv' and an importance column:
            'a_category' if who='initiator', else 'b_category'.
        out_png: Output PNG path; a CSV with plotted values is also written.
        who: 'initiator' or 'responder'.

    Returns:
        None. Writes PNG and CSV.
    """
    if pf is None or pf.empty:
        return

    cat_col = "a_category" if who == "initiator" else "b_category"
    if cat_col not in pf.columns:
        return

    df = pf.copy()
    df[cat_col] = (df[cat_col].astype(str).str.strip().str.lower()
                   .replace({"maj": "major", "main": "major", "primary": "major",
                             "int": "intermediate", "mid": "intermediate",
                             "min": "minor"}))
    levels = ["major", "intermediate", "minor"]
    df = df[df[cat_col].isin(levels)]
    if df.empty:
        return

    g = (df.groupby(["family", cat_col])["conv"].mean().unstack(cat_col))
    if g is None or g.empty:
        return

    present = [c for c in levels if c in g.columns]
    if not present:
        return

    g = g[present].fillna(0.0)
    g = g.reindex(index=g.mean(axis=1).sort_values(ascending=False).index)

    colors = {"major": OI_BLUE, "intermediate": OI_ORANGE, "minor": OI_GREEN}

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.2, max(3.9, 0.44 * len(g))))
    x = np.arange(len(g.index))
    w = 0.28
    for i, c in enumerate(present):
        plt.bar(x + i*w, g[c].to_numpy(), width=w, label=c.capitalize(),
                color=colors.get(c, plt.get_cmap("cividis")(0.6)),
                edgecolor=OI_BLACK, linewidth=0.6, alpha=0.95)

    plt.xticks(x + w*(len(present)-1)/2, g.index, rotation=25, ha="right")
    plt.ylabel("Average convergence  Conv(t)")
    if who == "initiator":
        plt.title("How others adjust by speaker importance (A)")
    else:
        plt.title("Who adapts when replying by importance (B)")
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.legend(title="Importance", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    g.reset_index().to_csv(out_png.with_suffix(".csv"), index=False, encoding="utf-8-sig")


# =========================
# Condition comparisons
# =========================
def plot_condition_comparison(pf_adj: Optional[pd.DataFrame],
                              pf_non: Optional[pd.DataFrame],
                              pf_rand: Optional[pd.DataFrame],
                              out_png: Path,
                              title: str) -> None:
    """
    Family means across experimental conditions.

    description:
        Compares per-family mean Conv(t) across available conditions:
        Adjacent, Nonadjacent, and Randomized. Bars are grouped by family. Useful
        to spot whether adjacency (true dialogue order) produces stronger convergence
        than baselines.

    Params:
        pf_adj: Per-(A,B,family) table for the “adjacent” condition, or None.
        pf_non: Per-(A,B,family) for “nonadjacent” baseline, or None.
        pf_rand: Per-(A,B,family) for “randomized” baseline, or None.
        out_png: Output PNG path; a CSV is also written next to it.
        title: Chart title (e.g., "Family means across conditions").

    Returns:
        None. Writes PNG and CSV.
    """
    frames = []
    if pf_adj is not None and not pf_adj.empty:
        frames.append(pf_adj.groupby("family")["conv"].mean().rename("Adjacent").to_frame())
    if pf_non is not None and not pf_non.empty:
        frames.append(pf_non.groupby("family")["conv"].mean().rename("Nonadjacent").to_frame())
    if pf_rand is not None and not pf_rand.empty:
        frames.append(pf_rand.groupby("family")["conv"].mean().rename("Randomized").to_frame())
    if not frames:
        return

    M = pd.concat(frames, axis=1).dropna(how="all")
    if M.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.6, max(3.9, 0.44 * len(M))))
    xs = np.arange(len(M.index))
    w = 0.26
    palette = {"Adjacent": OI_BLUE, "Nonadjacent": OI_ORANGE, "Randomized": OI_GREEN}
    cols = [c for c in ["Adjacent", "Nonadjacent", "Randomized"] if c in M.columns]

    for i, c in enumerate(cols):
        plt.bar(xs + i*w, M[c].to_numpy(), width=w, label=c, alpha=0.95,
                color=palette.get(c, plt.get_cmap("cividis")(0.6)),
                edgecolor=OI_BLACK, linewidth=0.6)

    plt.xticks(xs + w, M.index, rotation=30, ha="right")
    plt.ylabel("Mean Conv(t)")
    plt.title(title)
    plt.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    M.to_csv(out_png.with_suffix(".csv"), encoding="utf-8-sig")


# =========================
# Global profile with uncertainty
# =========================
def plot_global_profile_with_ci(pf: pd.DataFrame, out_png: Path, title: str, alpha: float = 0.05) -> None:
    """
    Family means with bootstrap confidence intervals.

    description:
        For each family, estimates the mean Conv(t) and a (1−alpha)×100% bootstrap
        confidence interval by resampling per-row values (with replacement).
        Error bars show the bootstrap percentile CI; bars show the family mean.

        Note: A confidence interval (CI) reflects the uncertainty in the sample mean.
        A 95% CI means that if we repeated this sampling many times, ~95% of such
        intervals would contain the true mean.


    Params:
        pf: Per-(A,B,family) convergence table with 'family' and 'conv'.
        out_png: Output PNG path; a CSV of mean/CI per family is also written.
        title: Chart title (e.g., "Family means with 95% CI").
        alpha: Significance level for the CI (default 0.05 → 95% CI).

    Returns:
        None. Writes PNG and CSV.
    """
    if pf is None or pf.empty or "family" not in pf.columns or "conv" not in pf.columns:
        return

    rows = []
    for fam, g in pf.groupby("family"):
        x = pd.to_numeric(g["conv"], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        rng = np.random.default_rng(42)
        n_boot = 2000
        boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
        mu = float(x.mean())
        lo = float(np.percentile(boots, 100 * (alpha / 2)))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        rows.append({"family": fam, "mean": mu, "ci_lo": lo, "ci_hi": hi, "n": len(x)})

    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    if df.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.6, max(4.2, 0.44 * len(df))))
    y = np.arange(len(df))
    bar_color = plt.get_cmap("cividis")(0.55)

    plt.barh(df["family"][::-1], df["mean"][::-1], alpha=0.95,
             color=bar_color, edgecolor=OI_BLACK, linewidth=0.6)

    errs_lo = (df["mean"] - df["ci_lo"]).to_numpy()[::-1]
    errs_hi = (df["ci_hi"] - df["mean"]).to_numpy()[::-1]
    plt.errorbar(df["mean"][::-1], y, xerr=[errs_lo, errs_hi],
                 fmt="none", ecolor=OI_BLACK, elinewidth=1.3, capsize=3)

    plt.xlabel("Mean convergence  Conv(t)")
    plt.ylabel("Feature family")
    plt.title(title)
    plt.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    df.to_csv(out_png.with_suffix(".csv"), index=False, encoding="utf-8-sig")


# =========================
# Two-mode edge heatmaps
# =========================
def plot_edge_gender_heatmap(edges: pd.DataFrame, out_png: Path, title: str) -> None:
    """
    Mean A→B edge convergence by gender pair.

    description:
        Builds a 2×2 table of average edge convergence by (A gender, B gender),
        using the best available weight column: 'mean_conv_w', then 'weight_work',
        then 'mean_conv'. Colors center at 0 (diverging palette). Missing cells are
        light gray with hatching so “no data” isn’t just white.

    Params:
        edges: Edge summary with 'a_gender','b_gender' and one of the weight columns above.
               Gender codes are normalized to 'F'/'M' by first letter.
        out_png: Output PNG path; a CSV table is written next to it.
        title: Chart title (e.g., "Mean edge convergence by gender (A→B)").

    Returns:
        None. Writes PNG and CSV.
    """
    if edges is None or edges.empty:
        return
    if not {"a_gender", "b_gender"}.issubset(edges.columns):
        return

    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    df = edges.copy()
    df["a_gender"] = df["a_gender"].astype(str).str.upper().str[0]
    df["b_gender"] = df["b_gender"].astype(str).str.upper().str[0]
    df = df[df["a_gender"].isin(["F", "M"]) & df["b_gender"].isin(["F", "M"])]
    if df.empty:
        # still render a 2×2 with all missing so the user sees the structure
        pivot = pd.DataFrame(index=["F", "M"], columns=["F", "M"], dtype=float)
    else:
        pivot = df.pivot_table(index="a_gender", columns="b_gender",
                               values=weight_col, aggfunc="mean")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    M = pivot.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.0
    vlim = vmax if vmax > 0 else 0.1
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)

    # grid & hatch missing
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = M[i, j]
            if not np.isfinite(val):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.6, alpha=0.6))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean edge Conv (weighted when available)")
    ax.set_title(title)
    ax.set_xlabel("Responder gender (B)")
    ax.set_ylabel("Initiator gender (A)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    pivot.to_csv(out_png.with_suffix(".csv"), encoding="utf-8-sig")


def plot_edge_category_heatmap(edges: pd.DataFrame, out_png: Path, title: str) -> None:
    """
    Mean A→B edge convergence by importance pair.

    description:
        Builds a 3×3 table of average edge convergence by (A importance, B importance),
        where importance ∈ {major, intermediate, minor}. Colors center at 0.
        Missing cells are light gray with hatching to avoid “white = nothing” confusion.

    Params:
        edges: Edge summary with 'a_category','b_category' and a weight column:
               'mean_conv_w' | 'weight_work' | 'mean_conv'.
        out_png: Output PNG path; a CSV table is written next to it.
        title: Chart title (e.g., "Mean edge convergence by importance (A→B)").

    Returns:
        None. Writes PNG and CSV.
    """
    if edges is None or edges.empty:
        return
    if not {"a_category", "b_category"}.issubset(edges.columns):
        return

    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    df = edges.copy()
    norm_names = {"maj": "major", "main": "major", "primary": "major",
                  "int": "intermediate", "mid": "intermediate",
                  "min": "minor"}
    for c in ["a_category", "b_category"]:
        df[c] = (df[c].astype(str).str.strip().str.lower().replace(norm_names))
    levels = ["major", "intermediate", "minor"]
    df = df[df["a_category"].isin(levels) & df["b_category"].isin(levels)]

    if df.empty:
        pivot = pd.DataFrame(index=levels, columns=levels, dtype=float)
    else:
        pivot = (df.pivot_table(index="a_category", columns="b_category", values=weight_col, aggfunc="mean")
                   .reindex(index=levels, columns=levels))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    M = pivot.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.0
    vlim = vmax if vmax > 0 else 0.1
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)), [c.capitalize() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [r.capitalize() for r in pivot.index])

    # grid & hatch missing
    ax.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(levels)):
        for j in range(len(levels)):
            val = M[i, j]
            if not np.isfinite(val):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.6, alpha=0.6))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean edge Conv (weighted when available)")
    ax.set_title(title)
    ax.set_xlabel("Responder importance (B)")
    ax.set_ylabel("Initiator importance (A)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close()
    pivot.to_csv(out_png.with_suffix(".csv"), encoding="utf-8-sig")


# =========================
# Runner
# =========================
def run_visuals_for_novel(in_dir: Path, out_root: Path, mode: str,
                          min_support: int = 10, max_edges: int = 50, top_k: int = 20) -> None:
    """
    Generate all figures for one analysis folder.

    description:
        Reads the convergence artifacts for a single analysis run (in_dir), and
        renders a standard pack of figures into out_root.
    Params:
        in_dir: Folder containing convergence CSVs, typically:
                  conv_by_pair_feature_{mode}.csv
                  conv_edge_summary_{mode}.csv
                  conv_by_pair_feature_{mode}_nonadjacent.csv (optional)
                  conv_by_pair_feature_{mode}_randomized.csv  (optional)
        out_root: Folder where figures (and companion CSVs) will be written.
        mode: "liwc" or "spacy". Only affects input file names.
        min_support: Minimum A-trigger count for including (pair,family) cells in the heatmap.
        max_edges: Max number of pairs to show in the heatmap (keeps rows with most coverage).
        top_k: Number of A→B pairs to show in the “top edges” bar chart.

    Returns:
        None. Writes all PNGs (and small CSVs) to out_root.
    """
    print(f"[viz] Running visualisation for {in_dir} ({mode})")

    pf_path = in_dir / f"conv_by_pair_feature_{mode}.csv"
    edge_path = in_dir / f"conv_edge_summary_{mode}.csv"
    pf_non_path = in_dir / f"conv_by_pair_feature_{mode}_nonadjacent.csv"
    pf_rand_path = in_dir / f"conv_by_pair_feature_{mode}_randomized.csv"

    if not pf_path.exists() or not edge_path.exists():
        print("[viz] Missing inputs, skipping.")
        return

    pf = pd.read_csv(pf_path, encoding="utf-8-sig")
    edges = pd.read_csv(edge_path, encoding="utf-8-sig")
    pf_non = pd.read_csv(pf_non_path, encoding="utf-8-sig") if pf_non_path.exists() else None
    pf_rand = pd.read_csv(pf_rand_path, encoding="utf-8-sig") if pf_rand_path.exists() else None

    # 1) Family overview
    plot_family_conv_bars(pf, out_root / "family_means.png", "Average convergence by feature family")
    plot_heatmap(pf, out_root / "heatmap.png", "Convergence by pair and feature",
                 min_support=min_support, max_edges=max_edges)

    # 2) Edge distributions & leaders
    plot_edge_distribution(edges, out_root / "edge_distribution.png", "Distribution of A→B edge convergence")
    plot_top_edges(edges, out_root / "top_edges.png", "Top converging pairs (A→B)", top_k=top_k)
    plot_adaptability_distribution(pf, out_root / "adaptability_distribution.png", "Distribution of speaker adaptability (A)")

    # 3) Gender effects
    plot_gender_levels(pf, out_root / "gender_levels_responder.png", who="responder")
    plot_gender_levels(pf, out_root / "gender_levels_initiator.png", who="initiator")

    # 4) Importance effects
    plot_importance_levels(pf, out_root / "importance_levels_initiator.png", who="initiator")
    plot_importance_levels(pf, out_root / "importance_levels_responder.png", who="responder")

    # 5) Condition comparison (if baselines exist)
    plot_condition_comparison(pf, pf_non, pf_rand, out_root / "adj_non_rand.png",
                              "Family means across conditions")

    # 6) Global profile with uncertainty
    plot_global_profile_with_ci(pf, out_root / "family_means_ci.png", "Family means with confidence intervals")

    # 7) Two-mode edge heatmaps
    plot_edge_gender_heatmap(edges, out_root / "edge_heatmap_gender.png", "Mean edge convergence by gender (A→B)")
    plot_edge_category_heatmap(edges, out_root / "edge_heatmap_importance.png", "Mean edge convergence by importance (A→B)")

    print(f"[viz] Visuals written to {out_root}")
