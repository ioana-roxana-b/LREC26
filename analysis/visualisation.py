from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import ttest_1samp

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
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# =========================
# CSV path helpers
# =========================
def csv_target(out_png: Path, csv_dir: Optional[Path]) -> Path:
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        return (csv_dir / out_png.name).with_suffix(".csv")
    return out_png.with_suffix(".csv")


def csv_named(out_png: Path, csv_dir: Optional[Path], new_name: str) -> Path:
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
        return csv_dir / new_name
    return out_png.with_name(new_name)


# =========================
# Family- and edge-level plots
# =========================
def plot_family_conv_bars(
    pf: pd.DataFrame,
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty or not {"family", "conv"}.issubset(pf.columns):
        return

    rows = []
    for fam, g in pf.groupby("family", dropna=False):
        vals = pd.to_numeric(g["conv"], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        rows.append({"family": fam, "mean_conv": float(vals.mean()), "n": int(vals.size)})

    g = pd.DataFrame(rows).sort_values("mean_conv", ascending=False).reset_index(drop=True)
    if g.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    # more vertical room per family
    plt.figure(figsize=(11.5, max(5.0, 0.60 * len(g))))
    bar_color = plt.get_cmap("cividis")(0.55)
    plt.barh(g["family"][::-1], g["mean_conv"][::-1], alpha=0.95,
             color=bar_color, edgecolor=OI_BLACK, linewidth=0.7)

    plt.xlabel("Mean convergence Conv(t) (higher = more matching)")
    plt.ylabel("Feature family")
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="x", linestyle=":", linewidth=1.0, alpha=0.4, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    if csv_dir is not None or True:
        out_csv = csv_target(out_png, csv_dir)
        g = g.assign(novel=novel_title if novel_title else "")
        g.to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_edge_distribution(
    edges: pd.DataFrame,
    out_png: Path,
    title: Optional[str] = None,
    novel_title: Optional[str] = None,
) -> None:
    if edges is None or edges.empty:
        return

    weight_col = next((c for c in ["mean_conv_w", "weight_work", "mean_conv"] if c in edges.columns), None)
    if weight_col is None:
        return

    vals = pd.to_numeric(edges[weight_col], errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.5, 6.2))
    bins = max(14, int(np.sqrt(len(vals))))

    plt.hist(vals, bins=bins, alpha=0.9, color=plt.get_cmap("cividis")(0.6),
             edgecolor=OI_BLACK, linewidth=0.7)

    plt.axvline(0, color=OI_BLACK, linewidth=1.4, linestyle="--", alpha=0.9, label="No adaptation (0)")
    mean_v = float(np.mean(vals))
    med_v  = float(np.median(vals))
    plt.axvline(mean_v, color=OI_BLUE,   linestyle="-",  linewidth=1.8, label=f"Mean = {mean_v:.3f}")
    plt.axvline(med_v,  color=OI_ORANGE, linestyle=":",  linewidth=2.0, label=f"Median = {med_v:.3f}")

    plt.xlabel("Per-pair convergence score (Conv; > 0 = B adapts to A, < 0 = diverges)")
    plt.ylabel("Number of A→B pairs")
    ttl = title or "How strongly each pair (A→B) adapts overall"
    plt.title(f"{ttl}" + (f"\n({novel_title})" if novel_title else ""))
    plt.legend(frameon=False, loc="upper left")
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()


def plot_top_edges(
    edges: pd.DataFrame,
    out_png: Path,
    title: str,
    top_k: int = 20,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
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
    plt.figure(figsize=(12.0, max(5.0, 0.60 * len(top))))
    bar_color = plt.get_cmap("cividis")(0.55)
    plt.barh(labels[::-1], top["__w__"].to_numpy()[::-1], alpha=0.95,
             color=bar_color, edgecolor=OI_BLACK, linewidth=0.7)

    plt.xlabel("Mean edge convergence")
    plt.ylabel("Speaker pair")
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="x", linestyle=":", linewidth=1.0, alpha=0.4, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    top = top.drop(columns=["__w__"]).assign(novel=novel_title if novel_title else "")
    top.to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_heatmap(
    pf: pd.DataFrame,
    out_png: Path,
    title: str,
    min_support: int = 10,
    max_edges: int = 50,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty:
        return

    df = pf.copy()
    if "n_triggers" in df.columns:
        df = df[pd.to_numeric(df["n_triggers"], errors="coerce") >= min_support]
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
    # adaptive sizing: more room for rows/cols
    fig_w = max(13.0, 0.45 * pivot.shape[1] + 6.5)
    fig_h = max(6.5, 0.60 * len(pivot))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=40, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)

    ax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.6, alpha=0.6))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Convergence Conv(t)")
    ax.set_title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    ax.set_xlabel("Feature family")
    ax.set_ylabel("Speaker pair (A→B)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    pivot = pivot.assign(_novel=novel_title if novel_title else "")
    pivot.to_csv(out_csv, encoding="utf-8-sig")


def plot_adaptability_distribution(
    pf: pd.DataFrame,
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty or not {"a_speaker", "conv"}.issubset(pf.columns):
        return

    g = (pf.groupby("a_speaker", dropna=False)
           .agg(adapt_mean=("conv", "mean"), n=("conv", "size"))
           .reset_index()
           .sort_values("adapt_mean", ascending=False))
    if g.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.2, 6.0))
    plt.hist(g["adapt_mean"], bins=max(12, int(np.sqrt(len(g)))), alpha=0.9,
             color=plt.get_cmap("cividis")(0.6), edgecolor=OI_BLACK, linewidth=0.7)
    plt.axvline(0, color=OI_BLACK, linewidth=1.4, linestyle="--", alpha=0.85)
    plt.xlabel("Initiator’s mean Conv(t) across partners and families")
    plt.ylabel("Number of speakers (A)")
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    g = g.assign(novel=novel_title if novel_title else "")
    g.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================
# Gender & importance effects
# =========================
def plot_gender_levels(
    pf: pd.DataFrame,
    out_png: Path,
    who: str = "responder",
    n_perm: int = 2000,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty:
        return

    side_col = "b_gender" if who == "responder" else "a_gender"
    if side_col not in pf.columns:
        return

    df = pf.dropna(subset=[side_col]).copy()
    df[side_col] = df[side_col].astype(str).str.upper().str[0]
    df = df[df[side_col].isin(["F", "M"])]
    if df.empty:
        return

    g = (df.groupby(["family", side_col])["conv"].mean().unstack(side_col))
    if g is None or g.empty:
        return

    cols = [c for c in ["F", "M"] if c in g.columns]
    if not cols:
        return

    g_plot = g[cols].fillna(0.0).reindex(index=g.mean(axis=1).sort_values(ascending=False).index)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11.8, max(5.0, 0.60 * len(g_plot))))
    x = np.arange(len(g_plot.index))
    w = 0.42 if len(cols) == 2 else 0.6

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
            linewidth=0.7,
            alpha=0.95,
        )

    plt.xticks(x + w * (len(cols) - 1) / 2, g_plot.index, rotation=25, ha="right")
    plt.ylabel("Average convergence, Conv(t)")
    if who == "responder":
        base = "How much women and men adapt when replying (responder's gender effect)"
    else:
        base = "How much others adapt when replying to women vs. men (initiator's gender effect)"
    plt.title(base + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(title="Gender", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    g_out = g_plot.reset_index().rename(columns={"index": "family"})
    g_out = g_out.assign(novel=novel_title if novel_title else "")
    g_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Differences + simple permutation p
    fam_order = g_plot.index.tolist()
    diffs_FM = []
    pvals = []
    rng = np.random.default_rng(42)

    for fam in fam_order:
        fam_rows = df[df["family"] == fam]
        f_vals = fam_rows[fam_rows[side_col] == "F"]["conv"].to_numpy()
        m_vals = fam_rows[fam_rows[side_col] == "M"]["conv"].to_numpy()
        if f_vals.size == 0 and m_vals.size == 0:
            diffs_FM.append(np.nan)
            pvals.append(np.nan)
            continue

        d_obs = float(np.nanmean(f_vals) - np.nanmean(m_vals))
        diffs_FM.append(d_obs)

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

    diff_df = pd.DataFrame({
        "family": fam_order,
        "diff_F_minus_M": diffs_FM,
        "diff_M_minus_F": [-d if np.isfinite(d) else np.nan for d in diffs_FM],
        "perm_p_two_sided": pvals,
        "novel": novel_title if novel_title else "",
    })

    diff_out_png_FM = out_png.with_name(out_png.stem + "_diff_F_minus_M.png")
    plt.figure(figsize=(11.8, max(5.0, 0.60 * len(diff_df))))
    plt.barh(diff_df["family"][::-1], diff_df["diff_F_minus_M"].to_numpy()[::-1],
             alpha=0.95, color=OI_GREEN, edgecolor=OI_BLACK, linewidth=0.7)
    plt.axvline(0, color=OI_BLACK, linewidth=1.3, linestyle="--", alpha=0.85)
    plt.xlabel("Difference in convergence (Female − Male)")
    plt.title("Gender difference in convergence (F−M): positive = women adapt more"
              + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(diff_out_png_FM, bbox_inches="tight", dpi=240)
    plt.close()

    diff_out_png_MF = out_png.with_name(out_png.stem + "_diff_M_minus_F.png")
    plt.figure(figsize=(11.8, max(5.0, 0.60 * len(diff_df))))
    plt.barh(diff_df["family"][::-1], diff_df["diff_M_minus_F"].to_numpy()[::-1],
             alpha=0.95, color=OI_BLUE, edgecolor=OI_BLACK, linewidth=0.7)
    plt.axvline(0, color=OI_BLACK, linewidth=1.3, linestyle="--", alpha=0.85)
    plt.xlabel("Difference in convergence (Male − Female)")
    plt.title("Gender difference in convergence (M−F): positive = men adapt more"
              + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(diff_out_png_MF, bbox_inches="tight", dpi=240)
    plt.close()

    diff_csv = csv_named(out_png, csv_dir, out_png.with_suffix(".diffs.csv").name)
    diff_df.to_csv(diff_csv, index=False, encoding="utf-8-sig")


def plot_importance_levels(
    pf: pd.DataFrame,
    out_png: Path,
    who: str = "initiator",
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
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
    plt.figure(figsize=(11.8, max(5.0, 0.60 * len(g))))
    x = np.arange(len(g.index))
    w = 0.30
    for i, c in enumerate(present):
        plt.bar(x + i*w, g[c].to_numpy(), width=w, label=c.capitalize(),
                color=colors.get(c, plt.get_cmap("cividis")(0.6)),
                edgecolor=OI_BLACK, linewidth=0.7, alpha=0.95)

    plt.xticks(x + w*(len(present)-1)/2, g.index, rotation=25, ha="right")
    plt.ylabel("Average convergence  Conv(t)")
    if who == "initiator":
        base = "For initiator (A): how much others adapt to them depending on A’s importance"
    else:
        base = "For responder (B): how much B adapts depending on B’s importance"
    plt.title(base + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(title="Importance", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    g_out = g.reset_index().assign(novel=novel_title if novel_title else "")
    g_out.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================
# Two-mode edge heatmaps (means + counts)
# =========================
def plot_edge_gender_heatmap(
    edges: pd.DataFrame,
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
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
    levels = ["F", "M"]

    mean_p = pd.DataFrame(index=levels, columns=levels, dtype=float)
    n_p    = pd.DataFrame(index=levels, columns=levels, dtype=int)

    for a in levels:
        for b in levels:
            vals = pd.to_numeric(df.loc[(df["a_gender"] == a) & (df["b_gender"] == b), weight_col],
                                 errors="coerce").dropna().to_numpy()
            n_p.loc[a, b] = int(vals.size)
            mean_p.loc[a, b] = float(vals.mean()) if vals.size else np.nan

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    M = mean_p.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.0
    vlim = vmax if vmax > 0 else 0.1
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(levels)), levels)

    ax.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, ia in enumerate(levels):
        for j, jb in enumerate(levels):
            mu = mean_p.loc[ia, jb]
            n  = n_p.loc[ia, jb]
            if not np.isfinite(mu):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.7, alpha=0.6))
            else:
                ax.text(j, i, f"{mu:+.3f}\n(n={int(n)})", ha="center", va="center", fontsize=9, color=OI_BLACK)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean edge Conv (weighted when available)")
    ax.set_title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    ax.set_xlabel("Responder gender (B)")
    ax.set_ylabel("Initiator gender (A)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_rows = []
    for a in levels:
        for b in levels:
            out_rows.append({"Initiator gender (A)": a,
                             "Responder gender (B)": b,
                             "mean": mean_p.loc[a, b],
                             "n": n_p.loc[a, b],
                             "novel": novel_title if novel_title else ""})
    pd.DataFrame(out_rows).to_csv(csv_target(out_png, csv_dir), index=False, encoding="utf-8-sig")


def plot_edge_category_heatmap(
    edges: pd.DataFrame,
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
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

    mean_p = pd.DataFrame(index=levels, columns=levels, dtype=float)
    n_p    = pd.DataFrame(index=levels, columns=levels, dtype=int)

    for a in levels:
        for b in levels:
            vals = pd.to_numeric(df.loc[(df["a_category"] == a) & (df["b_category"] == b), weight_col],
                                 errors="coerce").dropna().to_numpy()
            n_p.loc[a, b] = int(vals.size)
            mean_p.loc[a, b] = float(vals.mean()) if vals.size else np.nan

    disp = [s.capitalize() for s in levels]
    mean_p.index = disp; mean_p.columns = disp
    n_p.index  = disp; n_p.columns  = disp

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    M = mean_p.to_numpy()
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.0
    vlim = vmax if vmax > 0 else 0.1
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(color="#D9D9D9")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(disp)), disp)
    ax.set_yticks(range(len(disp)), disp)
    ax.set_xticks(np.arange(-0.5, len(disp), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(disp), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, ia in enumerate(disp):
        for j, jb in enumerate(disp):
            mu = mean_p.loc[ia, jb]
            n  = n_p.loc[ia, jb]
            if not np.isfinite(mu):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch="///", edgecolor=OI_BLACK, linewidth=0.7, alpha=0.6))
            else:
                ax.text(j, i, f"{mu:+.3f}\n(n={int(n)})", ha="center", va="center", fontsize=9, color=OI_BLACK)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean edge Conv (weighted when available)")
    ax.set_title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    ax.set_xlabel("Responder importance (B)")
    ax.set_ylabel("Initiator importance (A)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    tall = []
    for a in disp:
        for b in disp:
            tall.append({
                "Initiator importance (A)": a,
                "Responder importance (B)": b,
                "mean": mean_p.loc[a, b],
                "n": n_p.loc[a, b],
                "novel": novel_title if novel_title else "",
            })
    pd.DataFrame(tall).to_csv(csv_target(out_png, csv_dir), index=False, encoding="utf-8-sig")


def plot_family_p1_p0(
    pf: pd.DataFrame,
    out_png: Path,
    title: str,
    alpha: float = 0.05,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if pf is None or pf.empty or not {"family", "p0", "p1"}.issubset(pf.columns):
        return

    rows = []
    for fam, g in pf.groupby("family", dropna=False):
        per_pair = pd.DataFrame({
            "p0": pd.to_numeric(g["p0"], errors="coerce"),
            "p1": pd.to_numeric(g["p1"], errors="coerce"),
        }).dropna()

        if per_pair.empty:
            continue

        p0 = per_pair["p0"].to_numpy()
        p1 = per_pair["p1"].to_numpy()
        diffs = p1 - p0

        mu0 = float(p0.mean())
        mu1 = float(p1.mean())

        try:
            t_stat, p_one = ttest_1samp(diffs, 0.0, alternative="greater")
        except TypeError:
            t_stat, p_two = ttest_1samp(diffs, 0.0)
            p_one = p_two / 2 if np.nanmean(diffs) > 0 else 1.0

        rows.append({
            "family": fam,
            "mean_p0": mu0,
            "mean_p1": mu1,
            "diff": mu1 - mu0,
            "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
            "p_val_ttest_one_sided": float(p_one) if np.isfinite(p_one) else np.nan,
            "significant_ttest": bool(np.isfinite(p_one) and p_one < alpha),
            "n_pairs": int(diffs.size),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    df = df.sort_values("diff", ascending=False).reset_index(drop=True)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12.0, max(5.2, 0.60 * len(df))))
    x = np.arange(len(df))
    w = 0.36

    c0 = OI_BLUE
    c1 = "#5FA7D9"

    plt.bar(x - w/2, df["mean_p0"], width=w, color=c0, edgecolor=OI_BLACK, linewidth=0.7, label="P(B uses t)")
    plt.bar(x + w/2, df["mean_p1"], width=w, color=c1, edgecolor=OI_BLACK, linewidth=0.7, label="P(B uses t | A uses t)")

    y_top = np.maximum(df["mean_p0"], df["mean_p1"]).to_numpy()
    for i, sig in enumerate(df["significant_ttest"].tolist()):
        if sig:
            plt.text(i, y_top[i] * 1.02, "*", ha="center", va="bottom", fontsize=12)

    plt.xticks(x, df["family"], rotation=25, ha="right")
    plt.ylabel("Probability")
    plt.ylim(0, max(0.85, float(np.nanmax(np.c_[df["mean_p0"], df["mean_p1"]])) * 1.12))
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    df = df.assign(novel=novel_title if novel_title else "")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================
# Condition comparisons
# =========================
def plot_condition_comparison(
    pf_adj: Optional[pd.DataFrame],
    pf_non: Optional[pd.DataFrame],
    pf_rand: Optional[pd.DataFrame],
    out_png: Path,
    title: str,
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
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
    plt.figure(figsize=(12.0, max(5.0, 0.60 * len(M))))
    xs = np.arange(len(M.index))
    w = 0.28
    palette = {"Adjacent": OI_BLUE, "Nonadjacent": OI_ORANGE, "Randomized": OI_GREEN}
    cols = [c for c in ["Adjacent", "Nonadjacent", "Randomized"] if c in M.columns]

    for i, c in enumerate(cols):
        plt.bar(xs + i*w, M[c].to_numpy(), width=w, label=c, alpha=0.95,
                color=palette.get(c, plt.get_cmap("cividis")(0.6)),
                edgecolor=OI_BLACK, linewidth=0.7)

    plt.xticks(xs + w, M.index, rotation=30, ha="right")
    plt.ylabel("Mean Conv(t)")
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    M = M.assign(novel=novel_title if novel_title else "")
    M.to_csv(out_csv, encoding="utf-8-sig")


# =========================
# Character composition
# =========================
def plot_gender_counts(
    edges: pd.DataFrame,
    out_png: Path,
    title: str = "Number of characters by gender",
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if edges is None or edges.empty:
        return

    if not {"a_speaker", "b_speaker", "a_gender", "b_gender"}.issubset(edges.columns):
        return

    df = edges.copy()
    all_speakers = pd.DataFrame({
        "speaker": pd.concat([df["a_speaker"], df["b_speaker"]], ignore_index=True),
        "gender":  pd.concat([df["a_gender"], df["b_gender"]], ignore_index=True),
    })
    all_speakers["gender"] = all_speakers["gender"].astype(str).str.upper().str[0]
    all_speakers["gender"] = all_speakers["gender"].where(all_speakers["gender"].isin(["F", "M"]))

    unique_speakers = all_speakers.drop_duplicates(subset=["speaker"])
    counts = unique_speakers["gender"].value_counts().reindex(["F", "M"], fill_value=0)

    df_counts = pd.DataFrame({
        "gender": counts.index,
        "n": counts.values
    })

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.6, 5.0))
    color_map = {"F": OI_PURPLE, "M": OI_ORANGE}
    plt.bar(df_counts["gender"], df_counts["n"],
            color=[color_map.get(g, OI_GREY) for g in df_counts["gender"]],
            edgecolor=OI_BLACK, linewidth=0.7, alpha=0.95)
    plt.xlabel("Gender")
    plt.ylabel("Number of distinct characters")
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    df_counts = df_counts.assign(novel=novel_title if novel_title else "")
    df_counts.to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_importance_counts(
    edges: pd.DataFrame,
    out_png: Path,
    title: str = "Number of characters by importance",
    csv_dir: Optional[Path] = None,
    novel_title: Optional[str] = None,
) -> None:
    if edges is None or edges.empty:
        return
    if not {"a_category", "b_category", "a_speaker", "b_speaker"}.issubset(edges.columns):
        return

    df = edges.copy()
    norm_names = {
        "maj": "major", "main": "major", "primary": "major",
        "int": "intermediate", "mid": "intermediate",
        "min": "minor"
    }
    for c in ["a_category", "b_category"]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace(norm_names)

    all_speakers = pd.DataFrame({
        "speaker": pd.concat([df["a_speaker"], df["b_speaker"]], ignore_index=True),
        "category": pd.concat([df["a_category"], df["b_category"]], ignore_index=True),
    })
    all_speakers["category"] = all_speakers["category"].where(
        all_speakers["category"].isin(["major", "intermediate", "minor"]), "unknown"
    )

    speaker_main_cat = (
        all_speakers.groupby(["speaker", "category"]).size()
        .groupby(level=0).idxmax().apply(lambda x: x[1])
    )

    counts = speaker_main_cat.value_counts().reindex(
        ["major", "intermediate", "minor"], fill_value=0
    )

    df_counts = counts.rename_axis("importance").reset_index(name="n")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))
    color_map = {"major": OI_BLUE, "intermediate": OI_ORANGE, "minor": OI_GREEN, "unknown": OI_GREY}
    plt.bar(
        df_counts["importance"], df_counts["n"],
        color=[color_map.get(c, OI_GREY) for c in df_counts["importance"]],
        edgecolor=OI_BLACK, linewidth=0.7, alpha=0.95
    )
    plt.xlabel("Character importance")
    plt.ylabel("Number of distinct characters")
    plt.title(f"{title}" + (f"\n({novel_title})" if novel_title else ""))
    plt.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.35, color=OI_GREY)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close()

    out_csv = csv_target(out_png, csv_dir)
    df_counts = df_counts.assign(novel=novel_title if novel_title else "")
    df_counts.to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================
# Runner
# =========================
def run_visuals_for_novel(
    in_dir: Path,
    out_root: Path,
    mode: str,
    min_support: int = 10,
    max_edges: int = 50,
    top_k: int = 20,
    csv_subdir: str = "tables",
    novel_title: Optional[str] = None,
) -> None:
    """
    Generate all figures for one analysis folder.
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

    out_root.mkdir(parents=True, exist_ok=True)
    csv_dir = out_root / csv_subdir

    # 1) Family overview
    plot_family_conv_bars(
        pf, out_root / "family_means.png",
        "Average convergence by feature family",
        csv_dir=csv_dir, novel_title=novel_title
    )
    plot_heatmap(
        pf, out_root / "heatmap.png", "Convergence by pair and feature",
        min_support=min_support, max_edges=max_edges, csv_dir=csv_dir, novel_title=novel_title
    )

    # 2) Edge distributions and leaders
    plot_edge_distribution(
        edges, out_root / "edge_distribution.png",
        "Distribution of A→B edge convergence", novel_title=novel_title
    )
    plot_top_edges(
        edges, out_root / "top_edges.png",
        "Top converging pairs (A→B)", top_k=top_k, csv_dir=csv_dir, novel_title=novel_title
    )
    plot_adaptability_distribution(
        pf, out_root / "adaptability_distribution.png",
        "Distribution of speaker adaptability (A)", csv_dir=csv_dir, novel_title=novel_title
    )

    # 3) Gender effects
    plot_gender_levels(
        pf, out_root / "gender_levels_responder.png",
        who="responder", csv_dir=csv_dir, novel_title=novel_title
    )
    plot_gender_levels(
        pf, out_root / "gender_levels_initiator.png",
        who="initiator", csv_dir=csv_dir, novel_title=novel_title
    )

    # 4) Importance effects
    plot_importance_levels(
        pf, out_root / "importance_levels_initiator.png",
        who="initiator", csv_dir=csv_dir, novel_title=novel_title
    )
    plot_importance_levels(
        pf, out_root / "importance_levels_responder.png",
        who="responder", csv_dir=csv_dir, novel_title=novel_title
    )

    # 5) Condition comparison
    plot_condition_comparison(
        pf, pf_non, pf_rand, out_root / "adj_non_rand.png",
        "Family means across conditions", csv_dir=csv_dir, novel_title=novel_title
    )

    # 6) Two-mode edge heatmaps
    plot_edge_gender_heatmap(
        edges, out_root / "edge_heatmap_gender.png",
        "Mean edge convergence by gender (A→B)", csv_dir=csv_dir, novel_title=novel_title
    )
    plot_edge_category_heatmap(
        edges, out_root / "edge_heatmap_importance.png",
        "Mean edge convergence by importance (A→B)", csv_dir=csv_dir, novel_title=novel_title
    )

    # 7) Character composition
    plot_gender_counts(
        edges, out_root / "char_counts_gender.png",
        "Number of characters by gender", csv_dir=csv_dir, novel_title=novel_title
    )
    plot_importance_counts(
        edges, out_root / "char_counts_importance.png",
        "Number of characters by importance", csv_dir=csv_dir, novel_title=novel_title
    )

    # 8) Classic p0 vs p1 figure per family (one-sided t-tests on per-pair diffs)
    plot_family_p1_p0(
        pf, out_root / "family_p0_p1.png",
        "P(B uses t) vs. P(B uses t | A uses t) by family", csv_dir=csv_dir, novel_title=novel_title
    )

    print(f"[viz] Visuals written to {out_root}  CSVs in {csv_dir}")
