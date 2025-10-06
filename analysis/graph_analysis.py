from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.stats import pearsonr


def load_edges(edges_csv: Path, prefer_weight_col: str = "mean_conv_w") -> pd.DataFrame:
    """
    Load edge-level convergence results and pick the working weight column.

    Params:
        edges_csv: Path to conv_edge_summary_*.csv produced by the convergence step.
        prefer_weight_col: Name of the weight column to prefer when available.

    Returns:
        DataFrame with columns a_speaker b_speaker weight_work and any available a_* / b_* metadata.
    """
    df = pd.read_csv(edges_csv, encoding="utf-8-sig")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    ren = {}
    if "A" in df.columns and "a_speaker" not in df.columns:
        ren["A"] = "a_speaker"
    if "B" in df.columns and "b_speaker" not in df.columns:
        ren["B"] = "b_speaker"
    if ren:
        df = df.rename(columns=ren)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()

    # Choose the working weight column
    weight_col = None
    if prefer_weight_col in df.columns:
        weight_col = prefer_weight_col
    elif "mean_conv" in df.columns:
        weight_col = "mean_conv"
    else:
        for cand in ["conv_mean_unw", "conv_mean_w", "conv_mean"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "mean_conv"})
                weight_col = "mean_conv"
                break
    if weight_col is None:
        raise ValueError("No edge weight column found. Need mean_conv_w or mean_conv")

    required = {"a_speaker", "b_speaker", weight_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in edges CSV: {sorted(missing)}")

    keep = ["a_speaker", "b_speaker", weight_col, "n_pairs", "n_features", "mean_conv", "mean_conv_w", "n_triggers_total"]
    keep += [c for c in df.columns if c.startswith("a_") or c.startswith("b_")]
    keep = pd.Index(keep).drop_duplicates().tolist()
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[weight_col, "a_speaker", "b_speaker"])

    if "n_pairs" in df.columns:
        df = df[df["n_pairs"].fillna(0) > 0].copy()

    for col in ("a_speaker", "b_speaker"):
        df[col] = df[col].astype(str).map(lambda s: " ".join(s.split()))

    df = df.rename(columns={weight_col: "weight_work"})
    return df


def build_graph(
    edges: pd.DataFrame,
    positive_only: bool = False,
    weight_col: str = "weight_work",
) -> nx.DiGraph:
    """
    Build a directed graph where edges A→B are weighted by convergence.

    Params:
        edges: Edge table from load_edges with a_speaker b_speaker and weight_work.
        positive_only: Keep only edges with positive weight.
        weight_col: Column name to use as weight.

    Returns:
        A directed NetworkX DiGraph with edge attribute weight.
    """
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        a = str(r["a_speaker"]).strip()
        b = str(r["b_speaker"]).strip()
        w = float(r.get(weight_col, 0.0))
        if positive_only and w <= 0:
            continue
        G.add_edge(a, b, weight=w)
    return G


def compute_metrics(G: nx.DiGraph) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute node metrics and global graph statistics.

    Params:
        G: Directed graph with edge attribute weight.

    Returns:
        A tuple of
        First is a DataFrame of node metrics including strength and betweenness
        Second is a dict with n_nodes n_edges density and reciprocity.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after filtering")

    out_deg = dict(G.out_degree())
    in_deg = dict(G.in_degree())

    out_strength = {n: 0.0 for n in G.nodes()}
    in_strength = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        aw = abs(w)
        out_strength[u] += aw
        in_strength[v] += aw

    # Weighted betweenness: transform weights to lengths inline
    if G.number_of_edges() > 0:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 0.0))
            length = 1.0 / (abs(w) + 1e-9)
            H.add_edge(u, v, length=length)
        btw_w = nx.betweenness_centrality(H, normalized=True, weight="length")
    else:
        btw_w = {n: 0.0 for n in G.nodes()}

    btw = nx.betweenness_centrality(G, normalized=True) if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}

    df_nodes = pd.DataFrame({
        "character": list(G.nodes()),
        "out_degree": [out_deg.get(n, 0) for n in G.nodes()],
        "in_degree": [in_deg.get(n, 0) for n in G.nodes()],
        "adaptability_out_strength": [out_strength.get(n, 0.0) for n in G.nodes()],
        "adaptability_in_strength": [in_strength.get(n, 0.0) for n in G.nodes()],
        "betweenness": [btw.get(n, 0.0) for n in G.nodes()],
        "betweenness_weighted": [btw_w.get(n, 0.0) for n in G.nodes()],
    })

    genders = nx.get_node_attributes(G, "gender")
    if genders:
        df_nodes["gender"] = df_nodes["character"].map(genders)
    cats = nx.get_node_attributes(G, "category")
    if cats:
        df_nodes["category"] = df_nodes["character"].map(cats)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "reciprocity": nx.reciprocity(G) if G.number_of_edges() > 1 else np.nan,
    }
    return df_nodes.sort_values("adaptability_out_strength", ascending=False), stats


def detect_communities(G: nx.DiGraph) -> pd.DataFrame:
    """
    Detect undirected communities via greedy modularity.

    Params:
        G: Directed graph to project to undirected for community detection.

    Returns:
        DataFrame with columns character and community_id.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["character", "community_id"])
    UG = G.to_undirected(reciprocal=False)
    from networkx.algorithms import community
    comms = list(community.greedy_modularity_communities(UG, weight=None))
    mapping = {n: cid for cid, nodes in enumerate(comms) for n in nodes}
    return pd.DataFrame({"character": list(mapping.keys()),
                         "community_id": list(mapping.values())})


def reciprocity_pair_table(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Build a paired table of A→B and B→A weights for mutual dyads.

    Params:
        edges: Edge DataFrame with a_speaker b_speaker and weight_work.

    Returns:
        DataFrame with columns a_speaker b_speaker w_ab w_ba for mutual edges only.
    """
    df = edges[["a_speaker", "b_speaker", "weight_work"]].copy()
    df = df.rename(columns={"weight_work": "w_ab"})
    df_rev = edges[["a_speaker", "b_speaker", "weight_work"]].copy()
    df_rev = df_rev.rename(columns={"a_speaker": "b_speaker", "b_speaker": "a_speaker", "weight_work": "w_ba"})
    return df.merge(df_rev, on=["a_speaker", "b_speaker"], how="inner")


def plot_reciprocity(paired: pd.DataFrame, out_png: Path, out_csv: Path) -> Tuple[float, float]:
    """
    Plot A→B vs B→A scatter and save paired data.

    Params:
        paired: DataFrame with columns w_ab and w_ba for mutual dyads.
        out_png: Path for the scatter image.
        out_csv: Path for the CSV of paired values.

    Returns:
        Tuple with Pearson r and the share of pairs with matching sign.
    """
    if paired.empty:
        paired.to_csv(out_csv, index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No mutual dyads", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, bbox_inches="tight", dpi=220)
        plt.close(fig)
        return np.nan, np.nan

    r, _ = pearsonr(paired["w_ab"], paired["w_ba"])
    sign_match = float(np.mean(np.sign(paired["w_ab"]) == np.sign(paired["w_ba"])))

    # Colors from a colorblind friendly palette
    point_color = "#0072B2"      # blue
    diagonal_color = "#4D4D4D"   # dark gray
    grid_color = "#BFBFBF"       # light gray

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.scatter(
        paired["w_ab"], paired["w_ba"],
        s=24, alpha=0.75,
        facecolor=point_color,
        edgecolor="white", linewidth=0.4
    )

    lim = max(abs(paired["w_ab"]).max(), abs(paired["w_ba"]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim],
            linestyle="--", linewidth=1.2, color=diagonal_color)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("A → B weight")
    ax.set_ylabel("B → A weight")
    ax.set_title(f"Reciprocity r={r:.3f}  sign match={sign_match:.3f}")

    ax.grid(True, linestyle=":", linewidth=0.8, color=grid_color, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)

    paired.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return float(r), float(sign_match)



def plot_graph(
    G: nx.DiGraph,
    out_png: Path,
    color_attr: str = "gender",
    show_labels: bool = True,
    figsize: Tuple[float, float] = (12.5, 9.0),
    font_size: int = 10,
    weight_attr: str = "weight",
) -> None:
    """
    Draw a directed convergence network with colorblind safe styling.

    Params:
        G: Directed graph with weight on edges.
        out_png: Output path for the PNG image.
        color_attr: Node attribute to color by. Typically gender or category.
        show_labels: Whether to draw node labels.
        figsize: Figure size in inches.
        font_size: Label font size.
        weight_attr: Edge attribute to use when scaling widths.

    Returns:
        None. Writes the figure to disk.
    """
    if G.number_of_nodes() == 0:
        return

    G = nx.relabel_nodes(G, lambda n: " ".join(str(n).split()))

    out_strength = {n: 0.0 for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get(weight_attr, 0.0))
        out_strength[u] += abs(w)

    k = 1.1 if G.number_of_nodes() <= 40 else 1.3
    pos = nx.spring_layout(G, seed=7, k=k, iterations=100)

    # Colorblind friendly palette
    attr_values = nx.get_node_attributes(G, color_attr)
    palette = {"F": "#CC79A7", "M": "#0072B2", "U": "#999999"}  # magenta, blue, grey
    node_colors = [palette.get(attr_values.get(n, "U"), "#999999") for n in G.nodes()]

    def size_for(n: str) -> float:
        return 240.0 + 260.0 * np.sqrt(max(out_strength.get(n, 0.0), 0.0))

    node_sizes = {n: size_for(n) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=list(node_sizes.values()),
        edgecolors="#4D4D4D",
        linewidths=0.7,
        ax=ax,
    )

    def shrink_pts(n: str) -> float:
        return 0.62 * np.sqrt(node_sizes[n])

    def edge_width(w: float) -> float:
        return 0.6 + 3.0 * min(abs(w), 0.15) / 0.15

    drawn_pairs = set()
    for u, v, d in G.edges(data=True):
        pair = tuple(sorted((u, v)))
        if pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        has_uv = G.has_edge(u, v)
        has_vu = G.has_edge(v, u)

        if has_uv:
            w_uv = float(G[u][v].get(weight_attr, 0.0))
            a = FancyArrowPatch(
                pos[u], pos[v],
                arrowstyle="-|>",
                mutation_scale=10,
                shrinkA=shrink_pts(u),
                shrinkB=shrink_pts(v),
                connectionstyle=f"arc3,rad={0.18 if has_vu else 0.0}",
                lw=edge_width(w_uv),
                color="#6E6E6E",
                alpha=0.95,
            )
            ax.add_patch(a)
        if has_vu:
            w_vu = float(G[v][u].get(weight_attr, 0.0))
            b = FancyArrowPatch(
                pos[v], pos[u],
                arrowstyle="-|>",
                mutation_scale=10,
                shrinkA=shrink_pts(v),
                shrinkB=shrink_pts(u),
                connectionstyle="arc3,rad=-0.18",
                lw=edge_width(w_vu),
                color="#6E6E6E",
                alpha=0.95,
            )
            ax.add_patch(b)

    if show_labels:
        label_pos = {n: (x, y + 0.02) for n, (x, y) in pos.items()}
        labels = {n: n for n in G.nodes()}
        nx.draw_networkx_labels(
            G,
            label_pos,
            labels=labels,
            font_size=font_size,
            font_color="black",
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#BFBFBF", alpha=0.75),
        )

    # Legend
    if attr_values:
        from matplotlib.lines import Line2D
        keys = ["F", "M", "U"]
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=palette[k], markeredgecolor="#4D4D4D", label=k)
                   for k in keys if k in set(attr_values.values()) or k == "U"]
        if handles:
            ax.legend(handles=handles, title=color_attr, loc="upper left", frameon=False)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def run_graph_analysis(
    in_dir: Path,
    out_dir: Path,
    mode: str,
    positive_only: bool = False,
    prefer_weight_col: str = "mean_conv_w",
) -> None:
    """
    Build and analyze the convergence graph and write all artifacts.

    Params:
        in_dir: Folder with convergence results, typically {root}/analysis_results/convergence_results.
        out_dir: Destination folder for node_metrics.csv communities.csv reciprocity files graph image and summary.
        mode: liwc or spacy, used to pick the right edges filename.
        positive_only: Keep only edges with positive weight.
        prefer_weight_col: Which edge weight column to prefer if present.

    Returns:
        None. Writes graph data, plots, tables and a summary text file to out_dir.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_name = "conv_edge_summary_liwc.csv" if mode.lower() == "liwc" else "conv_edge_summary_spacy.csv"
    edges_csv = in_dir / edges_name
    if not edges_csv.exists():
        print(f"[graph] Skip: edges file not found: {edges_csv}")
        return

    # Load edges and build graph
    edges = load_edges(edges_csv, prefer_weight_col=prefer_weight_col)
    G = build_graph(edges, positive_only=positive_only, weight_col="weight_work")

    # Attach node attributes from modal values found in the edges table
    # Gender
    gmap: Dict[str, str] = {}
    for side in ["a", "b"]:
        s_col = f"{side}_speaker"
        g_col = f"{side}_gender"
        if g_col in edges.columns:
            tmp = edges[[s_col, g_col]].dropna()
            if not tmp.empty:
                mode_val = tmp.groupby(s_col)[g_col].agg(lambda x: x.value_counts().idxmax())
                gmap.update(mode_val.to_dict())
    if gmap:
        nx.set_node_attributes(G, gmap, "gender")

    # Category / importance
    cmap: Dict[str, str] = {}
    for side in ["a", "b"]:
        s_col = f"{side}_speaker"
        c_col = f"{side}_category"
        if c_col in edges.columns:
            tmp = edges[[s_col, c_col]].dropna()
            if not tmp.empty:
                mode_val = tmp.groupby(s_col)[c_col].agg(lambda x: x.value_counts().idxmax())
                cmap.update(mode_val.to_dict())
    if cmap:
        nx.set_node_attributes(G, cmap, "category")

    # Node and global metrics
    df_nodes, stats = compute_metrics(G)
    df_nodes.to_csv(out_dir / "node_metrics.csv", index=False, encoding="utf-8-sig")

    # Communities
    comm = detect_communities(G)
    comm.to_csv(out_dir / "communities.csv", index=False, encoding="utf-8-sig")

    # Assortativity and two-mode A→B tables (done inline here)
    from networkx.algorithms.assortativity import attribute_assortativity_coefficient

    assort_gender = None
    genders_attr = nx.get_node_attributes(G, "gender")
    if genders_attr:
        UG = G.to_undirected()
        nx.set_node_attributes(UG, genders_attr, "gender")
        try:
            assort_gender = float(attribute_assortativity_coefficient(UG, "gender"))
        except Exception:
            assort_gender = None

    assort_category = None
    cats_attr = nx.get_node_attributes(G, "category")
    if cats_attr:
        UG2 = G.to_undirected()
        nx.set_node_attributes(UG2, cats_attr, "category")
        try:
            assort_category = float(attribute_assortativity_coefficient(UG2, "category"))
        except Exception:
            assort_category = None

    # Two-mode edge averages
    if {"a_gender", "b_gender"}.issubset(edges.columns):
        gtab = (edges.groupby(["a_gender", "b_gender"], dropna=False)
                      .agg(mean_weight=("weight_work", "mean"), count=("weight_work", "size"))
                      .reset_index())
        gtab.to_csv(out_dir / "edge_table_gender_A_to_B.csv", index=False, encoding="utf-8-sig")

    if {"a_category", "b_category"}.issubset(edges.columns):
        ctab = (edges.groupby(["a_category", "b_category"], dropna=False)
                      .agg(mean_weight=("weight_work", "mean"), count=("weight_work", "size"))
                      .reset_index())
        ctab.to_csv(out_dir / "edge_table_category_A_to_B.csv", index=False, encoding="utf-8-sig")

    # Reciprocity
    paired = reciprocity_pair_table(edges)
    rec_r, rec_sign = plot_reciprocity(
        paired, out_png=out_dir / "reciprocity_scatter.png", out_csv=out_dir / "reciprocity_pairs.csv"
    )

    # Graph plot
    plot_graph(G, out_dir / "graph_network.png", color_attr="gender", show_labels=True)

    # Summary file written inline
    lines: List[str] = []
    lines.append(f"nodes {stats.get('n_nodes')}")
    lines.append(f"edges {stats.get('n_edges')}")
    lines.append(f"density {stats.get('density'):.4f}")
    rec = stats.get("reciprocity")
    lines.append(f"reciprocity {rec:.4f}" if rec is not None and not np.isnan(rec) else "reciprocity NA")
    if assort_gender is not None:
        lines.append(f"gender_assortativity {assort_gender:.4f}")
    if assort_category is not None:
        lines.append(f"category_assortativity {assort_category:.4f}")
    if rec_r is not None and not np.isnan(rec_r):
        lines.append(f"reciprocity_correlation {rec_r:.4f}")
    if rec_sign is not None and not np.isnan(rec_sign):
        lines.append(f"reciprocity_sign_match {rec_sign:.4f}")

    if not edges.empty:
        E = {(r.a_speaker, r.b_speaker) for r in edges.itertuples(index=False)}
        mutual = sum(1 for a, b in E if (b, a) in E)
        lines.append(f"mutual_edge_pairs {mutual // 2}")

    if not comm.empty:
        k = len(comm["community_id"].unique())
        counts = comm["community_id"].value_counts().sort_index()
        lines.append(f"communities_detected {k}")
        lines.append("community_sizes " + ", ".join(f"{cid}:{cnt}" for cid, cnt in counts.items()))

    (out_dir / "graph_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[graph] Results saved → {out_dir}")
