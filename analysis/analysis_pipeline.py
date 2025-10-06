from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from analysis import (
    convergence_score_spacy,
    convergence_score_liwc,
    merge_liwc,
    stats_tests,
    visualisation,
    graph_analysis,
)


def run_one_novel_analysis(
    in_dir: Path,
    out_conv_dir: Path,
    mode: str,
    min_support: int,
    liwc_results_csv: Optional[Path],
    liwc_results_nonadj_csv: Optional[Path],
    liwc_results_rand_csv: Optional[Path],
) -> None:
    """
    Compute convergence data for one novel.

    Params:
        in_dir: Preprocess outputs folder that contains pairs files.
        out_conv_dir: Destination folder for conv_by_pair_feature_* and conv_edge_summary_* CSVs.
        mode: Either spacy or liwc.
        min_support: Minimum number of A triggers per family within a dyad to include that family.
        liwc_results_csv: Path to LIWC desktop export for the adjacent condition or None.
        liwc_results_nonadj_csv: Path to LIWC desktop export for the nonadjacent condition or None.
        liwc_results_rand_csv: Path to LIWC desktop export for the randomized condition or None.

    Returns:
        None. Writes convergence CSVs to out_conv_dir.
    """
    mode = mode.lower()
    out_conv_dir.mkdir(parents=True, exist_ok=True)

    if mode == "spacy":
        family_rename = {
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

        configs = [
            ("pairs_features.csv",
             "conv_by_pair_feature_spacy.csv",
             "conv_edge_summary_spacy.csv"),
            ("pairs_features_nonadjacent.csv",
             "conv_by_pair_feature_spacy_nonadjacent.csv",
             "conv_edge_summary_spacy_nonadjacent.csv"),
            ("pairs_features_randomized.csv",
             "conv_by_pair_feature_spacy_randomized.csv",
             "conv_edge_summary_spacy_randomized.csv"),
        ]

        for in_name, out_pf, out_edge in configs:
            in_path = in_dir / in_name
            if not in_path.exists():
                print(f"[spaCy] Missing {in_path}, skipping.")
                continue

            convergence_score_spacy.compute_convergence_spacy(
                pairs_features_path=in_path,
                out_conv_pair_feature=out_conv_dir / out_pf,
                out_conv_edge=out_conv_dir / out_edge,
                min_support=min_support,
                family_rename=family_rename,
                min_resp_support=2,
            )

    elif mode == "liwc":
        pairs_adj = in_dir / "pairs_A_to_B.csv"
        pairs_non = in_dir / "pairs_A_to_B_nonadjacent.csv"
        pairs_rand = in_dir / "pairs_A_to_B_randomized.csv"

        have_adj = liwc_results_csv and Path(liwc_results_csv).exists()
        have_non = liwc_results_nonadj_csv and Path(liwc_results_nonadj_csv).exists()
        have_rand = liwc_results_rand_csv and Path(liwc_results_rand_csv).exists()

        if not have_adj:
            print("[LIWC] Missing adjacent LIWC export. Run LIWC desktop first.")
            return

        print("[LIWC] Merging LIWC outputs with pairs...")
        merged_adj = merge_liwc.run_for_condition_merge(
            pairs_csv=pairs_adj,
            liwc_results_csv=Path(liwc_results_csv),
            out_dir=out_conv_dir,
            label="adjacent",
        )

        merged_non = None
        if have_non and pairs_non.exists():
            merged_non = merge_liwc.run_for_condition_merge(
                pairs_csv=pairs_non,
                liwc_results_csv=Path(liwc_results_nonadj_csv),
                out_dir=out_conv_dir,
                label="nonadjacent",
            )
        else:
            print("[LIWC] Skipping nonadjacent missing pairs or export.")

        merged_rand = None
        if have_rand and pairs_rand.exists():
            merged_rand = merge_liwc.run_for_condition_merge(
                pairs_csv=pairs_rand,
                liwc_results_csv=Path(liwc_results_rand_csv),
                out_dir=out_conv_dir,
                label="randomized",
            )
        else:
            print("[LIWC] Skipping randomized missing pairs or export.")

        print("[LIWC] Computing convergence scores...")
        convergence_score_liwc.run_for_condition_convergence(
            pairs_with_liwc_csv=merged_adj,
            out_dir=out_conv_dir,
            label="adjacent",
            min_support=min_support,
            min_resp_support=None,
        )
        if merged_non:
            convergence_score_liwc.run_for_condition_convergence(
                pairs_with_liwc_csv=merged_non,
                out_dir=out_conv_dir,
                label="nonadjacent",
                min_support=min_support,
                min_resp_support=None,
            )
        if merged_rand:
            convergence_score_liwc.run_for_condition_convergence(
                pairs_with_liwc_csv=merged_rand,
                out_dir=out_conv_dir,
                label="randomized",
                min_support=min_support,
                min_resp_support=None,
            )

    else:
        raise ValueError("Mode must be spacy or liwc.")


def run_full_analysis(
    in_dir: Path,
    out_dir: Path,
    mode: str,
    min_support: int,
    liwc_paths: Dict[str, Optional[Path]],
) -> None:
    """
    Run the complete analysis workflow for one novel.

    Params:
        in_dir: Preprocess outputs folder for this novel.
        out_dir: Root folder where analysis_results will be created.
        mode: Either spacy or liwc.
        min_support: Minimum number of A triggers per family within a dyad to include that family.
        liwc_paths: Dict with keys adjacent nonadjacent randomized mapping to LIWC export paths or None.

    Returns:
        None. Creates analysis_results with convergence_results plots graph_results and stats_results.
    """
    base = out_dir / "analysis_results"
    conv_dir = base / "convergence_results"
    stats_dir = base / "stats_results"
    plot_dir = base / "plots"
    graph_dir = base / "graph_results"

    for p in (conv_dir, stats_dir, plot_dir, graph_dir):
        p.mkdir(parents=True, exist_ok=True)

    run_one_novel_analysis(
        in_dir=in_dir,
        out_conv_dir=conv_dir,
        mode=mode,
        min_support=min_support,
        liwc_results_csv=liwc_paths.get("adjacent"),
        liwc_results_nonadj_csv=liwc_paths.get("nonadjacent"),
        liwc_results_rand_csv=liwc_paths.get("randomized"),
    )

    visualisation.run_visuals_for_novel(conv_dir, plot_dir, mode)

    graph_analysis.run_graph_analysis(conv_dir, graph_dir, mode, positive_only=False)

    if mode == "spacy":
        adj_path = conv_dir / "conv_by_pair_feature_spacy.csv"
        non_path = conv_dir / "conv_by_pair_feature_spacy_nonadjacent.csv"
        rand_path = conv_dir / "conv_by_pair_feature_spacy_randomized.csv"
        out_stats = stats_dir / "stats_summary_spacy.csv"
    elif mode == "liwc":
        adj_path = conv_dir / "conv_by_pair_feature_liwc.csv"
        non_path = conv_dir / "conv_by_pair_feature_liwc_nonadjacent.csv"
        rand_path = conv_dir / "conv_by_pair_feature_liwc_randomized.csv"
        out_stats = stats_dir / "stats_summary_liwc.csv"

    pf_adj = pd.read_csv(adj_path, encoding="utf-8-sig") if adj_path.exists() else None
    pf_non = pd.read_csv(non_path, encoding="utf-8-sig") if non_path.exists() else None
    pf_rand = pd.read_csv(rand_path, encoding="utf-8-sig") if rand_path.exists() else None

    stats_tests.summarize_and_compare(
        pf_adj=pf_adj,
        pf_non=pf_non,
        pf_rand=pf_rand,
        out_csv=out_stats,
        alpha=0.05,
        n_boot=1000,
    )

    print(f"[analysis] Complete. Results written to {base}")
