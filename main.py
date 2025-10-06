from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore")

from preprocessing import preprocessing_pipeline
from analysis import analysis_pipeline


def resolve_liwc_paths(base_in_dir: Path, novel_label: str, liwc_adj: Optional[str], liwc_non: Optional[str], liwc_rnd: Optional[str]) -> Dict[str, Optional[Path]]:
    """
    Resolve LIWC export paths for the requested novel.
    Params:
        base_in_dir: Root directory containing preprocess outputs.
        novel_label: Name used to expand {novel} placeholders.
        liwc_adj: Optional path or pattern for adjacent exports.
        liwc_non: Optional path or pattern for nonadjacent exports.
        liwc_rnd: Optional path or pattern for randomized exports.
    Returns:
        Mapping of LIWC export categories to resolved file paths.
    """
    candidates = [base_in_dir, base_in_dir / "analysis_results"]
    names_adj = [
        "LIWC-22 Results - liwc_input - LIWC Analysis.csv",
        "liwc_input.csv",
    ]
    names_non = [
        "LIWC-22 Results - liwc_input_nonadjacent - LIWC Analysis.csv",
        "liwc_input_nonadjacent.csv",
    ]
    names_rnd = [
        "LIWC-22 Results - liwc_input_randomized - LIWC Analysis.csv",
        "liwc_input_randomized.csv",
    ]

    if liwc_adj:
        adj = Path(liwc_adj.format(novel=novel_label)) if "{novel}" in liwc_adj else Path(liwc_adj)
    else:
        adj = None
    if adj is None:
        for folder in candidates:
            for name in names_adj:
                candidate = folder / name
                if candidate.exists():
                    adj = candidate
                    break
            if adj is not None:
                break

    if liwc_non:
        non = Path(liwc_non.format(novel=novel_label)) if "{novel}" in liwc_non else Path(liwc_non)
    else:
        non = None
    if non is None:
        for folder in candidates:
            for name in names_non:
                candidate = folder / name
                if candidate.exists():
                    non = candidate
                    break
            if non is not None:
                break

    if liwc_rnd:
        rnd = Path(liwc_rnd.format(novel=novel_label)) if "{novel}" in liwc_rnd else Path(liwc_rnd)
    else:
        rnd = None
    if rnd is None:
        for folder in candidates:
            for name in names_rnd:
                candidate = folder / name
                if candidate.exists():
                    rnd = candidate
                    break
            if rnd is not None:
                break

    return {"adjacent": adj, "nonadjacent": non, "randomized": rnd}


def main():
    """
    Parse CLI arguments and run the requested pipeline stage.
    Params:
        None
    Returns:
        None
    """
    ap = argparse.ArgumentParser(description="Run CAT pipeline stage without subprocess.")
    ap.add_argument(
        "--stage",
        choices=["preprocess", "analysis"],
        required=True,
        help="Select which pipeline stage to execute.",
    )

    ap.add_argument(
        "--mode",
        choices=["spacy", "liwc"],
        required=True,
        help="Choose between spaCy feature extraction or LIWC integration.",
    )
    ap.add_argument("--novel_dir", nargs="+", required=True,
                    help="Either a single PDNC novel folder or a list of them for corpus-level preprocessing.")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Destination folder for per-novel outputs or the merged corpus.",
    )
    ap.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text while preprocessing (ignored for analysis stage).",
    )
    ap.add_argument("--character_info", type=str, default=None,
                    help="Path to character_info.csv for preprocess.")

    ap.add_argument(
        "--min_support",
        type=int,
        default=10,
        help="Minimum co-occurrence threshold for statistical comparisons.",
    )
    ap.add_argument(
        "--liwc_results",
        type=str,
        default=None,
        help="Path to the adjacent LIWC export CSV.",
    )
    ap.add_argument(
        "--liwc_results_nonadj",
        type=str,
        default=None,
        help="Path to the nonadjacent LIWC export CSV.",
    )
    ap.add_argument(
        "--liwc_results_rand",
        type=str,
        default=None,
        help="Path to the randomized LIWC export CSV.",
    )

    args = ap.parse_args()

    stage = args.stage.lower()
    mode = args.mode.lower()
    out_root = Path(args.out_dir)

    if stage == "preprocess":
        if len(args.novel_dir) > 1:
            # multi-novel case
            print("[stage] Running corpus-level preprocessing...")
            preprocessing_pipeline.run_corpus_prep(
                novel_dirs=args.novel_dir,
                out_dir=out_root,
                mode=mode,
                lowercase=args.lowercase,
                character_info_pattern=args.character_info,
            )
        else:
            # single novel case
            novel_dir = Path(args.novel_dir[0])
            preprocessing_pipeline.run_one_novel_prep(
                novel_dir=novel_dir,
                out_dir=out_root,
                mode=mode,
                lowercase=args.lowercase,
                character_info_arg=args.character_info,
            )
        print(f"[preprocess] Done → {out_root}")

    else:
        # ------------ analysis ------------
        in_dir = Path(args.novel_dir[0])
        novel_label = in_dir.name
        out_root.mkdir(parents=True, exist_ok=True)

        liwc_paths = resolve_liwc_paths(
            base_in_dir=in_dir,
            novel_label=novel_label,
            liwc_adj=args.liwc_results,
            liwc_non=args.liwc_results_nonadj,
            liwc_rnd=args.liwc_results_rand,
        )

        analysis_pipeline.run_full_analysis(
            in_dir=in_dir,
            out_dir=out_root,
            mode=mode,
            min_support=args.min_support,
            liwc_paths=liwc_paths,
        )

        print(f"[analysis] Done → {out_root}")

if __name__ == "__main__":
    main()
