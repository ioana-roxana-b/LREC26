from __future__ import annotations
from pathlib import Path

import pandas as pd

from preprocessing import preprocess_data          # cleans the PDNC quotation csv
from preprocessing import build_pairs              # merges contiguous turns and builds A→B pairs
from preprocessing import features                 # spaCy-based feature extractor
from preprocessing import prepare_liwc             # makes LIWC input csv

def run_one_novel_prep(
    novel_dir: Path,
    out_dir: Path,
    mode: str,
    lowercase: bool,
    character_info_arg: str | None,
) -> None:
    """
    Generate per-novel preprocessing artifacts for downstream analysis.
    Params:
        novel_dir: Path to the PDNC novel folder containing source CSVs.
        out_dir: Output directory where derived artifacts are written.
        mode: Either "spacy" or "liwc" to control final outputs.
        lowercase: Whether to lowercase quotes during cleaning.
        character_info_arg: Optional override path or {novel} pattern for metadata.
    Returns:
        None
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    quotes_csv = novel_dir/"quotation_info.csv"
    if not quotes_csv.exists():
        raise FileNotFoundError(f"Missing {quotes_csv}")

    # character metadata
    character_info_csv: Path | None = None
    if character_info_arg:
        character_info_csv = (
            Path(character_info_arg.format(novel=novel_dir.name))
            if "{novel}" in character_info_arg
            else Path(character_info_arg)
        )
    else:
        auto_candidate = novel_dir / "character_info.csv"
        if auto_candidate.exists():
            character_info_csv = auto_candidate

    # 1 clean quotes
    cleaned_csv = out_dir/"cleaned_quotes.csv"
    preprocess_data.preprocess_quotes(
        path=str(quotes_csv),
        out_path=str(cleaned_csv),
        lowercase=lowercase,
    )

    # 2 merge and pairs
    quotes_df = build_pairs.load_quotes(Path(cleaned_csv))
    merged_df = build_pairs.merge_contiguous_turns(quotes_df)
    merged_path = out_dir/"merged_turns.csv"
    merged_df.to_csv(merged_path, index=False)

    # adjacent
    pairs_df_adj = build_pairs.build_pairs_from_merged(merged_df)
    if character_info_csv and Path(character_info_csv).exists():
        pairs_df_adj = build_pairs.attach_metadata_to_pairs(pairs_df_adj, character_info_csv)
    pairs_path_adj = out_dir/"pairs_A_to_B.csv"
    pairs_df_adj.to_csv(pairs_path_adj, index=False)

    # nonadjacent
    pairs_df_nonadj = build_pairs.build_pairs_from_merged_nonadjacent(merged_df)
    if character_info_csv and Path(character_info_csv).exists():
        pairs_df_nonadj = build_pairs.attach_metadata_to_pairs(pairs_df_nonadj, character_info_csv)
    pairs_path_nonadj = out_dir/"pairs_A_to_B_nonadjacent.csv"
    pairs_df_nonadj.to_csv(pairs_path_nonadj, index=False)

    # randomized
    pairs_df_rand = build_pairs.build_pairs_randomized_from_adjacent(pairs_df_adj, seed=123)
    pairs_path_rand = out_dir/"pairs_A_to_B_randomized.csv"
    pairs_df_rand.to_csv(pairs_path_rand, index=False)

    # 3 final prep per mode
    mode = mode.lower()
    if mode == "spacy":
        # extract spaCy based features for all three conditions
        features.process_pairs(pairs_path_adj, out_dir/"pairs_features.csv")
        features.process_pairs(pairs_path_nonadj, out_dir/"pairs_features_nonadjacent.csv")
        features.process_pairs(pairs_path_rand, out_dir/"pairs_features_randomized.csv")
        print(f"[prep] spaCy features written in {out_dir}")
        
    elif mode == "liwc":
        # write LIWC inputs for all three conditions
        prepare_liwc.make_liwc_input(pairs_csv=pairs_path_adj, liwc_input_csv=out_dir/"liwc_input.csv")
        prepare_liwc.make_liwc_input(pairs_csv=pairs_path_nonadj, liwc_input_csv=out_dir/"liwc_input_nonadjacent.csv")
        prepare_liwc.make_liwc_input(pairs_csv=pairs_path_rand, liwc_input_csv=out_dir/"liwc_input_randomized.csv")
        print(f"[prep] LIWC inputs written in {out_dir}")
    else:
        raise ValueError("mode must be spacy or liwc or both")

def run_corpus_prep(
    novel_dirs,
    out_dir,
    mode,
    lowercase=False,
    character_info_pattern=None,
):
    """
    Build merged corpus artifacts by running preprocessing for each novel.
    Params:
        novel_dirs: Iterable of novel directories to include in the corpus.
        out_dir: Root output directory where corpus artifacts are written.
        mode: Either "spacy" or "liwc" to control final artifacts.
        lowercase: Whether to lowercase quotes during cleaning.
        character_info_pattern: Optional {novel} pattern to locate metadata files.
    Returns:
        None
    """
    mode = mode.lower()
    out_root = Path(out_dir)
    out_all = out_root/"All_Novels"
    out_root.mkdir(parents=True, exist_ok=True)
    out_all.mkdir(parents=True, exist_ok=True)

    all_adj, all_non, all_rand = [], [], []

    for nd in novel_dirs:
        novel_dir = Path(nd)
        title = novel_dir.name
        print(f"[corpus] processing {title}...")

        # Run normal preprocessing
        out_novel = out_root/title
        run_one_novel_prep(
            novel_dir=novel_dir,
            out_dir=out_novel,
            mode=mode,
            lowercase=lowercase,
            character_info_arg=character_info_pattern.format(novel=title) if character_info_pattern else None,
        )

        # Load per-novel outputs
        adj = pd.read_csv(out_novel/"pairs_A_to_B.csv")
        non = pd.read_csv(out_novel/"pairs_A_to_B_nonadjacent.csv")
        rnd = pd.read_csv(out_novel/"pairs_A_to_B_randomized.csv")
        for df in [adj, non, rnd]:
            df["novel"] = title

        all_adj.append(adj)
        all_non.append(non)
        all_rand.append(rnd)

    # Concatenate everything
    adj_all = pd.concat(all_adj, ignore_index=True)
    non_all = pd.concat(all_non, ignore_index=True)
    rnd_all = pd.concat(all_rand, ignore_index=True)

    # Write unified pairs
    adj_path = out_all/"pairs_A_to_B.csv"
    non_path = out_all/"pairs_A_to_B_nonadjacent.csv"
    rnd_path = out_all/"pairs_A_to_B_randomized.csv"
    adj_all.to_csv(adj_path, index=False)
    non_all.to_csv(non_path, index=False)
    rnd_all.to_csv(rnd_path, index=False)

    print(f"[corpus] merged pairs written to {out_all}")

    # Create LIWC or spaCy features depending on mode
    if mode == "spacy":
        from preprocessing import features
        features.process_pairs(adj_path, out_all/"pairs_features.csv")
        features.process_pairs(non_path, out_all/"pairs_features_nonadjacent.csv")
        features.process_pairs(rnd_path, out_all/"pairs_features_randomized.csv")
        print(f"[corpus] spaCy feature CSVs saved in {out_all}")
    elif mode == "liwc":
        from preprocessing import prepare_liwc
        prepare_liwc.make_liwc_input(adj_path, out_all/"liwc_input.csv")
        prepare_liwc.make_liwc_input(non_path, out_all/"liwc_input_nonadjacent.csv")
        prepare_liwc.make_liwc_input(rnd_path, out_all/"liwc_input_randomized.csv")
        print(f"[corpus] LIWC input CSVs saved in {out_all}")
    else:
        raise ValueError("mode must be 'liwc' or 'spacy'")

    print(f"[corpus] done. Combined corpus ready at {out_all}")
