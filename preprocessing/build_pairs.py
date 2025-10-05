import ast
import re
from pathlib import Path
from typing import List

import pandas as pd


def parse_addressees(raw_value) -> List[str]:
    """
    Clean the addressees cell into a list of names.
    Params:
        raw_value: Original addressees field from the quotes table.
    Returns:
        List of normalized addressee strings.
    """

    if pd.isna(raw_value):
        return []

    if isinstance(raw_value, list):
        cleaned_items: List[str] = []
        for item in raw_value:
            text = re.sub(r"\s+", " ", str(item)).strip()
            if text:
                cleaned_items.append(text)
        return cleaned_items

    serialized = str(raw_value).strip()
    if not serialized:
        return []

    try:
        parsed = ast.literal_eval(serialized)
    except Exception:
        parsed = serialized.split(",")

    if isinstance(parsed, list):
        cleaned_items = []
        for item in parsed:
            text = re.sub(r"\s+", " ", str(item)).strip()
            if text:
                cleaned_items.append(text)
        return cleaned_items

    return [re.sub(r"\s+", " ", str(parsed)).strip()]


def load_quotes(path: Path) -> pd.DataFrame:
    """
    Load a quotes CSV, standardize columns, and compute turn indices.
    Params:
        path: File path to the cleaned quotations export.
    Returns:
        DataFrame with normalized quotes and parsed addressee lists.
    """

    df = pd.read_csv(path)
    df = df.rename(columns={"speaker": "quoteBy", "addressees": "quoteTo"})

    required_columns = ["quoteID", "quoteText", "quoteBy", "quoteTo"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df["quoteText"] = (
        df["quoteText"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df["quoteTo_list"] = df["quoteTo"].apply(parse_addressees)
    df = df[["quoteID", "quoteText", "quoteBy", "quoteTo_list"]].copy()
    df = df.reset_index(drop=True)
    df["turn_index"] = df.index
    return df


def merge_contiguous_turns(quotes: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive turns that share speaker and addressee set.
    Params:
        quotes: DataFrame produced by `load_quotes`.
    Returns:
        DataFrame of merged turns with contiguous dialogue segments.
    """

    if quotes.empty:
        return pd.DataFrame(
            columns=[
                "merge_id",
                "speaker",
                "addressees",
                "text_merged",
                "first_turn_index",
                "last_turn_index",
                "source_quoteIDs",
            ]
        )

    rows = []
    merge_id = 0
    current_speaker = re.sub(r"\s+", " ", str(quotes.at[0, "quoteBy"])).strip()
    current_addressees = tuple(
        sorted(
            re.sub(r"\s+", " ", str(addr)).strip()
            for addr in quotes.at[0, "quoteTo_list"]
            if str(addr).strip()
        )
    )
    current_texts = [quotes.at[0, "quoteText"]]
    first_index = quotes.at[0, "turn_index"]
    source_ids = [quotes.at[0, "quoteID"]]

    for row_index in range(1, len(quotes)):
        speaker_value = re.sub(r"\s+", " ", str(quotes.at[row_index, "quoteBy"])).strip()
        addressee_tuple = tuple(
            sorted(
                re.sub(r"\s+", " ", str(addr)).strip()
                for addr in quotes.at[row_index, "quoteTo_list"]
                if str(addr).strip()
            )
        )

        if (speaker_value, addressee_tuple) == (current_speaker, current_addressees):
            current_texts.append(quotes.at[row_index, "quoteText"])
            source_ids.append(quotes.at[row_index, "quoteID"])
            continue

        rows.append(
            {
                "merge_id": merge_id,
                "speaker": current_speaker,
                "addressees": list(current_addressees),
                "text_merged": " ".join(text for text in current_texts if text),
                "first_turn_index": first_index,
                "last_turn_index": quotes.at[row_index - 1, "turn_index"],
                "source_quoteIDs": ",".join(map(str, source_ids)),
            }
        )

        merge_id += 1
        current_speaker = speaker_value
        current_addressees = addressee_tuple
        current_texts = [quotes.at[row_index, "quoteText"]]
        first_index = quotes.at[row_index, "turn_index"]
        source_ids = [quotes.at[row_index, "quoteID"]]

    rows.append(
        {
            "merge_id": merge_id,
            "speaker": current_speaker,
            "addressees": list(current_addressees),
            "text_merged": " ".join(text for text in current_texts if text),
            "first_turn_index": first_index,
            "last_turn_index": quotes.at[len(quotes) - 1, "turn_index"],
            "source_quoteIDs": ",".join(map(str, source_ids)),
        }
    )

    merged = pd.DataFrame(rows)
    return merged.reset_index(drop=True)


def build_pairs_from_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Derive adjacent A→B pairs when B appears in A's addressees.
    Params:
        merged: DataFrame generated by `merge_contiguous_turns`.
    Returns:
        DataFrame of adjacent conversational pairs.
    """

    pairs = []
    for row_index in range(len(merged) - 1):
        speaker_a = merged.at[row_index, "speaker"]
        addressee_set = set(merged.at[row_index, "addressees"] or [])
        speaker_b = merged.at[row_index + 1, "speaker"]
        if speaker_a and speaker_b and speaker_b in addressee_set:
            pairs.append(
                {
                    "A": speaker_a,
                    "B": speaker_b,
                    "i_merge_id": merged.at[row_index, "merge_id"],
                    "j_merge_id": merged.at[row_index + 1, "merge_id"],
                    "i_text": merged.at[row_index, "text_merged"],
                    "j_text": merged.at[row_index + 1, "text_merged"],
                    "i_addressees": "; ".join(merged.at[row_index, "addressees"]),
                }
            )

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df[pairs_df["A"] != pairs_df["B"]].reset_index(drop=True)
    return pairs_df


def build_pairs_from_merged_nonadjacent(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Build skip-one control pairs where B speaks two turns after A.
    Params:
        merged: DataFrame generated by `merge_contiguous_turns`.
    Returns:
        DataFrame of non-adjacent conversational pairs.
    """

    pairs = []
    for row_index in range(len(merged) - 2):
        speaker_a = merged.at[row_index, "speaker"]
        addressee_set = set(merged.at[row_index, "addressees"] or [])
        speaker_b = merged.at[row_index + 2, "speaker"]
        if speaker_a and speaker_b and speaker_b in addressee_set:
            pairs.append(
                {
                    "A": speaker_a,
                    "B": speaker_b,
                    "i_merge_id": merged.at[row_index, "merge_id"],
                    "j_merge_id": merged.at[row_index + 2, "merge_id"],
                    "i_text": merged.at[row_index, "text_merged"],
                    "j_text": merged.at[row_index + 2, "text_merged"],
                    "i_addressees": "; ".join(merged.at[row_index, "addressees"]),
                }
            )

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df[pairs_df["A"] != pairs_df["B"]].reset_index(drop=True)
    return pairs_df


def build_pairs_randomized_from_adjacent(pairs_adj: pd.DataFrame, seed: int = 123 ) -> pd.DataFrame:
    """
    Shuffle replies within each dyad to build a randomized control set.
    Params:
        pairs_adj: Adjacent pair DataFrame from `build_pairs_from_merged`.
        seed: Random state used for deterministic shuffling.
    Returns:
        DataFrame with randomized B-side alignments.
    """

    if pairs_adj.empty:
        return pairs_adj.copy()

    meta_cols_a = [column for column in pairs_adj.columns if column.startswith("A_")]
    meta_cols_b = [column for column in pairs_adj.columns if column.startswith("B_")]

    randomized_blocks = []
    for (speaker_a, speaker_b), group in pairs_adj.groupby(["A", "B"], sort=False, dropna=False):
        group = group.reset_index(drop=True)
        shuffled = group.sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)

        block = pd.DataFrame(
            {
                "A": group["A"],
                "B": group["B"],
                "i_merge_id": group["i_merge_id"],
                "j_merge_id": shuffled["j_merge_id"],
                "i_text": group["i_text"],
                "j_text": shuffled["j_text"],
                "i_addressees": group.get("i_addressees", pd.Series([""] * len(group))),
            }
        )

        for column in meta_cols_a:
            block[column] = group[column].values
        for column in meta_cols_b:
            block[column] = shuffled[column].values if column in shuffled.columns else group[column].values

        randomized_blocks.append(block)

    randomized = pd.concat(randomized_blocks, ignore_index=True)
    randomized = randomized[randomized["A"] != randomized["B"]].reset_index(drop=True)
    return randomized


def attach_metadata_to_pairs(pairs_df: pd.DataFrame, character_info_csv: Path, name_col: str = "Main Name", gender_col: str = "Gender", category_col: str = "Category") -> pd.DataFrame:
    """
    Merge character metadata (gender, category) into each pair.
    Params:
        pairs_df: DataFrame of dialogue pairs to enrich.
        character_info_csv: Path to PDNC character metadata CSV.
        name_col: Column name holding canonical character names.
        gender_col: Column name holding gender labels.
        category_col: Column name holding character category labels.
    Returns:
        Copy of `pairs_df` with A_* and B_* metadata columns if available.
    """

    if pairs_df.empty or not {"A", "B"}.issubset(pairs_df.columns):
        return pairs_df.copy()

    if character_info_csv is None:
        return pairs_df.copy()

    metadata_path = Path(character_info_csv)
    if not metadata_path.exists():
        return pairs_df.copy()

    metadata = pd.read_csv(metadata_path)
    required = {name_col, gender_col, category_col}
    if not required.issubset(metadata.columns):
        return pairs_df.copy()

    metadata = metadata.rename(
        columns={
            name_col: "Name",
            gender_col: "gender",
            category_col: "category",
        }
    )
    metadata = metadata[["Name", "gender", "category"]].copy()

    enriched = pairs_df.copy()
    enriched = enriched.merge(metadata.rename(columns={"Name": "A"}), how="left", on="A")
    enriched = enriched.rename(columns={"gender": "A_gender", "category": "A_category"})
    enriched = enriched.merge(metadata.rename(columns={"Name": "B"}), how="left", on="B")
    enriched = enriched.rename(columns={"gender": "B_gender", "category": "B_category"})

    return enriched
