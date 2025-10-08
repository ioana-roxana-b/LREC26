import pandas as pd
import re

def clean_text(text: str, lowercase: bool = False) -> str:
    """
    Clean an individual quote string before downstream processing.
    Params:
        text: Raw quote text as pulled from the CSV.
        lowercase: Whether to lowercase the cleaned string.
    Returns:
        Cleaned text ready for LIWC or spaCy input.
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()

    # Remove all punctuation and symbols
    t = re.sub(r"[^A-Za-z0-9\s]", " ", t)

    # Normalize multiple spaces
    t = re.sub(r"\s+", " ", t)

    # Optional lowercase
    if lowercase:
        t = t.lower()

    return t.strip()


def preprocess_quotes(path, out_path, lowercase=False) -> pd.DataFrame:
    """
    Prepare the raw quotes export for building dialogue pairs.
    Params:
        path: Path to `quotation_info.csv` (cleaned from PDNC extraction).
        out_path: Destination path for the cleaned quotes CSV.
        lowercase: Flag controlling optional quote lowercasing.
    Returns:
        DataFrame containing the cleaned and standardized quotes.
    """
    df = pd.read_csv(path)

    # Rename speaker/addressees to standardized names
    df = df.rename(columns={"speaker": "quoteBy", "addressees": "quoteTo"})
    keep_cols = ["quoteID", "quoteText", "quoteBy", "quoteTo"]
    df = df[keep_cols].copy()

    # Clean text
    df["quoteText"] = df["quoteText"].astype(str).apply(lambda x: clean_text(x, lowercase=lowercase))

    # Drop rows in case quoteText is empty after cleaning
    df = df[df["quoteText"].str.len() > 0].reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved cleaned quotes to {out_path} ({len(df)} rows)")

    return df


