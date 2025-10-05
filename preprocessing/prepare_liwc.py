import pandas as pd

def make_liwc_input(pairs_csv, liwc_input_csv) -> None:
    """
    Expand dialogue pairs into LIWC-ready A and B utterances.
    Params:
        pairs_csv: Path to pairs CSV containing i_text and j_text columns.
        liwc_input_csv: Destination path for the flattened LIWC input CSV.
    Returns:
        None
    """
    df = pd.read_csv(pairs_csv)

    need = {"i_text", "j_text"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"pairs file is missing columns: {sorted(missing)}")

    # A side
    a = df[["i_text"]].rename(columns={"i_text": "text"})
    a["pair_id"] = df.index.astype(str).str.zfill(5)
    a["side"] = "A"
    a["uid"] = "p" + a["pair_id"] + "_A"

    # B side
    b = df[["j_text"]].rename(columns={"j_text": "text"})
    b["pair_id"] = df.index.astype(str).str.zfill(5)
    b["side"] = "B"
    b["uid"] = "p" + b["pair_id"] + "_B"

    out = pd.concat(
        [a[["uid", "pair_id", "side", "text"]],
         b[["uid", "pair_id", "side", "text"]]],
        ignore_index=True
    )
    out.to_csv(liwc_input_csv, index=False)
    print(f"Saved LIWC input: {liwc_input_csv} rows={len(out)}")
