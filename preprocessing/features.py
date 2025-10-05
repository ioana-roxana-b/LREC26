import spacy
import pandas as pd
from pathlib import Path
from typing import Dict

# load spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.enable_pipe("senter")

# nine families via POS and morphology
def is_article(t) -> bool:
    return t.pos_ == "DET" and t.lemma_.lower() in {"a", "an", "the"}

def is_aux(t) -> bool:
    return t.pos_ == "AUX"

def is_conj(t) -> bool:
    return t.pos_ in {"CCONJ", "SCONJ"}

def is_hi_adv(t) -> bool:
    return t.pos_ == "ADV"

def is_neg(t) -> bool:
    if t.dep_ == "neg":
        return True
    pron_type = t.morph.get("PronType")
    return t.pos_ == "DET" and "Neg" in pron_type

def is_prep(t) -> bool:
    return t.pos_ == "ADP"

def is_pers_pron(t) -> bool:
    return t.pos_ == "PRON" and "Prs" in t.morph.get("PronType")

def is_imp_pron(t) -> bool:
    if t.pos_ in {"PRON", "DET"}:
        pt = set(t.morph.get("PronType"))
        if pt & {"Dem", "Ind", "Tot", "Neg", "Rel"}:
            return True
        if t.pos_ == "PRON":
            person = set(t.morph.get("Person"))
            number = set(t.morph.get("Number"))
            if "3" in person and "Sing" in number:
                return True
    return False

def is_quant(t) -> bool:
    if t.pos_ in {"DET", "PRON"}:
        pt = set(t.morph.get("PronType"))
        if pt & {"Tot", "Ind"}:
            return True
    return False

FNS = {
    "article": is_article,
    "auxverb": is_aux,
    "conj": is_conj,
    "adverb": is_hi_adv,
    "ipron": is_imp_pron,
    "negate": is_neg,
    "ppron": is_pers_pron,
    "prep": is_prep,
    "quant": is_quant,
}

def count_families(text: str) -> Dict[str, int]:
    """Count occurrences of each LIWC-aligned linguistic family in a text using spaCy POS and morphology."""
    counts = {k: 0 for k in FNS}
    n_tokens = 0
    doc = nlp(text if isinstance(text, str) else "")
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        n_tokens += 1
        for name, fn in FNS.items():
            if fn(t):
                counts[name] += 1
    counts["n_tokens"] = n_tokens
    return counts

def process_pairs(pairs_path: Path, out_path: Path) -> pd.DataFrame:
    """Apply family-level feature extraction to dialogue pairs and write results for CAT computation."""
    pairs = pd.read_csv(pairs_path)

    need = ["A", "B", "i_text", "j_text"]
    miss = [c for c in need if c not in pairs.columns]
    if miss:
        raise ValueError(f"Pairs file missing columns: {miss}")

    rows = []
    for _, r in pairs.iterrows():
        ci = count_families(r["i_text"])
        cj = count_families(r["j_text"])

        row = {
            "A": r["A"],
            "B": r["B"],
            "i_text": r["i_text"],
            "j_text": r["j_text"],
        }
        # counts per side
        for fam in FNS.keys():
            row[f"{fam}_i_count"] = ci[fam]
            row[f"{fam}_j_count"] = cj[fam]
            # binary presence flags for CAT style computations
            row[f"{fam}_i_present"] = int(ci[fam] > 0)
            row[f"{fam}_j_present"] = int(cj[fam] > 0)

        row["i_tokens"] = ci["n_tokens"]
        row["j_tokens"] = cj["n_tokens"]

        for extra in ["i_merge_id", "j_merge_id", "i_addressees", "distance", "i_quoteID", "j_quoteID"]:
            if extra in pairs.columns:
                row[extra] = r[extra]

        for c in pairs.columns:
            if c.startswith("A_") or c.startswith("B_"):
                row[c] = r.get(c)

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")
    return out

