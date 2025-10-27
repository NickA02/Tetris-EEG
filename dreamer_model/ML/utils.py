from __future__ import annotations
from typing import Iterable, List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from itertools import chain, combinations, product
import re


def read_table(filename: str = "datasets/features_table.csv") -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir.parent / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)
    return df


def generate_all_subsets(
    columns: Iterable[str],
    *,
    sensor_families: Optional[Iterable[str]] = None,
    feature_tokens: Optional[Iterable[str]] = None,
    freq_tokens: Optional[Iterable[str]] = None,
) -> List[List[str]]:
    if sensor_families is None:
        sensor_families = ["AF", "F", "FC", "C", "CP", "P", "PO", "O", "T", "FT", "TP"]
    if feature_tokens is None:
        feature_tokens = [
            "imfentropy",
            "imfenergy",
            "wenergy",
            "wentropy",
            "mobility",
            "activity",
            "complexity",
            "psd",
            "entropy",
        ]
    if freq_tokens is None:
        freq_tokens = [
            "betaH",
            "betaL",
            "theta",
            "alpha",
            # "delta",
            # "gamma",
        ]

    sensor_families = list(sensor_families)
    feature_tokens_sorted = sorted(set(feature_tokens), key=len, reverse=True)
    freq_tokens_sorted = sorted(set(freq_tokens), key=len, reverse=True)

    leading_chan_re = re.compile(r"^(?P<prefix>[A-Za-z]+)[0-9A-Za-z]*", re.ASCII)

    def infer_sensor_family(col: str) -> Optional[str]:
        m = leading_chan_re.match(col)
        if m:
            prefix = m.group("prefix").upper()
            for fam in sorted(sensor_families, key=len, reverse=True):
                if prefix.startswith(fam.upper()):
                    return fam

        up = col.upper()
        for fam in sorted(sensor_families, key=len, reverse=True):
            if fam.upper() in up:
                return fam
        return None

    def infer_freq(col: str) -> Optional[str]:
        low = col.lower()
        for f in freq_tokens_sorted:
            if f.lower() in low:
                return f
        return None

    def infer_feature(col: str) -> str:
        low = col.lower()
        for tok in feature_tokens_sorted:
            if tok.lower() in low:
                return tok
        return "psd"

    parsed: Dict[str, Tuple[Optional[str], str, Optional[str]]] = {}
    for c in columns:
        fam = infer_sensor_family(c)
        feat = infer_feature(c)
        freq = infer_freq(c)
        parsed[c] = (fam, feat, freq)

    present_fams = sorted({fam for fam, _, _ in parsed.values() if fam is not None})
    present_feats = sorted({feat for _, feat, _ in parsed.values() if feat is not None})
    present_freqs = sorted({fr for _, _, fr in parsed.values() if fr is not None})

    def powerset_nonempty(items: List[str]) -> List[Tuple[str, ...]]:
        s = list(items)
        return list(
            chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
        )

    fam_subsets = powerset_nonempty(present_fams) if present_fams else [()]
    feat_subsets = powerset_nonempty(present_feats) if present_feats else [()]
    freq_subsets = powerset_nonempty(present_freqs) if present_freqs else [()]

    all_subsets: List[List[str]] = []
    original_order = list(columns)

    for fam_sel, feat_sel, freq_sel in product(fam_subsets, feat_subsets, freq_subsets):
        fam_set = set(fam_sel) if fam_sel else None
        feat_set = set(feat_sel) if feat_sel else None
        freq_set = set(freq_sel) if freq_sel else None

        selected = []
        for c in original_order:
            fam, feat, fr = parsed[c]
            if fam_set is not None and fam not in fam_set:
                continue
            if feat_set is not None and feat not in feat_set:
                continue
            if freq_set is not None and fr not in freq_set:
                continue
            selected.append(c)

        if selected:
            all_subsets.append(selected)

    return all_subsets


if __name__ == "__main__":
    cols = [
        "AF3_gamma_wenergy",
        "AF3_betaH_wenergy",
        "AF3_alpha",
        "F7_theta_imfentropy",
        "F7_beta",
        "O1_betaL_wenergy",
        "TP7_theta",
    ]
    subsets = generate_all_subsets(cols)

    seen = set()
    unique = []
    for s in subsets:
        key = tuple(s)        # order-preserving key
        if key not in seen:
            seen.add(key)
            unique.append(s)


    print(f"Generated {len(unique)} unique")
    print(unique)
