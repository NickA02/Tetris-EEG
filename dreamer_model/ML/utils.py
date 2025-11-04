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
    min_size: int = 6,
) -> List[List[str]]:
    if sensor_families is None:
        sensor_families = []
    if feature_tokens is None:
        feature_tokens = ["entropy", "", "da", "ra"]
    if freq_tokens is None:
        freq_tokens = ["theta", "beta", "alpha", "delta", "gamma"]

    sensor_families = list(sensor_families)
    feature_tokens_sorted = sorted(set(feature_tokens), key=len, reverse=True)
    freq_tokens_sorted = sorted(set(freq_tokens), key=len, reverse=True)

    leading_chan_re = re.compile(r"^(?P<prefix>[A-Za-z]+)[0-9A-Za-z]*", re.ASCII)

    def infer_sensor_family(col: str) -> Optional[str]:
        # Try channel-like prefix first (e.g., AF3 -> AF)
        m = leading_chan_re.match(col)
        if m:
            prefix = m.group("prefix").upper()
            for fam in sorted(sensor_families, key=len, reverse=True):
                if prefix.startswith(fam.upper()):
                    return fam
        # Fallback: substring search
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
        parsed[c] = (infer_sensor_family(c), infer_feature(c), infer_freq(c))

    present_fams = sorted({fam for fam, _, _ in parsed.values() if fam is not None})
    present_feats = sorted({feat for _, feat, _ in parsed.values() if feat is not None})
    present_freqs = sorted({fr for _, _, fr in parsed.values() if fr is not None})

    def powerset_with_empty(items: List[str]):
        s = list(items)
        return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))

    fam_subsets = list(powerset_with_empty(present_fams))
    feat_subsets = list(powerset_with_empty(present_feats))
    freq_subsets = list(powerset_with_empty(present_freqs))

    original_order = list(columns)
    all_subsets: List[List[str]] = []
    seen: set = set()

    for fam_sel, feat_sel, freq_sel in product(fam_subsets, feat_subsets, freq_subsets):
        fam_set = set(fam_sel) if len(fam_sel) else None
        feat_set = set(feat_sel) if len(feat_sel) else None
        freq_set = set(freq_sel) if len(freq_sel) else None

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

        if len(selected) >= min_size:
            key = tuple(selected)
            if key not in seen:
                seen.add(key)
                all_subsets.append(selected)

    return all_subsets


import numpy as np
import pandas as pd

# Homologous left–right pairs for Emotiv/DREAMER (14ch)
HOMOLOGOUS_PAIRS = [
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("P7", "P8"),
    ("O1", "O2"),
]


AS_BANDS = {"delta", "theta", "alpha", "beta", "gamma"}
def compute_asymmetry_from_psd(
    psd: pd.DataFrame,
    pairs: list[tuple[str, str]] = HOMOLOGOUS_PAIRS,
    eps: float = 1e-12,
    add_log: bool = True,
    prefix_da: str = "da",
    prefix_ra: str = "ra",
) -> pd.DataFrame:
    bands_present = set()
    for col in psd.columns:
        if "_" in col:
            ch, band = col.rsplit("_", 1)
            if band in AS_BANDS:
                bands_present.add(band)

    out_cols = {}

    for L, R in pairs:
        for band in bands_present:
            cL = f"{L}_{band}"
            cR = f"{R}_{band}"
            if cL not in psd.columns or cR not in psd.columns:
                continue 

            PL = psd[cL].astype(float)
            PR = psd[cR].astype(float)

            if add_log:
                da = np.log(PR + eps) - np.log(PL + eps)
            else:
                da = (PR + eps) - (PL + eps)


            ra = (PR - PL) / (PR + PL + eps)

            out_cols[f"{R}_{L}_{band}_{prefix_da}"] = da
            out_cols[f"{R}_{L}_{band}_{prefix_ra}"] = ra

    return pd.DataFrame(out_cols, index=psd.index)
