import numpy as np
import pandas as pd
import math
from scipy.stats import zscore, kurtosis
from typing import Dict, Any, Optional, Tuple

import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from mne_icalabel import label_components


def ica_clean_eeg_df(
    df: pd.DataFrame,
    fs: float,
    *,
    ch_names: Optional[list] = None,
    montage: Optional[str] = "standard_1020",
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    n_components: Optional[int] = None,
    method: str = "fastica",  # "fastica", "picard", "infomax", "extended-infomax"
    random_state: int = 97,
    use_icalabel_if_available: bool = True,
    blink_ratio_thr: float = 0.35,
    muscle_ratio_thr: float = 0.30,
    kurtosis_z_thr: float = 2.5,
    var_z_thr: float = 3.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run bandpass -> ICA -> auto-detect artifact ICs -> remove -> return cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Columns are EEG channels, rows are time samples.
    fs : float
        Sampling frequency (Hz).
    ch_names : list, optional
        Channel names; by default uses df.columns.
    montage : str or None
        Name of an MNE montage (e.g., "standard_1020") or None to skip setting.
    l_freq, h_freq : float
        Bandpass for ICA fitting (data returned is filtered only if df was—see note).
    n_components : int or None
        Number of ICA components. None -> min(n_channels, 0.99*rank).
    method : str
        ICA solver.
    random_state : int
        Reproducibility.
    reject_by_annotation : bool
        Respect raw annotations during ICA.apply (kept True by default).
    use_icalabel_if_available : bool
        If mne_icalabel is installed, use it to classify components.
    blink_ratio_thr, muscle_ratio_thr, kurtosis_z_thr, var_z_thr : float
        Heuristic thresholds when ICLabel is unavailable.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Same shape/columns as input, with artifact ICs removed via ICA.
    report : dict
        Keys: 'excluded_idx', 'n_components', 'ica_method', 'metrics', 'strategy', 'notes'.
        'metrics' gives per-component dict with spectral ratios, kurtosis, var_z, label (if any).

    Notes
    -----
    * No EOG/ECG reference is used. Auto-detection is approximate.
    * If `mne_icalabel` is present, labels like 'eye blink', 'muscle', 'heart', 'brain', etc.
      are used to choose components to reject. Otherwise simple spectral/kurtosis heuristics apply.
    """
    if ch_names is None:
        ch_names = list(df.columns)
    df_num = df.apply(pd.to_numeric, errors="coerce")
    in_uV = True  # set False if your CSV is already in Volts
    to_V = 1e-6 if in_uV else 1.0
    data_V = df_num.to_numpy(np.float64).T * to_V

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data_V, info, verbose=False)

    raw.set_montage(montage, on_missing="ignore")
    mne.set_eeg_reference(raw, ref_channels="average", projection=False, verbose=False)

    # Filter and ICA fitting
    h_label = min(100.0, 0.5 * fs - 1.0)  # ≤100 Hz and below Nyquist
    raw_filt = raw.copy().filter(
        l_freq=1.0, h_freq=h_label, fir_design="firwin", verbose=False
    )

    ica = ICA(
        method=method,
        n_components=n_components,
        random_state=random_state,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw_filt, verbose=False)

    # IMPORTANT: ICLabel on the same CAR + 1–100 Hz object you fit with:
    labels, proba = label_components(raw_filt, ica, method="iclabel")

    src = ica.get_sources(raw_filt).get_data()  # shape: (n_components, n_times)
    n_comp = src.shape[0]
    metrics = []

    bands = {
        "blink_low": (0.5, 3.0),
        "alpha": (8.0, 13.0),
        "muscle": (25.0, 45.0),
        "total": (0.5, 45.0),
    }

    def band_power(x, fs, fmin, fmax):
        target_secs = 4.0
        n_per_seg = int(min(len(x), max(int(fs * target_secs), 64)))
        n_fft = 1 << int(math.ceil(math.log2(max(n_per_seg, 64))))
        # overlap < n_per_seg
        n_overlap = int(0.5 * n_per_seg) if n_per_seg > 1 else 0

        # clamp band to valid range
        fmin = max(0.0, float(fmin))
        fmax = min(float(fmax), fs / 2.0)

        psd, freqs = psd_array_welch(
            x[np.newaxis, :],
            sfreq=fs,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            verbose=False,
        )
        return float(psd.sum())

    variances = np.var(src, axis=1)
    var_z = zscore(variances)
    for k in range(n_comp):
        x = src[k, :]
        bp_total = band_power(x, fs, *bands["total"])
        bp_blink = band_power(x, fs, *bands["blink_low"])
        bp_muscle = band_power(x, fs, *bands["muscle"])
        blink_ratio = float(bp_blink / (bp_total + 1e-12))
        muscle_ratio = float(bp_muscle / (bp_total + 1e-12))
        kurt = float(kurtosis(x, fisher=True, bias=False))
        metrics.append(
            {
                "idx": k,
                "blink_ratio_0_2Hz": blink_ratio,
                "muscle_ratio_20_40Hz": muscle_ratio,
                "kurtosis": kurt,
                "var_z": float(var_z[k]),
                "label": None,  # may be filled by ICLabel below
                "reject_reason": None,
            }
        )

    excluded = set()
    strategy = "heuristic"
    notes = []

    if use_icalabel_if_available:
        try:
            strategy = "ICLabel"
            cat_idx = {
                c: i
                for i, c in enumerate(
                    [
                        "brain",
                        "muscle",
                        "eye",
                        "heart",
                        "line_noise",
                        "channel_noise",
                        "other",
                    ]
                )
            }

            for k in range(n_comp):
                lbl = labels[k]
                p = proba[k]
                metrics[k]["label"] = lbl
                metrics[k]["proba"] = {
                    name: float(p[cat_idx[name]]) for name in cat_idx
                }

                if lbl in {"eye", "muscle", "heart", "line_noise", "channel_noise"}:
                    if metrics[k]["proba"][lbl] >= 0.90:
                        excluded.add(k)
                        metrics[k][
                            "reject_reason"
                        ] = f"ICLabel:{lbl} (p={metrics[k]['proba'][lbl]:.2f})"
        except Exception as e:
            notes.append(
                f"ICLabel unavailable or failed ({e}); falling back to heuristics."
            )
            strategy = "heuristic"

    if strategy == "heuristic":
        for m in metrics:
            k = m["idx"]
            reasons = []
            if (
                m["blink_ratio_0_2Hz"] >= blink_ratio_thr
                and abs(m["kurtosis"]) >= kurtosis_z_thr
            ):
                reasons.append("blink/slow (high 0-2Hz + high kurtosis)")
            if m["muscle_ratio_20_40Hz"] >= muscle_ratio_thr:
                reasons.append("muscle (high 20-40Hz)")
            if abs(m["var_z"]) >= var_z_thr:
                reasons.append("outlier variance")
            if reasons:
                excluded.add(k)
                m["reject_reason"] = "; ".join(reasons)

    excluded_idx = sorted(excluded)

    max_frac = 0.5  # at most 50% ICs removed
    max_k = max(1, int(max_frac * n_comp))

    if len(excluded_idx) > max_k:
        # keep only the most confident artifact ICs (ICLabel prob or heuristic score)
        def score(m):
            if m.get("proba"):
                return max(
                    m["proba"].get("eye", 0),
                    m["proba"].get("muscle", 0),
                    m["proba"].get("heart", 0),
                    m["proba"].get("line_noise", 0),
                    m["proba"].get("channel_noise", 0),
                )
            return max(
                m["blink_ratio_0_2Hz"], m["muscle_ratio_20_40Hz"], abs(m["var_z"]) / 5.0
            )

        top = sorted(
            [mm for mm in metrics if mm["idx"] in excluded_idx], key=score, reverse=True
        )[:max_k]
        excluded_idx = sorted([mm["idx"] for mm in top])

    raw_clean = raw.copy()
    if excluded_idx:
        ica.exclude = excluded_idx
        ica.apply(raw_clean, verbose=False)

    # gentle post-ICA filter to clean DC/drift; keeps physiology
    raw_clean.filter(l_freq=0.5, h_freq=45.0, fir_design="firwin", verbose=False)

    # Convert back to µV exactly once
    from_V = 1e6 if in_uV else 1.0
    cleaned_uV = (raw_clean.get_data() * from_V).T
    cleaned_df = pd.DataFrame(
        cleaned_uV, columns=ch_names, index=df.index[: len(cleaned_uV)]
    )

    report = {
        "excluded_idx": excluded_idx,
        "n_components": int(n_comp),
        "ica_method": method,
        "strategy": strategy,
        "thresholds": {
            "blink_ratio_thr": blink_ratio_thr,
            "muscle_ratio_thr": muscle_ratio_thr,
            "kurtosis_z_thr": kurtosis_z_thr,
            "var_z_thr": var_z_thr,
        },
        "metrics": metrics,
        "notes": notes,
    }
    return cleaned_df, report


if __name__ == "__main__":
    FS = 128
    csv_path = "dreamer_model/datasets/DREAMER.csv"
    dreamer_df = pd.read_csv(csv_path)
    ch_names = [
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
    ]

    patients = dreamer_df["patient_index"]
    videos = dreamer_df["video_index"]
    del dreamer_df["arousal"]
    del dreamer_df["valence"]

    for patient_id in dreamer_df["patient_index"].unique():
        for video_id in dreamer_df["video_index"].unique():
            mask = (dreamer_df["patient_index"] == patient_id) & (
                dreamer_df["video_index"] == video_id
            )
            eeg_df = dreamer_df.loc[mask, :].copy().reset_index(drop=True)
            del eeg_df["patient_index"]
            del eeg_df["video_index"]

            clean_df, rep = ica_clean_eeg_df(
                eeg_df, fs=FS, ch_names=ch_names, method="fastica"
            )
            print("Removed components:", rep["excluded_idx"], "via", rep["strategy"])
            print(clean_df)
            exit()
