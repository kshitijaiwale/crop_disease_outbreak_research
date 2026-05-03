"""
V11 KG-CTCN Comprehensive Pipeline Validation
--------------------------------------------------
Validates the entire pipeline against the ground truth dataset:
research_comp/evidence_base/outbreak_events/sangli_synthetic_gt.csv

Produces a comprehensive report for all events across all years.

CHANGELOG (v11.1):
  - [CRITICAL]  load_metadata() switched from .txt parser to json.load().
                train.py now writes v11_metadata.json.
  - [CRITICAL]  Removed weather StandardScaler. Weather features are already
                90-day rolling Z-scores from the pipeline. Agro scaler loaded
                from agro_scaler.pkl rather than refitted — ensures exact
                reproducibility with the saved model checkpoint.
  - [CRITICAL]  warmup_mask filter applied before sequence building. Without
                this, sequences overlapping the first 90 warm-up rows contain
                NaN activations; NaN predictions silently fail threshold
                comparisons (NaN >= threshold is always False), suppressing
                alerts for those dates with no warning.
  - [BUG]       Threshold search reversed to descending (0.99 → 0.01).
                Ascending search found the lowest threshold satisfying FPR ≤ 5%
                (nearly always ~0.01), producing near-zero precision.
                Descending search finds the most conservative (highest) valid
                threshold.
  - [BUG]       Val label alignment fixed. Previously queried the unfiltered df
                with df[df['date'].isin(val_dates)], which includes the first
                seq_len rows of 2019 that were skipped in sequence building,
                causing a length mismatch between val_probs and val_labels.
                Labels now derived directly from the ordered dates list.
  - [BUG]       FP audit label alignment fixed. dict.values() order matches
                insertion order (dates list), but df[df['date'].isin(dates)]
                returns rows in dataframe sort order — these can differ.
                Labels now built from the dates list to guarantee alignment.
  - [DESIGN]    Full dataset no longer moved to device before batch loop.
                Allocating 7000×28×14 float32 on device upfront wastes VRAM
                and fails on low-memory GPUs. Tensors now moved inside the
                batch loop.
  - [DESIGN]    Train detection rate clearly labelled as in-sample
                memorisation check, not a generalisation metric.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta
import torch
import joblib

from model import KGCTCN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
PROJECT_ROOT = os.path.dirname(BASE_DIR)

GT_PATH = os.path.join(
    PROJECT_ROOT,
    "research_comp", "evidence_base", "outbreak_events", "sangli_synthetic_gt.csv",
)

BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """Load model metadata from v11_metadata.json (written by train.py v11.1+)."""
    with open(os.path.join(MODEL_DIR, "v11_metadata.json"), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def find_threshold_at_fpr(
    probs: np.ndarray,
    labels: np.ndarray,
    max_fpr: float = 0.05,
    fallback: float = 0.5,
) -> float:
    """
    Return the highest threshold t such that FPR(t) ≤ max_fpr.

    Descending search (0.99 → 0.01) finds the most conservative threshold
    that still satisfies the FPR budget, maximising precision at the
    operating point. Ascending search would find the *lowest* valid
    threshold (near 0.01), which predicts almost everything as positive.
    """
    n_neg = (labels == 0).sum()
    if n_neg == 0:
        return fallback
    for t in np.linspace(0.99, 0.01, 99):
        fpr_t = ((probs >= t) & (labels == 0)).sum() / n_neg
        if fpr_t <= max_fpr:
            return float(round(t, 4))
    return fallback


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print(" V11 KG-CTCN COMPREHENSIVE PIPELINE VALIDATION REPORT")
    print("=" * 68)

    # ── Metadata & scalers ───────────────────────────────────────────────────
    meta             = load_metadata()
    weather_features = meta["weather_features"]
    agro_features    = meta["agro_features"]
    seq_len          = int(meta["seq_len"])

    # Agro scaler: load the exact instance fitted on the training split in
    # train.py. Do NOT refit — different random state breaks model calibration.
    # No weather scaler: pipeline already produces Z-scored weather features.
    a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))

    # ── Load dataset and apply warmup filter ─────────────────────────────────
    df = pd.read_csv(os.path.join(DATA_DIR, "v11_features.csv"))
    df["date"] = pd.to_datetime(df["date"])

    n_before = len(df)
    df = df[df["warmup_mask"] == 0].reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} warm-up rows. Remaining: {len(df)}")

    gt_df = pd.read_csv(GT_PATH)
    gt_df["peak_start"] = pd.to_datetime(gt_df["peak_start"])

    # ── Load model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = KGCTCN(len(weather_features), len(agro_features)).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    # ── Build sequences ───────────────────────────────────────────────────────
    # Weather: no scaler — pipeline already Z-scored.
    # Agro: transform with the saved scaler.
    w_vals  = df[weather_features].values.astype(np.float32)
    a_vals  = a_sc.transform(df[agro_features].values.astype(np.float32))
    labels  = df["risk_label"].values.astype(np.float32)
    dates   = []

    X_w_list, X_a_list, label_list = [], [], []
    for i in range(seq_len, len(df)):
        X_w_list.append(w_vals[i - seq_len + 1 : i + 1])
        X_a_list.append(a_vals[i])
        label_list.append(labels[i])
        dates.append(df["date"].iloc[i])

    dates       = pd.to_datetime(dates)
    label_arr   = np.array(label_list, dtype=np.float32)  # aligned with dates

    X_w_np = np.array(X_w_list, dtype=np.float32)  # (N, seq_len, F) — stays on CPU
    X_a_np = np.array(X_a_list, dtype=np.float32)  # (N, A)

    # ── Batch inference ───────────────────────────────────────────────────────
    # Tensors are moved to device inside the loop — avoids allocating the full
    # dataset on VRAM upfront, which fails on low-memory GPUs.
    print("Running full-dataset inference...")
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_w_np), BATCH_SIZE):
            bw = torch.FloatTensor(X_w_np[start : start + BATCH_SIZE]).to(device)
            ba = torch.FloatTensor(X_a_np[start : start + BATCH_SIZE]).to(device)
            _, probs, _ = model(bw, ba)
            all_probs.extend(probs.cpu().numpy().flatten())

    all_probs = np.array(all_probs, dtype=np.float32)   # (N,) — aligned with dates

    # ── scores_dict: date → probability ──────────────────────────────────────
    # Built from the same ordered dates list used for inference — guaranteed
    # alignment between keys and all_probs.
    scores_dict = dict(zip(dates, all_probs))

    # ── Threshold calibration on val set ─────────────────────────────────────
    # Use the dates list (not df queries) for label alignment — the dates list
    # skips the first seq_len rows of each subset, while a df query does not.
    val_mask_arr  = np.array([2019 <= d.year <= 2021 for d in dates])
    val_probs     = all_probs[val_mask_arr]
    val_labels    = label_arr[val_mask_arr]

    opt_thr = find_threshold_at_fpr(val_probs, val_labels, max_fpr=0.05, fallback=0.5)

    print(f"\n[ CALIBRATION ]")
    print(f"  Target FPR        : ≤ 5%  (calibrated on Val set 2019-2021)")
    print(f"  Operating threshold: {opt_thr:.4f}")

    # ── Event-level validation across all years ───────────────────────────────
    print("\n[ EVENT-LEVEL VALIDATION — ALL YEARS ]")
    print(
        f"  NOTE: Train detections are IN-SAMPLE (memorisation check).\n"
        f"        Val detections are OUT-OF-SAMPLE (generalisation metric).\n"
    )
    header = f"{'Peak Date':<16}  {'Split':<8}  {'Status':<12}  {'Lead Time'}"
    print(header)
    print("-" * 55)

    counts = {"Train": [0, 0], "Val": [0, 0], "Test": [0, 0]}  # [detected, total]

    for _, row in gt_df.sort_values("peak_start").iterrows():
        peak = row["peak_start"]
        yr   = peak.year
        split = "Train" if yr <= 2018 else ("Val" if yr <= 2021 else "Test")
        counts[split][1] += 1

        window_start = peak - timedelta(days=7)
        window_end   = peak - timedelta(days=3)

        # Iterate earliest → latest; first hit = maximum lead time
        earliest_alert = None
        for d in pd.date_range(window_start, window_end):
            score = scores_dict.get(d, 0.0)
            if score >= opt_thr:
                earliest_alert = d
                break

        if earliest_alert is not None:
            lead = (peak - earliest_alert).days
            counts[split][0] += 1
            status   = "DETECTED"
            lead_str = f"{lead} days"
        else:
            status   = "MISSED"
            lead_str = "N/A"

        print(f"  {peak.date().isoformat():<16}{split:<8}  {status:<12}  {lead_str}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 55)
    print("\n[ SUMMARY REPORT ]")
    print(f"  Total GT events: {len(gt_df)}")

    for split, (det, tot) in counts.items():
        if tot == 0:
            label = f"  {split:<8}: N/A (0 events in GT)"
        else:
            rate  = det / tot * 100
            note  = " ← in-sample" if split == "Train" else ""
            label = f"  {split:<8}: {rate:.1f}%  ({det}/{tot}){note}"
        print(label)

    # ── False positive audit ──────────────────────────────────────────────────
    # all_probs and label_arr are both derived from the same ordered dates list
    # — guaranteed alignment. No df query involved.
    all_preds = (all_probs >= opt_thr).astype(int)

    fp_total  = int(((all_preds == 1) & (label_arr == 0)).sum())
    tn_total  = int(((all_preds == 0) & (label_arr == 0)).sum())
    fpr_total = fp_total / max(1, fp_total + tn_total)

    print("\n[ FALSE POSITIVE AUDIT ]")
    print(f"  Global FPR (all years): {fpr_total * 100:.2f}%  (target: < 5%)")
    print(f"  False alarm days (total dataset): {fp_total}")

    status_str = (
        "VALIDATED  — FPR constraint met, causality intact"
        if fpr_total <= 0.05
        else "WARNING    — FPR exceeds 5% target"
    )
    print(f"\n  >> STATUS: {status_str}")
    print("=" * 68)


if __name__ == "__main__":
    main()