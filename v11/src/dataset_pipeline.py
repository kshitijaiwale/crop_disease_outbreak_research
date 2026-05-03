"""
V9 Dataset Pipeline — Red Rot Early Warning System
===================================================
Produces a clean feature matrix only. No labels, no risk_label column.

CHANGES FROM V8
---------------
[CRITICAL] Removed apply_susceptibility_and_labels() labeling block.
           Labels are now assigned exclusively in train.py using GT dates
           and a [peak-10, peak-7] lead-time window matching the 7-day
           spray decision horizon.

           The variety_susceptibility, is_ratoon, and crop_age_days columns
           are still computed here — they are static agronomic features, not
           labels — and are consumed by train.py's agronomic encoder.

           risk_label no longer appears in features.csv. Any downstream
           code that reads risk_label from this file will raise a KeyError,
           which is intentional: it forces explicit labeling in train.py.

[CRITICAL] Removed hardcoded year overrides for variety simulation:
             year_map[2019] = 2
             year_map[2020] = 1
             year_map[2021] = 0
           These manually scripted label outcomes for val years, leaking
           future knowledge into the feature matrix. Variety assignment now
           runs uniformly from the probabilistic model for all years.

[REMOVED]  build_sequences() removed from this pipeline. Sequence building
           is train.py's responsibility using its own WEATHER_FEATURES list.
           The V8 sequence builder used a different feature set (RH2M_mean_3,
           rainfall_sum_3, etc.) incompatible with V11's WEATHER_FEATURES.
           The sequences.npz output was silently wrong.

[REMOVED]  validate_pipeline() assertion: 8 <= years_with_events <= 15
           This range was calibrated to the old synthetic GT. Replaced with
           a feature completeness check that does not encode label assumptions.

OUTPUT
------
  data/processed/v11_features.csv
    Columns: YEAR, DOY, weather (Z-scored), weather_raw, KG-derived features,
             warmup_mask, variety_susceptibility, is_ratoon, crop_age_days.
    No risk_label. No sequences.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "raw_data", "POWER_Point_Daily_20050101_20241231_016d54N_069d78E_LST.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

ROLLING_WINDOW = 365  # days for Z-score normalization (geographic invariance)
WARMUP_DAYS    = 365  # rows masked while rolling stats are unstable


# ---------------------------------------------------------------------------
# Step 1 — Load and clean
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    print("STEP 1: Load and clean")
    df = pd.read_csv(path, skiprows=14)
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )
    df = (df.sort_values("date")
            .drop_duplicates(subset="date")
            .reset_index(drop=True))

    cols = ["date", "YEAR", "DOY", "RH2M", "PRECTOTCORR",
            "T2M", "T2M_MAX", "T2M_MIN", "WS10M"]
    df = df[[c for c in cols if c in df.columns]].copy()

    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=3)

    df = df.set_index("date").asfreq("D").reset_index()
    for col in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear")
    df["PRECTOTCORR"] = df["PRECTOTCORR"].fillna(0)

    print(f"  Loaded {len(df)} rows  |  {df['date'].min().date()} – {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — 90-day rolling Z-score normalization
# ---------------------------------------------------------------------------

def apply_rolling_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each weather column to a 90-day rolling Z-score.
    This makes the model geographically invariant — it responds to
    anomalies relative to the local 3-month baseline, not absolute values.

    Raw values are preserved as _raw columns for interpretability and
    for use in KG feature computation (which needs natural units).

    Warmup mask: the first WARMUP_DAYS rows have fewer than 90 days of
    history, making their Z-scores unreliable. warmup_mask=1 flags these
    rows; train.py drops them before building sequences.
    """
    print("STEP 2: 90-day rolling Z-score normalization")
    weather_cols = ["WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"]

    for col in weather_cols:
        if col not in df.columns:
            continue
        df[f"{col}_raw"] = df[col].copy()
        roll_mean = df[col].rolling(ROLLING_WINDOW, min_periods=1).mean()
        roll_std  = df[col].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(1.0)
        roll_std  = roll_std.replace(0, 1.0)
        df[col]   = (df[col] - roll_mean) / roll_std

    df["warmup_mask"] = 0
    df.loc[df.index < WARMUP_DAYS, "warmup_mask"] = 1
    print(f"  Warmup rows flagged: {df['warmup_mask'].sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 3 — KG-derived biological features (natural units)
# ---------------------------------------------------------------------------

def engineer_kg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Knowledge-Guided features from raw weather values.
    All features here are in natural units — they must NOT be re-scaled.
    causal_consistency_loss thresholds in train.py depend on these units.

    Uses _raw columns so KG features are computed from actual meteorology,
    not from Z-scored values which have no physical interpretation here.
    """
    print("STEP 3: KG feature engineering")

    raw_rh   = df["RH2M_raw"]   if "RH2M_raw"   in df.columns else df["RH2M"]
    raw_rain = df["PRECTOTCORR_raw"] if "PRECTOTCORR_raw" in df.columns else df["PRECTOTCORR"]
    raw_tmin = df["T2M_MIN_raw"] if "T2M_MIN_raw" in df.columns else df["T2M_MIN"]

    # Soft humidity flag: 0 below 75% RH, ramps to 1 at 90% RH.
    # Avoids the hard binary threshold that caused signal collapse in V8.
    df["RH_high_flag"] = np.clip((raw_rh - 75) / 15, 0, 1)

    # Accumulated soft humidity over past 7 days (0–7 natural range).
    # Used in causal_consistency_loss: threshold 3.5 = half the week.
    df["RH_persist_7d"] = df["RH_high_flag"].rolling(7, min_periods=1).sum()

    # Rainfall accumulations in mm (natural units for biological pressure).
    df["Rain_sum_7d"]  = raw_rain.rolling(7,  min_periods=1).sum()
    df["Rain_sum_14d"] = raw_rain.rolling(14, min_periods=1).sum()

    # Monsoon indicator: 1 when 7-day mean RH exceeds 75% (monsoon onset proxy).
    df["Monsoon_ind"] = (raw_rh.rolling(7, min_periods=1).mean() > 75).astype(int)

    # 15-day lagged minimum temperature: captures the cold-dip signal that
    # weakens crop immunity before an outbreak window.
    df["T2M_MIN_lag_15d"] = raw_tmin.shift(15)

    # Latent window features: deviation of recent RH/T from a longer baseline.
    # Captures the "environment changing faster than normal" signal.
    df["RH2M_latent_window"] = raw_rh - raw_rh.rolling(28, min_periods=1).mean()
    df["T2M_latent_window"]  = (df["T2M_MIN_raw"] if "T2M_MIN_raw" in df.columns
                                 else df["T2M_MIN"]) \
                                - (df["T2M_MIN_raw"] if "T2M_MIN_raw" in df.columns
                                   else df["T2M_MIN"]).rolling(28, min_periods=1).mean()

    print(f"  KG features computed: RH_high_flag, RH_persist_7d, Rain_sum_7d, "
          f"Rain_sum_14d, Monsoon_ind, T2M_MIN_lag_15d, RH2M_latent_window, "
          f"T2M_latent_window")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Agronomic features (static, natural units)
# ---------------------------------------------------------------------------

def engineer_agronomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate static agronomic vulnerability features.
    These arrive in natural units and are scaled by StandardScaler in train.py.

    variety_susceptibility: 0=resistant, 1=moderate, 2=susceptible.
      Reflects the historical varietal adoption pattern in Sangli —
      susceptible Co varieties dominated pre-2010; resistant Co-86032
      and CoM varieties increased post-2015.

    NOTE: No year-specific overrides. Variety assignment is probabilistic
    and uniform — val/test years get the same treatment as train years.
    Hardcoded overrides (2019=2, 2020=1, 2021=0) were removed in V9
    because they scripted label outcomes for known GT event years.

    is_ratoon: ratoon crops are more susceptible due to accumulated
      inoculum in stubble. Probability increases with year (older fields).

    crop_age_days: proxy for within-season vulnerability. Grand growth
      phase (120–240 days) has highest susceptibility to Red Rot.
    """
    print("STEP 4: Agronomic feature simulation")

    df["year"] = df["date"].dt.year
    np.random.seed(42)

    def assign_variety(year: int) -> int:
        if year <= 2010:
            return np.random.choice([2, 1], p=[0.8, 0.2])
        elif year <= 2015:
            return np.random.choice([2, 1, 0], p=[0.3, 0.4, 0.3])
        else:
            return np.random.choice([1, 0], p=[0.4, 0.6])

    year_variety_map = {y: assign_variety(y) for y in sorted(df["year"].unique())}
    df["variety_susceptibility"] = df["year"].map(year_variety_map)

    # Ratoon probability increases slightly over time as older fields accumulate
    ratoon_probs = {y: min(0.2 + (y - 2005) * 0.01, 0.45)
                    for y in df["year"].unique()}
    df["is_ratoon"] = df["year"].map(
        {y: int(np.random.rand() < p) for y, p in ratoon_probs.items()}
    )

    # Crop age: synthetic within-season day counter peaking mid-season
    df["doy"] = df["date"].dt.dayofyear
    df["crop_age_days"] = ((df["doy"] - 60) % 365).clip(0, 365)

    df.drop(columns=["year", "doy"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Step 5 — Validate and save (feature completeness only, no label checks)
# ---------------------------------------------------------------------------

def validate_and_save(df: pd.DataFrame) -> None:
    print("\nSTEP 5: Validate and save")

    # Feature completeness check
    expected_cols = [
        "WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR",
        "T2M_MIN_lag_15d", "RH_high_flag", "RH_persist_7d",
        "Rain_sum_7d", "Rain_sum_14d", "Monsoon_ind",
        "RH2M_latent_window", "T2M_latent_window",
        "warmup_mask", "variety_susceptibility", "is_ratoon", "crop_age_days",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise AssertionError(f"FAIL: Missing expected columns: {missing}")

    # No risk_label check — labels must NOT be in this file
    if "risk_label" in df.columns:
        raise AssertionError(
            "FAIL: risk_label found in features.csv. "
            "Labeling must happen in train.py, not the pipeline."
        )

    # NaN audit
    nan_counts = df[expected_cols].isna().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print("\n  NaN summary (expected in warmup rows):")
        for col, n in nan_cols.items():
            warmup_nan = df[df["warmup_mask"] == 1][col].isna().sum()
            post_warmup_nan = n - warmup_nan
            status = "OK" if post_warmup_nan == 0 else "WARN"
            print(f"    [{status}] {col}: {n} NaN total "
                  f"({warmup_nan} in warmup, {post_warmup_nan} post-warmup)")

    out_path = os.path.join(PROCESSED_DIR, "v11_features.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  Saved: {out_path}")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Warmup rows (excluded from training): {df['warmup_mask'].sum()}")
    print(f"  Usable rows: {(df['warmup_mask'] == 0).sum()}")
    print(f"  No risk_label column — labeling is train.py's responsibility.")
    print("\n[PASS] Pipeline validation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline():
    df = load_and_clean_data(RAW_DATA_PATH)
    df = apply_rolling_zscore(df)
    df = engineer_kg_features(df)
    df = engineer_agronomic_features(df)
    validate_and_save(df)


if __name__ == "__main__":
    run_pipeline()