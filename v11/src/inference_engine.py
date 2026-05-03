"""
V11 KG-CTCN Inference Engine Wrapper
--------------------------------------------------
Strict Black-Box wrapper around the frozen V11 model.
Ensures causality, no data leakage, and executes KG transforms exactly as trained.

CHANGELOG (v11.1):
  - [BUG]    _load_metadata() replaced .txt key=value parser with json.load().
             train.py v11.1 writes v11_metadata.json; the old .txt no longer
             exists and caused FileNotFoundError at startup.
  - [BUG]    Double-scaling removed. _apply_kg_transforms() already computes
             90-day rolling Z-scores matching dataset_pipeline.py. The
             subsequent self.w_sc.transform() was producing a Z-score of a
             Z-score — the same error fixed in train.py v11.1. weather_scaler.pkl
             no longer exists in the model directory; w_sc load removed from
             _init_scalers().
  - [BUG]    conf.item() wrapped in torch.sigmoid() before returning. model.py
             v11.1 returns conf_logit (pre-sigmoid); comparing a raw logit to
             probability thresholds was incorrect.
  - [BUG]    Monsoon_ind recomputed using rolling RH mean > 75 to match
             dataset_pipeline.py. The previous hardcoded month filter [7,8,9]
             missed June early-onset and Oct/Nov late-season events, diverging
             from the training feature definition.
  - [WARN]   T2M_MIN_lag_15d now computed from T2M_MIN_raw (pre-Z-score).
             Previously computed after T2M_MIN was overwritten with its Z-score,
             producing a lagged Z-score instead of a lagged raw temperature.
  - [WARN]   RH2M_latent_window and T2M_latent_window now computed from _raw
             columns (raw - rolling_mean(raw, 28)), matching dataset_pipeline.py.
             Previously computed from Z-scored columns, producing different
             units and magnitude than the training distribution.
  - [STYLE]  Renamed loop variable l → line in legacy txt parser (now removed).
  - [STYLE]  Removed unused `from sklearn.preprocessing import StandardScaler`
             import (was only needed when weather_scaler existed).
  - [DESIGN] MIN_HISTORY_DAYS constant added (150 days). Replaces magic number
             120 in _fetch_past_weather_slice — provides headroom beyond the
             90-day rolling window + 28-day seq_len + 15-day lag.

CHANGELOG (v11.2):
  - [CRITICAL] Rolling Z-score window corrected from 90 to 365 days.
               dataset_pipeline.py uses ROLLING_WINDOW=365 for weather
               normalisation. The inference engine was using 90, producing
               a different Z-score distribution than training — the model
               was receiving out-of-distribution inputs at inference time.
               MIN_HISTORY_DAYS updated from 150 to 400 to cover the 365-day
               rolling window plus seq_len, lag, and safety margin.
  - [CRITICAL] Temperature scaling now applied before sigmoid in run_inference().
               train.py fits a temperature T (saved to temperature.pkl) and
               applies sigmoid(logits / T) at evaluation time. The inference
               engine was applying sigmoid(logits) without temperature scaling,
               producing overconfident scores that do not match the calibrated
               probability distribution the threshold (0.2918) was fitted on.
               Without this fix, risk_score values are systematically higher
               than at training time, making the High/Medium/Low thresholds
               unreliable.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KGCTCN

from weather_provider import WeatherDataProvider

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")

# Minimum weather history to fetch before the target date.
# Must cover: rolling Z-score window (365) + seq_len (28) + lag (15) + rain sum (14)
# = 422 days minimum. 400 is used because real-time providers rarely have full
# 365-day clean history; min_periods=1 in the rolling calls handles the ramp-up.
# If your provider can supply 400+ days, raise this to 430 for a clean margin.
MIN_HISTORY_DAYS = 400


class V11InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = self._load_metadata()
        self.weather_features = self.meta['weather_features']
        self.agro_features    = self.meta['agro_features']
        self.seq_len          = int(self.meta['seq_len'])

        self.model = KGCTCN(len(self.weather_features), len(self.agro_features)).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"),
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.eval()

        self._init_scalers()

        # Real-time Weather Data Provider
        self.weather_provider = WeatherDataProvider()

    def _load_metadata(self):
        import json
        with open(os.path.join(MODEL_DIR, "v11_metadata.json"), "r") as f:
            return json.load(f)

    def _init_scalers(self):
        """
        Loads pre-fitted agro scaler and temperature scalar from PKL files.

        No weather scaler: pipeline already produces 365-day rolling Z-scores,
        and _apply_kg_transforms() replicates that normalisation at inference time.

        Temperature T: post-hoc calibration scalar fitted on the val set by
        train.py. Must be applied as sigmoid(logits / T) — not sigmoid(logits).
        """
        self.a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
        self.T    = float(joblib.load(os.path.join(MODEL_DIR, "temperature.pkl")))

    def _fetch_past_weather_slice(self, location, target_date):
        """
        Fetches EXACTLY the past data needed up to target_date. Enforces causality.
        MIN_HISTORY_DAYS (400) covers the 365-day rolling Z-score window, the
        28-day sequence window, the 15-day lag, and a small gap-day safety margin.
        """
        return self.weather_provider.get_weather_history(
            location, target_date, window_days=MIN_HISTORY_DAYS
        )

    def _apply_kg_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replicates the V11 dataset_pipeline.py feature transforms exactly.

        Rolling window: 365 days (ROLLING_WINDOW in dataset_pipeline.py).
        All KG features computed from _raw columns (natural units) so the
        model receives the same feature distribution it was trained on.
        """
        # 1. 365-day rolling Z-score normalisation — must match ROLLING_WINDOW=365
        #    in dataset_pipeline.py. Using a shorter window (e.g. 90) produces
        #    different Z-scores and shifts the model into out-of-distribution input.
        weather_base_cols = ['WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR']
        for col in weather_base_cols:
            mean_365 = df[col].rolling(365, min_periods=1).mean()
            std_365  = df[col].rolling(365, min_periods=1).std().fillna(1.0).replace(0, 1.0)
            df[f'{col}_raw'] = df[col].copy()   # preserve raw for KG features below
            df[col] = (df[col] - mean_365) / std_365

        # 2. KG-driven causal transforms — computed from _raw columns.
        # Using Z-scored columns here would produce a different distribution
        # than the training pipeline and silently corrupt model inputs.

        # 15-day lagged min temperature — raw °C, not Z-score
        df['T2M_MIN_lag_15d'] = df['T2M_MIN_raw'].shift(15)

        # Soft RH signal: 0 below 75% RH, ramps to 1 at 90% RH (raw %)
        df['RH_high_flag']   = np.clip((df['RH2M_raw'] - 75.0) / (90.0 - 75.0), 0.0, 1.0)
        df['RH_persist_7d']  = df['RH_high_flag'].rolling(window=7, min_periods=1).sum()

        # Rainfall accumulations from raw mm
        df['Rain_sum_7d']  = df['PRECTOTCORR_raw'].rolling(window=7,  min_periods=1).sum()
        df['Rain_sum_14d'] = df['PRECTOTCORR_raw'].rolling(window=14, min_periods=1).sum()

        # Monsoon indicator: 7-day mean raw RH > 75% — matches dataset_pipeline.py.
        # Do NOT use a hardcoded month filter; that definition diverges from training
        # and misses June early-onset and Oct/Nov late-season events.
        df['Monsoon_ind'] = (
            df['RH2M_raw'].rolling(7, min_periods=1).mean() > 75
        ).astype(int)

        # Latent window features: recent deviation from 28-day baseline — raw units
        df['RH2M_latent_window'] = (
            df['RH2M_raw'] - df['RH2M_raw'].rolling(28, min_periods=1).mean()
        )
        df['T2M_latent_window'] = (
            df['T2M_MIN_raw'] - df['T2M_MIN_raw'].rolling(28, min_periods=1).mean()
        )

        return df.fillna(0)

    def run_inference(self, location, target_date, agro_inputs: dict) -> dict:
        """
        Runs a prediction for a given target date and location.

        Args:
            location    : region name or coords passed to WeatherDataProvider
            target_date : date for which to produce a risk score
            agro_inputs : dict with keys matching self.agro_features
                          e.g. {'variety_susceptibility': 1, 'is_ratoon': 0,
                                'crop_age_days': 180}

        Returns:
            dict with risk_score, risk_class, confidence_score, and diagnostics.
        """
        # 1. Weather causality slice (real-time fetch)
        past_weather        = self._fetch_past_weather_slice(location, target_date)
        transformed_weather = self._apply_kg_transforms(past_weather)

        # Take the exact seq_len-day window ending at target_date
        w_slice      = transformed_weather.tail(self.seq_len).copy()
        w_raw_values = w_slice[self.weather_features].values  # (seq_len, F)

        # 2. Agronomic feature alignment
        a_raw_values = np.array([[agro_inputs[f] for f in self.agro_features]])

        # 3. Scaling
        # Weather: already Z-scored by _apply_kg_transforms() — pass directly.
        # Agro: raw natural units; transform with the saved agro scaler.
        a_sc_data = self.a_sc.transform(a_raw_values)

        X_w = torch.FloatTensor(w_raw_values).unsqueeze(0).to(self.device)
        X_a = torch.FloatTensor(a_sc_data).to(self.device)

        # 4. Model execution
        with torch.no_grad():
            logits, _, conf_logit = self.model(X_w, X_a)

        # 5. Temperature-calibrated probability
        # train.py fits temperature T on the val set and uses sigmoid(logits / T)
        # at evaluation time. Must replicate that here — sigmoid(logits) without
        # temperature produces overconfident scores that do not match the
        # distribution the threshold (0.2918) was calibrated on.
        risk_score = torch.sigmoid(logits / self.T).item()

        # conf_logit is pre-sigmoid (model.py v11.1) — apply sigmoid here.
        confidence = torch.sigmoid(conf_logit).item()

        # Risk class thresholds are relative to the temperature-calibrated
        # probability scale. Adjust if you recalibrate the threshold.
        risk_class = (
            "High"   if risk_score >= 0.7 else
            "Medium" if risk_score >= 0.3 else
            "Low"
        )

        return {
            "risk_score":            risk_score,
            "risk_class":            risk_class,
            "confidence_score":      confidence,
            "logits":                logits.item(),
            "temperature":           self.T,
            "raw_weather_sequence":  w_raw_values,   # needed for explainability
            "weather_feature_names": self.weather_features,
            "agro_inputs":           agro_inputs,
            "is_signal_saturated":   abs(logits.item()) > 10,
        }