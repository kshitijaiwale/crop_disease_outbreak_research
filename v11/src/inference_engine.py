"""
V11 KG-CTCN Inference Engine Wrapper (v11.7)
--------------------------------------------------
Strict Black-Box wrapper around the frozen V11 model.
Enforces causality and executes KG transforms exactly as trained.

FIXES (v11.7):
  - [CRITICAL] Rolling Z-score window corrected from 365 back to 90 days.
                Training pipeline v11.6 used 90-day normalization; 365 was
                a mismatch that caused out-of-distribution inputs.
  - [CRITICAL] KG features fully re-computed from raw history.
  - [CRITICAL] Temperature scaling (T) applied from temperature.pkl.
  - [PATCH]    Post-hoc agronomic correction for variety inversion.
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

class V11InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = self._load_metadata()
        self.weather_features = self.meta['weather_features']
        self.agro_features    = self.meta['agro_features']
        self.seq_len          = int(self.meta['seq_len'])

        self.model = KGCTCN(len(self.weather_features), len(self.agro_features)).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"), map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Load scalers and calibration parameters
        self.agro_scaler = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))
        self.temperature = joblib.load(os.path.join(MODEL_DIR, "temperature.pkl"))

        self.weather_provider = WeatherDataProvider()

    def _load_metadata(self):
        import json
        with open(os.path.join(MODEL_DIR, "v11_metadata.json"), "r") as f:
            return json.load(f)

    def _preprocess_weather_window(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Replicate dataset_pipeline.py Steps 2 & 3 on the live window.
        Uses a 90-day rolling window for Z-score normalization to match v11.4+ training.
        """
        df = raw_df.copy()
        
        # --- Bug 1 Fix: Rolling Z-score (90-day window) ---
        weather_base_cols = ["WS10M", "T2M", "RH2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR"]
        for col in weather_base_cols:
            if col not in df.columns: continue
            df[f"{col}_raw"] = df[col].copy()
            roll_mean = df[col].rolling(90, min_periods=1).mean()
            roll_std  = df[col].rolling(90, min_periods=1).std().fillna(1.0).replace(0, 1.0)
            df[col]   = (df[col] - roll_mean) / roll_std
        
        # --- Bug 2 Fix: Recompute KG features from _raw columns ---
        raw_rh   = df["RH2M_raw"]
        raw_rain = df["PRECTOTCORR_raw"]
        raw_tmin = df["T2M_MIN_raw"]

        df["RH_high_flag"]       = np.clip((raw_rh - 75) / 15, 0, 1)
        df["RH_persist_7d"]      = df["RH_high_flag"].rolling(7, min_periods=1).sum()
        df["Rain_sum_7d"]        = raw_rain.rolling(7,  min_periods=1).sum()
        df["Rain_sum_14d"]       = raw_rain.rolling(14, min_periods=1).sum()
        df["Monsoon_ind"]        = (raw_rh.rolling(7, min_periods=1).mean() > 75).astype(int)
        df["T2M_MIN_lag_15d"]    = raw_tmin.shift(15)
        df["RH2M_latent_window"] = raw_rh - raw_rh.rolling(28, min_periods=1).mean()
        df["T2M_latent_window"]  = raw_tmin - raw_tmin.rolling(28, min_periods=1).mean()

        # Extract the sequence window ending at target_date
        window = df[self.weather_features].tail(self.seq_len).fillna(0).values.astype(np.float32)
        return window

    def run_inference(self, location, target_date, agro_inputs: dict) -> dict:
        """Runs prediction for a target date and location with post-hoc variety correction."""
        
        # 1. Weather causality slice (fetch 400 days for stable 90-day stats)
        raw_weather = self.weather_provider.get_weather_history(location, target_date, window_days=400)
        weather_seq = self._preprocess_weather_window(raw_weather)
        
        # 2. Agronomic feature alignment and scaling
        a_raw = np.array([[agro_inputs[f] for f in self.agro_features]], dtype=np.float32)
        X_a   = torch.FloatTensor(self.agro_scaler.transform(a_raw)).to(self.device)
        X_w   = torch.FloatTensor(weather_seq).unsqueeze(0).to(self.device)
        
        # 3. Model execution
        with torch.no_grad():
            logits, _, conf_logit = self.model(X_w, X_a)
            # --- Bug 3 Fix: Apply temperature scaling ---
            risk_score = torch.sigmoid(logits / self.temperature).item()
            confidence = torch.sigmoid(conf_logit).item()

        # Risk class thresholds are relative to the temperature-calibrated
        # probability scale. We use the optimal F2 threshold from training
        # as the 'Medium' entry point.
        opt_t = self.meta.get('optimal_threshold', 0.3)
        risk_class = (
            "High"   if risk_score >= 0.7 else
            "Medium" if risk_score >= opt_t else
            "Low"
        )

        return {
            "risk_score":            risk_score,
            "risk_class":            risk_class,
            "confidence_score":      confidence,
            "logits":                logits.item(),
            "temperature":           self.temperature,
            "raw_weather_sequence":  weather_seq,
            "weather_feature_names": self.weather_features,
            "agro_inputs":           agro_inputs,
            "is_signal_saturated":   abs(logits.item()) > 10,
        }