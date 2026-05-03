"""
V11 KG-CTCN Inference Engine Wrapper
--------------------------------------------------
Strict Black-Box wrapper around the frozen V11 model.
Ensures causality, no data leakage, and executes KG transforms exactly as trained.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KGCTCN

from weather_provider import WeatherDataProvider

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

class V11InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = self._load_metadata()
        self.weather_features = self.meta['weather_features']
        self.agro_features = self.meta['agro_features']
        self.seq_len = int(self.meta['seq_len'])
        
        self.model = KGCTCN(len(self.weather_features), len(self.agro_features)).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "v11_kg_ctcn.pth"), map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Load scalers dynamically based on training split (2005-2018)
        self._init_scalers()
        
        # Real-time Weather Data Provider
        self.weather_provider = WeatherDataProvider()

    def _load_metadata(self):
        with open(os.path.join(MODEL_DIR, "v11_metadata.txt"), "r") as f:
            lines = f.readlines()
        meta = {}
        for l in lines:
            k, v = l.strip().split("=")
            meta[k] = v.split(",") if "," in v else v
        return meta

    def _init_scalers(self):
        """Loads pre-fitted scalers from PKL files (Zero CSV dependency)."""
        self.w_sc = joblib.load(os.path.join(MODEL_DIR, "weather_scaler.pkl"))
        self.a_sc = joblib.load(os.path.join(MODEL_DIR, "agro_scaler.pkl"))

    def _fetch_past_weather_slice(self, location, target_date):
        """Fetches EXACTLY the past data needed up to target_date. Enforces causality."""
        return self.weather_provider.get_weather_history(location, target_date, window_days=120)

    def _apply_kg_transforms(self, df):
        """Applies strict V11 transforms without leakage."""
        # 1. Window-Local Normalization (Geographic Invariance)
        weather_base_cols = ['WS10M', 'T2M', 'RH2M', 'T2M_MIN', 'T2M_MAX', 'PRECTOTCORR']
        for col in weather_base_cols:
            mean_90 = df[col].rolling(90, min_periods=1).mean()
            std_90 = df[col].rolling(90, min_periods=1).std() + 1e-5
            df[f'{col}_raw'] = df[col] # backup raw
            df[col] = (df[col] - mean_90) / std_90
            df[f'{col}_z90'] = df[col]

        # 2. KG-driven causal transforms
        df['T2M_MIN_lag_15d'] = df['T2M_MIN'].shift(15)
        # Soft RH signal (75% to 90%) - calibrated for regional variance
        df['RH_high_flag'] = np.clip((df['RH2M_raw'] - 75.0) / (90.0 - 75.0), 0.0, 1.0)
        df['RH_persist_7d'] = df['RH_high_flag'].rolling(window=7, min_periods=1).sum()
        
        df['Rain_sum_7d'] = df['PRECTOTCORR_raw'].rolling(window=7, min_periods=1).sum()
        df['Rain_sum_14d'] = df['PRECTOTCORR_raw'].rolling(window=14, min_periods=1).sum()
        
        df['Monsoon_ind'] = df['date'].dt.month.isin([7, 8, 9]).astype(int)
        
        df['RH2M_latent_window'] = df['RH2M'].shift(7).rolling(window=21, min_periods=1).mean()
        df['T2M_latent_window'] = df['T2M'].shift(7).rolling(window=21, min_periods=1).mean()
        
        return df.fillna(0)

    def run_inference(self, location, target_date, agro_inputs):
        """
        Runs a prediction for a given target date and location.
        location: region name or coords
        agro_inputs: dict with keys matching self.agro_features
        """
        # 1. Weather causality slice (Real-time Fetch)
        past_weather = self._fetch_past_weather_slice(location, target_date)
        transformed_weather = self._apply_kg_transforms(past_weather)
        
        # Take the exact 28-day window ending at target_date
        w_slice = transformed_weather.tail(self.seq_len).copy()
        w_raw_values = w_slice[self.weather_features].values
        
        # 2. Agronomic feature alignment
        a_raw_values = np.array([[agro_inputs[f] for f in self.agro_features]])
        
        # 3. Scaling
        w_sc_data = self.w_sc.transform(w_raw_values)
        a_sc_data = self.a_sc.transform(a_raw_values)
        
        X_w = torch.FloatTensor(w_sc_data).unsqueeze(0).to(self.device)
        X_a = torch.FloatTensor(a_sc_data).to(self.device)
        
        # 4. Model execution
        with torch.no_grad():
            logits, prob, conf = self.model(X_w, X_a)
            
        risk_score = prob.item()
        confidence = conf.item()
        
        # Risk class logic matching V11
        risk_class = "High" if risk_score >= 0.7 else ("Medium" if risk_score >= 0.3 else "Low")
        
        # Return structured output and raw inputs for explainability/logging
        return {
            "risk_score": risk_score,
            "risk_class": risk_class,
            "confidence_score": confidence,
            "logits": logits.item(),
            "raw_weather_sequence": w_raw_values,  # needed for explainability
            "weather_feature_names": self.weather_features,
            "agro_inputs": agro_inputs,
            "is_signal_saturated": abs(logits.item()) > 10
        }
