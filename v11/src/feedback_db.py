"""
V11 KG-CTCN Feedback Loop System
--------------------------------------------------
Records predictions and collects delayed ground-truth feedback
for periodic offline retraining.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "feedback_loop.sqlite")

def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    """Initializes the feedback database schema."""
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            timestamp TEXT,
            target_date TEXT,
            location TEXT,
            weather_snapshot_json TEXT,
            agro_inputs_json TEXT,
            risk_score REAL,
            risk_class TEXT,
            confidence_score REAL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            prediction_id TEXT PRIMARY KEY,
            feedback_timestamp TEXT,
            outbreak_observed TEXT,  -- "Yes", "No", "Unknown"
            expert_validated INTEGER -- 1 or 0
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(pred_id, target_date, location, weather_sequence, agro_inputs, risk_score, risk_class, confidence):
    """Stores inference state for future feedback mapping."""
    init_db()
    conn = _get_connection()
    c = conn.cursor()
    
    weather_json = json.dumps(weather_sequence.tolist())
    agro_json = json.dumps(agro_inputs)
    
    c.execute('''
        INSERT INTO predictions 
        (prediction_id, timestamp, target_date, location, weather_snapshot_json, agro_inputs_json, risk_score, risk_class, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (pred_id, datetime.now().isoformat(), str(target_date.date()), location, weather_json, agro_json, risk_score, risk_class, confidence))
    
    conn.commit()
    conn.close()

def submit_feedback(pred_id, outbreak_observed, expert_validated=0):
    """Logs farmer feedback against a previous prediction."""
    init_db()
    conn = _get_connection()
    c = conn.cursor()
    
    # Check if prediction exists
    c.execute('SELECT 1 FROM predictions WHERE prediction_id = ?', (pred_id,))
    if not c.fetchone():
        conn.close()
        raise ValueError(f"Prediction ID {pred_id} not found in database.")
        
    c.execute('''
        INSERT OR REPLACE INTO feedback 
        (prediction_id, feedback_timestamp, outbreak_observed, expert_validated)
        VALUES (?, ?, ?, ?)
    ''', (pred_id, datetime.now().isoformat(), outbreak_observed, expert_validated))
    
    conn.commit()
    conn.close()
    return True

def get_retraining_dataset():
    """Extracts completed feedback loops for retraining pipeline."""
    init_db()
    conn = _get_connection()
    df = pd.read_sql_query('''
        SELECT p.*, f.outbreak_observed, f.expert_validated 
        FROM predictions p
        JOIN feedback f ON p.prediction_id = f.prediction_id
        WHERE f.outbreak_observed IN ('Yes', 'No')
    ''', conn)
    conn.close()
    return df
