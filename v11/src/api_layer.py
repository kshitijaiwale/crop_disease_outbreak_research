"""
V11 KG-CTCN Production API (FastAPI)
--------------------------------------------------
Serves the Deployment Layer logic via HTTP endpoints:
/predict and /feedback
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deployment_layer import DeploymentAPI
from feedback_db import submit_feedback

app = FastAPI(
    title="V11 KG-CTCN Agricultural Early Warning System",
    description="Causal, explainable, event-driven Red Rot forecasting API.",
    version="11.0"
)

# Initialize the deployment API (loads frozen model and scalers)
deployment_api = DeploymentAPI()

class PredictRequest(BaseModel):
    location: str
    date: Optional[str] = None
    crop_stage: Optional[str] = None
    is_ratoon: Optional[bool] = None

class FeedbackRequest(BaseModel):
    prediction_id: str
    outbreak_observed: str # "Yes", "No", "Unknown"
    expert_validated: Optional[bool] = False

@app.post("/predict")
def predict_risk(req: PredictRequest):
    """
    Generate a 3-7 day early warning risk prediction.
    """
    farmer_inputs = {}
    if req.is_ratoon is not None:
        farmer_inputs["is_ratoon"] = int(req.is_ratoon)
        
    try:
        result = deployment_api.predict(req.location, req.date, farmer_inputs)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def log_feedback(req: FeedbackRequest):
    """
    Submit delayed ground-truth feedback for a previous prediction.
    """
    try:
        submit_feedback(
            req.prediction_id, 
            req.outbreak_observed, 
            1 if req.expert_validated else 0
        )
        return {"status": "success", "message": "Feedback successfully recorded for offline retraining."}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # This acts as the local development server launcher
    print("=" * 60)
    print(" Starting V11 API Layer ")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
