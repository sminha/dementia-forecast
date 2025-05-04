import os
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from models.multimodal import MultimodalModel
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

app = FastAPI(title="Dementia Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


@app.get("/")
def read_root():
    return {"message": "Dementia Forecast API is running"}

multimodal_model = MultimodalModel()
multimodal_model.load_models()

class LifelogRecord(BaseModel):
    activity_cal_active: float
    activity_cal_total: float
    activity_daily_movement: float
    activity_day_end: float
    activity_day_start: float
    activity_high: float
    activity_inactive: float
    activity_medium: float
    activity_met_1min: float
    activity_met_min_high: float
    activity_met_min_inactive: float
    activity_met_min_low: float
    activity_met_min_medium: float
    activity_non_wear: float
    activity_steps: float
    activity_total: float
    sleep_awake: float
    sleep_bedtime_end: float
    sleep_bedtime_start: float
    sleep_deep: float
    sleep_duration: float
    sleep_efficiency: float
    sleep_hypnogram_5min: float
    sleep_is_longest: float
    sleep_light: float
    sleep_midpoint_at_delta: float
    sleep_midpoint_time: float
    sleep_period_id: float
    sleep_rem: float
    sleep_rmssd: float
    sleep_rmssd_5min: float
    sleep_total: float

class MultimodalRequest(BaseModel):
    lifelog: List[LifelogRecord]  
    lifestyle: Dict[str, Any]     

class MultimodalResponse(BaseModel):
    statusCode: int
    message: str
    risk_score: float

@app.post("/prediction/multimodal", response_model=MultimodalResponse)
def predict_multimodal(req: MultimodalRequest):
    try:
        df_lifelog = pd.DataFrame([r.dict() for r in req.lifelog])
 
        lifelog_ary = df_lifelog.values.flatten().reshape(1, -1)

        df_lifestyle = pd.DataFrame([req.lifestyle])

        preds = multimodal_model.predict(lifelog_ary, df_lifestyle)
        score = float(preds[0]) if isinstance(preds, np.ndarray) else float(preds)

        return MultimodalResponse(
            statusCode=200,
            message=f"Multimodal prediction completed successfully using {len(req.lifelog)} days of lifelog data",
            risk_score=score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")