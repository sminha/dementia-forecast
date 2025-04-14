import os
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from multimodal import MultimodalModel
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

class LifelogData(BaseModel):
    day1: List[float]
    day2: Optional[List[float]] = None
    day3: Optional[List[float]] = None

class LifestyleData(BaseModel):
    data: Dict[str, Any]

class MultimodalPredictionRequest(BaseModel):
    lifelog: LifelogData
    lifestyle: LifestyleData

class MultimodalPredictionResponse(BaseModel):
    statusCode: int
    message: str
    risk_score: float


@app.post("/prediction/multimodal", response_model=MultimodalPredictionResponse)
def multimodal_prediction(request: MultimodalPredictionRequest):
    try:
        lifelog_list = [request.lifelog.day1]
        days_used = 1
        
        if request.lifelog.day2 is not None:
            lifelog_list.append(request.lifelog.day2)
            days_used = 2
            
        if request.lifelog.day3 is not None:
            lifelog_list.append(request.lifelog.day3)
            days_used = 3
        
        lifelog_data = np.array(lifelog_list).flatten().reshape(1, -1)
        
        lifestyle_data = pd.DataFrame([request.lifestyle.data])
        
        prediction_result = multimodal_model.predict(lifelog_data, lifestyle_data)
        
        if isinstance(prediction_result, np.ndarray) and len(prediction_result) > 0:
            risk_score = float(prediction_result[0])
        else:
            risk_score = float(prediction_result)
            
        return {
            "statusCode": 200,
            "message": f"Multimodal prediction completed successfully using {days_used} days of lifelog data",
            "risk_score": risk_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process multimodal prediction: {str(e)}")