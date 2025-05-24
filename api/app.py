import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from api.ML_utils import MultimodalModel

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

class MultimodalRequest(BaseModel):
    lifelog: List[float]    # 1차원 리스트 (3일 x 28피처 = 84개 값)
    lifestyle: List[float]  # 1차원 리스트 (lifestyle 피처들)

class MultimodalResponse(BaseModel):
    statusCode: int
    message: str
    risk_score: float
    is_dementia: bool      

@app.post("/prediction/multimodal", response_model=MultimodalResponse)
def predict_multimodal(req: MultimodalRequest):
    try:
        if len(req.lifelog) != 84:
            raise HTTPException(status_code=400, detail="The lifelog data must contain exactly 84 values (3 days x 28 features).")
        
        risk_score = multimodal_model.predict(req.lifelog, req.lifestyle)
        is_dementia = risk_score > 0.5
        
        return MultimodalResponse(
            statusCode=200,
            message="success",
            risk_score=risk_score,
            is_dementia=is_dementia
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failure: {str(e)}")
