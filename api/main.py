import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.feature_engineering import add_generic_features

MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/model_metadata.json")
app = FastAPI(title="Universal MLOps Prediction API", version="1.0.0")
model = None
metadata = {}

@app.on_event("startup")
def load_model():
    global model, metadata
    if MODEL_PATH.exists() and META_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(META_PATH.read_text())

class PredictionRequest(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "Universal MLOps API", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    try:
        row = pd.DataFrame([request.features])
        row = add_generic_features(row)
        prediction = model.predict(row)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()
        return {
            "prediction": prediction,
            "task": metadata.get("task"),
            "model": metadata.get("model_name")
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
