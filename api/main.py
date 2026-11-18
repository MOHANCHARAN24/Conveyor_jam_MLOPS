import json
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.feature_engineering import add_conveyor_features
from src.preprocessing import FEATURE_COLUMNS


MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/model_metadata.json")
model = None
metadata: dict = {}


class SensorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    motor_current_a: float = Field(ge=0, le=100)
    rated_motor_current_a: float = Field(gt=0, le=100)
    vibration_mm_s: float = Field(ge=0, le=50)
    belt_age_years: float = Field(ge=0, le=50)
    parcels_per_minute: float = Field(ge=0, le=1000)
    dust_accumulation_index: float = Field(ge=0, le=100)
    days_since_maintenance: float = Field(ge=0, le=3650)
    humidity_pct: float = Field(ge=0, le=100)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: SensorInput


def load_model() -> None:
    global model, metadata
    if MODEL_PATH.exists() and META_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="Conveyor Jam Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def home() -> dict:
    return {"message": "Conveyor Jam Risk API", "model_loaded": model is not None}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    row = pd.DataFrame([request.features.model_dump()])
    row = add_conveyor_features(row)[FEATURE_COLUMNS]
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(row)[0][1])
    return {
        "prediction": "Jam Risk" if prediction == 1 else "Normal Operation",
        "jam_probability": probability,
        "model": metadata.get("model_name"),
    }
