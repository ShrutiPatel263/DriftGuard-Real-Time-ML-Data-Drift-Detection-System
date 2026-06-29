
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import io
import sys
sys.path.append('..')

from api.schemas import TransactionInput, PredictionResponse, DriftResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_detection import detect_drift

app = FastAPI(
    title="Fraud Detection + Drift Monitor API",
    description="Detects fraud in transactions and monitors for data drift",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"]
)

# with open('models/xgb_model.pkl', 'rb') as f:
#     model = pickle.load(f)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


PREDICTORS = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
              'V11','V12','V13','V14','V15','V16','V17','V18','V19',
              'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# Health check
@app.get("/")
def root():
    """Simple check to confirm API is running."""
    return {"status": "running", "message": "Fraud Drift Detection API is live!"}

# Predict fraud for a single transaction 
@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionInput):

    # Convert incoming request to a DataFrame — XGBoost needs DataFrame input
    data = pd.DataFrame([transaction.dict()])

    # DMatrix = XGBoost's special data format (required for xgb.train models)
    dmatrix = xgb.DMatrix(data[PREDICTORS])

    # Get probability (0.0 to 1.0) — not just 0 or 1
    probability = float(model.predict(dmatrix)[0])

    # Convert probability to risk level — more useful than just True/False
    if probability < 0.3:
        risk = "LOW"
    elif probability < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return PredictionResponse(
        fraud_probability=round(probability, 4),
        is_fraud=probability >= 0.5,
        risk_level=risk
    )

#Check drift on a batch of uploaded transactions 
@app.post("/check-drift", response_model=DriftResponse)
async def check_drift(file: UploadFile = File(...)):
    """
    User uploads a CSV file of new transactions.
    API compares it against reference data and returns drift report.

    UploadFile = FastAPI's way of accepting file uploads
    File(...)  = means the file field is required (... = required in FastAPI)
    """
    # Read uploaded CSV into bytes, then convert to DataFrame
    contents = await file.read()                      # read file bytes
    df = pd.read_csv(io.StringIO(contents.decode())) # decode bytes → string → DataFrame

    # Basic validation — make sure required columns exist
    missing = [col for col in PREDICTORS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {missing}"
        )

    # Run drift detection (from our drift_detection.py)
    result = detect_drift(df)

    # Add a recommendation message based on result
    if result['drift_detected']:
        recommendation = (
            f"Drift detected in {len(result['drifted_features'])} features: "
            f"{result['drifted_features']}. Immediate retraining recommended."
        )
    else:
        recommendation = "No significant drift detected. Model is stable."

    return DriftResponse(
        drift_detected=result['drift_detected'],
        drifted_features=result['drifted_features'],
        feature_results=result['feature_results'],
        recommendation=recommendation
    )