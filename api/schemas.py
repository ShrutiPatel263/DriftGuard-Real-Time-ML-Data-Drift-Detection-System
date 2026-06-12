from pydantic import BaseModel
from typing import List

# Request: what data the user sends TO the API 
class TransactionInput(BaseModel):
    Time: float
    V1: float;  V2: float;  V3: float;  V4: float;  V5: float
    V6: float;  V7: float;  V8: float;  V9: float;  V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


# API response for a fraud prediction.
# fraud_probability: 0.0 to 1.0 (higher = more likely fraud)
# is_fraud: True/False based on 0.5 threshold
class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str       # "LOW", "MEDIUM", "HIGH" — more human-readable


# Drift result for one individual feature.
class FeatureDriftResult(BaseModel):
    feature: str
    ks_stat: float        # how big the drift is
    p_value: float        # statistical significance
    drifted: bool 


#  Full drift check response returned by the /check-drift endpoint.
class DriftResponse(BaseModel):
    drift_detected: bool               # overall verdict
    drifted_features: List[str]        # which features drifted
    feature_results: List[FeatureDriftResult]
    recommendation: str                 # what to do — retrain or not