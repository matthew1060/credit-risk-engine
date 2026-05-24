from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import json
import os

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'credit_risk_lgbm.pkl'))
feature_cols = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_cols.pkl'))
explainer = joblib.load(os.path.join(BASE_DIR, 'models', 'shap_explainer.pkl'))

with open(os.path.join(BASE_DIR, 'models', 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts probability of loan default using LightGBM with SHAP explainability",
    version="1.0.0"
)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class BorrowerProfile(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str
    recommendation: str
    top_risk_factors: list
    model_version: str

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Credit Risk Scoring API",
        "version": "1.0.0",
        "model_auc": metadata['auc_roc_test']
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(borrower: BorrowerProfile):
    try:
        # Prepare input
        total_late = (borrower.NumberOfTime30_59DaysPastDueNotWorse + 
                     borrower.NumberOfTimes90DaysLate + 
                     borrower.NumberOfTime60_89DaysPastDueNotWorse)
        
        debt_to_income = borrower.DebtRatio / (borrower.MonthlyIncome + 1)

        input_data = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': borrower.RevolvingUtilizationOfUnsecuredLines,
            'age': borrower.age,
            'NumberOfTime30-59DaysPastDueNotWorse': borrower.NumberOfTime30_59DaysPastDueNotWorse,
            'DebtRatio': borrower.DebtRatio,
            'MonthlyIncome': borrower.MonthlyIncome,
            'NumberOfOpenCreditLinesAndLoans': borrower.NumberOfOpenCreditLinesAndLoans,
            'NumberOfTimes90DaysLate': borrower.NumberOfTimes90DaysLate,
            'NumberRealEstateLoansOrLines': borrower.NumberRealEstateLoansOrLines,
            'NumberOfTime60-89DaysPastDueNotWorse': borrower.NumberOfTime60_89DaysPastDueNotWorse,
            'NumberOfDependents': borrower.NumberOfDependents,
            'total_late_payments': total_late,
            'debt_to_income': debt_to_income
        }])

        # Predict
        prob = float(model.predict(input_data)[0])

        # Risk category
        if prob < 0.3:
            risk_category = "LOW"
            recommendation = "APPROVE"
        elif prob < 0.6:
            risk_category = "MEDIUM"
            recommendation = "REVIEW"
        else:
            risk_category = "HIGH"
            recommendation = "DECLINE"

        # SHAP explanation
        shap_vals = explainer.shap_values(input_data)[0]
        feature_importance = list(zip(feature_cols, shap_vals))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_factors = [
            {"feature": f, "impact": round(float(v), 4)} 
            for f, v in feature_importance[:5]
        ]

        return PredictionResponse(
            default_probability=round(prob, 4),
            risk_category=risk_category,
            recommendation=recommendation,
            top_risk_factors=top_factors,
            model_version=f"LightGBM v1.0 (AUC={metadata['auc_roc_test']})"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return metadata