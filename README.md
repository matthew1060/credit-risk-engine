# Credit Risk Scoring Engine

A production-grade credit risk scoring system built on 150,000 real borrower records. Predicts the probability of loan default using LightGBM with SHAP explainability, deployed as a REST API with an interactive Streamlit dashboard.

## Live Demo

![Dashboard](reports/dashboard_screenshot.png)

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8641 |
| Training samples | 104,999 |
| Default rate | 6.68% |
| Algorithm | LightGBM |

## Features

- **LightGBM model** trained on Kaggle's Give Me Some Credit dataset (150,000 borrower records)
- **SHAP explainability** — every prediction comes with a feature-level explanation showing why the model flagged a borrower
- **FastAPI backend** serving predictions via REST API with full OpenAPI documentation
- **Streamlit dashboard** — interactive interface for credit analysts to assess borrower risk
- **Evidently AI drift monitoring** — detects when incoming borrower profiles deviate from training distribution
- **Azure ML integration** — training pipeline, model registry, custom environment
- **GitHub Actions CI/CD** — automated retraining when training code changes

## Business Case

A commercial bank processing 10,000 loan applications per month spends approximately 15 minutes per application on manual credit assessment. This system automates the initial risk triage, reducing analyst time on clear approve/decline cases by an estimated 60%, while providing SHAP-based explanations that satisfy IFRS 9 regulatory requirements for model transparency.

**Estimated savings:** 1,500 analyst hours per month on a 10,000 application volume.

## Tech Stack

- **Model:** LightGBM with scale_pos_weight for class imbalance
- **Explainability:** SHAP TreeExplainer
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit + Plotly
- **Monitoring:** Evidently AI
- **Cloud:** Azure ML (training pipeline, model registry)
- **CI/CD:** GitHub Actions

## Project Structure

credit-risk-engine/
├── data/                    # Raw and processed data (gitignored)
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── api/
│   │   └── main.py          # FastAPI application
│   └── dashboard/
│       └── app.py           # Streamlit dashboard
├── models/                  # Saved model artifacts (gitignored)
├── reports/                 # Plots and visualisations
├── requirements.txt
└── README.md

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn src.api.main:app --reload

# In a new terminal, start the dashboard
streamlit run src/dashboard/app.py
```

## API Usage

```python
import requests

payload = {
    "RevolvingUtilizationOfUnsecuredLines": 0.9,
    "age": 35,
    "NumberOfTime30_59DaysPastDueNotWorse": 3,
    "DebtRatio": 0.8,
    "MonthlyIncome": 2500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 2,
    "NumberRealEstateLoansOrLines": 0,
    "NumberOfTime60_89DaysPastDueNotWorse": 1,
    "NumberOfDependents": 2
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
```

## Dataset

[Give Me Some Credit — Kaggle](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)

150,000 anonymised borrower records with 10 financial features and a binary default indicator.

## Author

Matthew1060