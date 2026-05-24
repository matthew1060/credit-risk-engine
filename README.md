---
title: Credit Risk Engine
emoji: 🏦
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# Credit Risk Scoring Engine

A production-grade credit risk scoring system built on 150,000 real borrower records. Predicts the probability of loan default using LightGBM with SHAP explainability, deployed as an interactive Streamlit dashboard.

## Live Demo

🔗 [Try it live on Hugging Face Spaces](https://huggingface.co/spaces/matthew1060/credit-risk-engine)

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
- **Streamlit dashboard** — interactive interface for credit analysts to assess borrower risk
- **Dockerised deployment** — containerised and deployed on Hugging Face Spaces

## Business Case

A commercial bank processing 10,000 loan applications per month spends approximately 15 minutes per application on manual credit assessment. This system automates the initial risk triage, reducing analyst time on clear approve/decline cases by an estimated 60%, while providing SHAP-based explanations that satisfy IFRS 9 regulatory requirements for model transparency.

**Estimated savings:** 1,500 analyst hours per month on a 10,000 application volume.

## Tech Stack

- **Model:** LightGBM with scale_pos_weight for class imbalance
- **Explainability:** SHAP TreeExplainer
- **Dashboard:** Streamlit + Plotly
- **Deployment:** Docker + Hugging Face Spaces

## Project Structure
```
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
```

## Quick Start

```bash
pip install -r requirements_hf.txt
streamlit run src/dashboard/app_standalone.py
```

## Dataset

[Give Me Some Credit — Kaggle](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)

150,000 anonymised borrower records with 10 financial features and a binary default indicator.

## Author

Matthew1060