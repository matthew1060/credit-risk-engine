import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# LOAD MODEL DIRECTLY (no API needed)
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load('models/credit_risk_lgbm.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return model, feature_cols, explainer

model, feature_cols, explainer = load_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(revolving_util, age, monthly_income, debt_ratio, 
            open_credit_lines, real_estate_loans, dependents,
            late_30_59, late_60_89, late_90):
    
    total_late = late_30_59 + late_60_89 + late_90
    debt_to_income = debt_ratio / (monthly_income + 1)

    input_data = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': late_90,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfDependents': dependents,
        'total_late_payments': total_late,
        'debt_to_income': debt_to_income
    }])

    prob = float(model.predict(input_data)[0])

    if prob < 0.3:
        risk_category = "LOW"
        recommendation = "APPROVE"
    elif prob < 0.6:
        risk_category = "MEDIUM"
        recommendation = "REVIEW"
    else:
        risk_category = "HIGH"
        recommendation = "DECLINE"

    shap_vals = explainer.shap_values(input_data)[0]
    feature_importance = list(zip(feature_cols, shap_vals))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = [
        {"feature": f, "impact": round(float(v), 4)}
        for f, v in feature_importance[:5]
    ]

    return prob, risk_category, recommendation, top_factors

# ============================================================
# HEADER
# ============================================================

st.title("🏦 Credit Risk Scoring Dashboard")
st.markdown("**Powered by LightGBM + SHAP Explainability | AUC-ROC: 0.8641**")
st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📋 Borrower Profile")
st.sidebar.markdown("Enter the borrower's financial details below:")

revolving_util = st.sidebar.slider("Revolving Credit Utilization", 0.0, 1.0, 0.3, 0.01)
age = st.sidebar.number_input("Age", 18, 100, 45)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 0, 100000, 5000, 100)
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 1.0, 0.3, 0.01)
open_credit_lines = st.sidebar.number_input("Open Credit Lines", 0, 50, 5)
real_estate_loans = st.sidebar.number_input("Real Estate Loans", 0, 20, 1)
dependents = st.sidebar.number_input("Number of Dependents", 0, 20, 0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Late Payment History:**")
late_30_59 = st.sidebar.number_input("30-59 Days Past Due", 0, 10, 0)
late_60_89 = st.sidebar.number_input("60-89 Days Past Due", 0, 10, 0)
late_90 = st.sidebar.number_input("90+ Days Late", 0, 10, 0)

predict_button = st.sidebar.button("🔍 Assess Credit Risk", type="primary", use_container_width=True)

# ============================================================
# MAIN PANEL
# ============================================================

if predict_button:
    prob, risk, rec, factors = predict(
        revolving_util, age, monthly_income, debt_ratio,
        open_credit_lines, real_estate_loans, dependents,
        late_30_59, late_60_89, late_90
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Default Probability", f"{prob:.1%}")
    with col2:
        color = "🟢" if risk == "LOW" else "🟡" if risk == "MEDIUM" else "🔴"
        st.metric("Risk Category", f"{color} {risk}")
    with col3:
        rec_color = "✅" if rec == "APPROVE" else "⚠️" if rec == "REVIEW" else "❌"
        st.metric("Recommendation", f"{rec_color} {rec}")
    with col4:
        st.metric("Model AUC-ROC", "0.8641")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Default Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prob > 0.6 else "orange" if prob > 0.3 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 60], 'color': "#fff3cd"},
                    {'range': [60, 100], 'color': "#f8d7da"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top Risk Factors (SHAP)")
        features = [f['feature'] for f in factors]
        impacts = [f['impact'] for f in factors]
        colors = ['red' if i > 0 else 'green' for i in impacts]
        fig2 = px.bar(x=impacts, y=features, orientation='h',
                      color=colors,
                      color_discrete_map={'red': '#dc3545', 'green': '#28a745'},
                      labels={'x': 'SHAP Impact', 'y': 'Feature'})
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("📊 Decision Explanation")

    top_factor = factors[0]['feature']
    second_factor = factors[1]['feature']

    if rec == "DECLINE":
        st.error(f"**Application Declined** — Default probability of {prob:.1%} exceeds acceptable risk threshold. "
                f"Primary driver: **{top_factor}** (impact: {factors[0]['impact']:.2f}), "
                f"followed by **{second_factor}** (impact: {factors[1]['impact']:.2f}).")
    elif rec == "REVIEW":
        st.warning(f"**Manual Review Required** — Default probability of {prob:.1%} falls in the medium risk band. "
                  f"Primary concern: **{top_factor}** (impact: {factors[0]['impact']:.2f}).")
    else:
        st.success(f"**Application Approved** — Default probability of {prob:.1%} is within acceptable limits. "
                  f"Strongest positive signal: **{top_factor}** (impact: {factors[0]['impact']:.2f}).")

else:
    st.info("👈 Enter borrower details in the sidebar and click **Assess Credit Risk** to get a prediction.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", "104,999")
    with col2:
        st.metric("Model AUC-ROC", "0.8641")
    with col3:
        st.metric("Default Rate", "6.68%")