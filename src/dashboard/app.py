import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Scoring Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("🏦 Credit Risk Scoring Dashboard")
st.markdown("**Powered by LightGBM + SHAP Explainability | AUC-ROC: 0.8641**")
st.divider()

# ============================================================
# SIDEBAR - BORROWER INPUT
# ============================================================

st.sidebar.header("📋 Borrower Profile")
st.sidebar.markdown("Enter the borrower's financial details below:")

revolving_util = st.sidebar.slider(
    "Revolving Credit Utilization", 
    min_value=0.0, max_value=1.0, value=0.3, step=0.01,
    help="Proportion of revolving credit being used (0-1)"
)

age = st.sidebar.number_input(
    "Age", min_value=18, max_value=100, value=45
)

monthly_income = st.sidebar.number_input(
    "Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=100
)

debt_ratio = st.sidebar.slider(
    "Debt Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
    help="Monthly debt payments divided by monthly gross income"
)

open_credit_lines = st.sidebar.number_input(
    "Open Credit Lines", min_value=0, max_value=50, value=5
)

real_estate_loans = st.sidebar.number_input(
    "Real Estate Loans", min_value=0, max_value=20, value=1
)

dependents = st.sidebar.number_input(
    "Number of Dependents", min_value=0, max_value=20, value=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Late Payment History:**")

late_30_59 = st.sidebar.number_input(
    "30-59 Days Past Due", min_value=0, max_value=10, value=0
)

late_60_89 = st.sidebar.number_input(
    "60-89 Days Past Due", min_value=0, max_value=10, value=0
)

late_90 = st.sidebar.number_input(
    "90+ Days Late", min_value=0, max_value=10, value=0
)

# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.sidebar.button("🔍 Assess Credit Risk", type="primary", use_container_width=True)

# ============================================================
# MAIN PANEL
# ============================================================

if predict_button:
    # Call API
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": revolving_util,
        "age": age,
        "NumberOfTime30_59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60_89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        # ---- ROW 1: KEY METRICS ----
        col1, col2, col3, col4 = st.columns(4)

        prob = result['default_probability']
        risk = result['risk_category']
        rec = result['recommendation']

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

        # ---- ROW 2: GAUGE + RISK FACTORS ----
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
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Top Risk Factors (SHAP)")
            factors = result['top_risk_factors']
            features = [f['feature'] for f in factors]
            impacts = [f['impact'] for f in factors]
            colors = ['red' if i > 0 else 'green' for i in impacts]

            fig2 = px.bar(
                x=impacts, y=features, orientation='h',
                color=colors,
                color_discrete_map={'red': '#dc3545', 'green': '#28a745'},
                labels={'x': 'SHAP Impact', 'y': 'Feature'}
            )
            fig2.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ---- ROW 3: EXPLANATION ----
        st.subheader("📊 Decision Explanation")

        top_factor = factors[0]['feature'].replace('_', ' ').replace('NumberOf', '').replace('DaysPastDue', ' days past due')
        second_factor = factors[1]['feature'].replace('_', ' ').replace('NumberOf', '').replace('DaysPastDue', ' days past due')

        if rec == "DECLINE":
            st.error(f"**Application Declined** — Default probability of {prob:.1%} exceeds acceptable risk threshold. "
                    f"Primary driver: **{top_factor}** (impact: {factors[0]['impact']:.2f}), "
                    f"followed by **{second_factor}** (impact: {factors[1]['impact']:.2f}).")
        elif rec == "REVIEW":
            st.warning(f"**Manual Review Required** — Default probability of {prob:.1%} falls in the medium risk band. "
                      f"Primary concern: **{top_factor}** (impact: {factors[0]['impact']:.2f}). "
                      f"Recommend further assessment by a senior credit analyst.")
        else:
            st.success(f"**Application Approved** — Default probability of {prob:.1%} is within acceptable limits. "
                      f"Strongest positive signal: **{top_factor}** (impact: {factors[0]['impact']:.2f}).")

    except Exception as e:
        st.error(f"API Error: {str(e)}. Make sure the FastAPI server is running.")

else:
    # Default state
    st.info("👈 Enter borrower details in the sidebar and click **Assess Credit Risk** to get a prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", "104,999")
    with col2:
        st.metric("Model AUC-ROC", "0.8641")
    with col3:
        st.metric("Default Rate", "6.68%")