

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# FastAPI is running on port 8000, Streamlit on 8501
API_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriftGuard",
    page_icon="🛡️",
    layout="wide"   # use full width of browser
)

st.title("🛡️ DriftGuard — Real-Time ML Data Drift Detection System")
st.markdown("*Real-time fraud detection with automatic drift monitoring*")

# ── Sidebar navigation ─────────────────────────────────────────────────────
# This creates a sidebar menu on the left
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🔎 Predict Fraud", "📊 Check Data Drift"]
)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.header("What does this system do?")

    # st.columns splits the page into side-by-side sections
    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        **🔎 Fraud Detection**
        - Enter transaction details
        - Get real-time fraud probability
        - Risk level: LOW / MEDIUM / HIGH
        """)

    with col2:
        st.warning("""
        **📊 Drift Detection**
        - Upload a CSV of new transactions
        - System compares vs training data
        - Alerts if fraud patterns shifted
        """)

    st.success("**Why this matters:** Even a great fraud model degrades over time \
    as fraudsters adapt. This system automatically detects when retraining is needed.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICT FRAUD (single transaction)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔎 Predict Fraud":
    st.header("🔎 Single Transaction Fraud Check")
    st.markdown("Enter transaction values below:")

    # Two columns for input layout
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        time   = st.number_input("Time (seconds from first transaction)", value=50000.0)
        v1  = st.number_input("V1",  value=-1.36)
        v2  = st.number_input("V2",  value=-0.07)
        v3  = st.number_input("V3",  value=2.54)
        v4  = st.number_input("V4",  value=1.38)
    with col2:
        v14 = st.number_input("V14", value=-0.31)
        v17 = st.number_input("V17", value=-0.99)
        v12 = st.number_input("V12", value=-1.16)
        v10 = st.number_input("V10", value=-0.17)
        v11 = st.number_input("V11", value=-0.17)
        v16 = st.number_input("V16", value=-0.45)

    # When button is clicked, send data to FastAPI /predict endpoint
    if st.button("🔍 Check for Fraud", type="primary"):

        # Build payload — a dictionary matching TransactionInput schema
        payload = {
            "Time": time, "Amount": amount,
            "V1": v1, "V2": v2, "V3": v3, "V4": v4,
            "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0,
            "V10": v10, "V11": v11, "V12": v12, "V13": 0.0, "V14": v14,
            "V15": 0.0, "V16": v16, "V17": v17, "V18": 0.0, "V19": 0.0,
            "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
            "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0
        }

        # Send POST request to FastAPI
        with st.spinner("Checking transaction..."):
            response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            prob   = result['fraud_probability']
            risk   = result['risk_level']

            # Show result with color based on risk level
            if risk == "HIGH":
                st.error(f"🚨 FRAUD DETECTED! Probability: {prob:.1%} | Risk: {risk}")
            elif risk == "MEDIUM":
                st.warning(f"⚠️ Suspicious Transaction. Probability: {prob:.1%} | Risk: {risk}")
            else:
                st.success(f"✅ Transaction looks legitimate. Probability: {prob:.1%} | Risk: {risk}")

            # Gauge chart — visually shows the probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if risk=="HIGH" else "orange" if risk=="MEDIUM" else "green"},
                    'steps': [
                        {'range': [0, 30],  'color': "#d4edda"},
                        {'range': [30, 70], 'color': "#fff3cd"},
                        {'range': [70, 100],'color': "#f8d7da"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("API error. Make sure FastAPI is running.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: DRIFT DETECTION (CSV upload)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Check Data Drift":
    st.header("📊 Data Drift Detection")
    st.markdown("Upload a CSV of new transactions to check if fraud patterns have shifted.")

    # st.file_uploader = drag and drop file upload widget
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Show preview of uploaded data
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"✅ Uploaded {len(df_preview):,} transactions")
        st.dataframe(df_preview.head(3))  # show first 3 rows

        # Reset file pointer — important! After reading, pointer is at end
        uploaded_file.seek(0)

        if st.button("🔍 Run Drift Detection", type="primary"):

            # Send file to FastAPI /check-drift endpoint
            with st.spinner("Analyzing for drift..."):
                response = requests.post(
                    f"{API_URL}/check-drift",
                    files={"file": ("data.csv", uploaded_file, "text/csv")}
                )

            if response.status_code == 200:
                result = response.json()

                # Show overall verdict
                if result['drift_detected']:
                    st.error("🚨 DATA DRIFT DETECTED! Model retraining recommended.")
                else:
                    st.success("✅ No significant drift. Model is stable.")

                st.info(result['recommendation'])

                # Show per-feature drift table
                st.subheader("Feature-level Drift Analysis")
                features_df = pd.DataFrame(result['feature_results'])
                features_df['Status'] = features_df['drifted'].map(
                    {True: '🔴 DRIFTED', False: '🟢 OK'}
                )
                st.dataframe(features_df[['feature','ks_stat','p_value','Status']])

                # Bar chart of KS statistics
                fig = px.bar(
                    features_df,
                    x='feature', y='ks_stat',
                    color='drifted',
                    color_discrete_map={True: 'red', False: 'green'},
                    title='KS Statistic per Feature (higher = more drift)',
                    labels={'ks_stat': 'KS Statistic', 'feature': 'Feature'}
                )
                fig.add_hline(y=0.05, line_dash="dash",
                              annotation_text="Drift threshold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("API error. Make sure FastAPI is running.")
