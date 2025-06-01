import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and feature columns
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# ---------- CSS STYLE ----------
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 1.5em;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
        margin-top: 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.03);
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- APP HEADER ----------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📱 Expresso Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Predict whether a customer will churn based on their usage patterns</p>", unsafe_allow_html=True)
st.markdown("---")
st.subheader("📋 Enter Customer Information:")

# ---------- INPUT FORM ----------
col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("🌍 Region", ['Dakar', 'Thies', 'Saint-Louis'])
    tenure = st.selectbox("📆 Tenure", ['<1 month', '1–3 months', '3–6 months', '>6 months'])
    montant = st.number_input("💰 Montant")
    frequence_rech = st.number_input("🔁 Frequence Rech")
    revenue = st.number_input("📈 Revenue")
    arpu_segment = st.number_input("📊 ARPU Segment")
    frequence = st.number_input("📞 Frequence")
    data_volume = st.number_input("📶 Data Volume")

with col2:
    on_net = st.number_input("📱 On Net")
    orange = st.number_input("📲 Orange")
    tigo = st.number_input("📡 Tigo")
    zone1 = st.number_input("🗺️ Zone1")
    zone2 = st.number_input("🗺️ Zone2")
    regularity = st.number_input("📌 Regularity")
    top_pack = st.selectbox("🎁 Top Pack", ['No Pack', 'Social Pack', 'Internet Pack'])
    freq_top_pack = st.number_input("🔂 Freq Top Pack")
    mrg = st.selectbox("🧾 MRG", ['Prepaid', 'Postpaid'])

# ---------- PREDICTION BUTTON ----------
if st.button("🚀 Predict"):
    # Convert categorical inputs to match training preprocessing
    region_map = {'Dakar': 'REGION_Dakar', 'Thies': 'REGION_Thies', 'Saint-Louis': 'REGION_Saint-Louis'}
    top_pack_map = {'No Pack': 'TOP_PACK_No Pack', 'Social Pack': 'TOP_PACK_Social Pack', 'Internet Pack': 'TOP_PACK_Internet Pack'}
    tenure_map = {'<1 month': 0, '1–3 months': 1, '3–6 months': 2, '>6 months': 3}
    mrg_map = {'Prepaid': 0, 'Postpaid': 1}

    # Base features
    data = {
        'TENURE': tenure_map[tenure],
        'MONTANT': montant,
        'FREQUENCE_RECH': frequence_rech,
        'REVENUE': revenue,
        'ARPU_SEGMENT': arpu_segment,
        'FREQUENCE': frequence,
        'DATA_VOLUME': data_volume,
        'ON_NET': on_net,
        'ORANGE': orange,
        'TIGO': tigo,
        'ZONE1': zone1,
        'ZONE2': zone2,
        'MRG': mrg_map[mrg],
        'REGULARITY': regularity,
        'FREQ_TOP_PACK': freq_top_pack
    }

    # One-hot encoding for REGION and TOP_PACK
    for col in feature_columns:
        if col.startswith('REGION_'):
            data[col] = 1 if col == region_map[region] else 0
        elif col.startswith('TOP_PACK_'):
            data[col] = 1 if col == top_pack_map[top_pack] else 0

    # Ensure all features exist and are in the right order
    input_df = pd.DataFrame([data])[feature_columns]

    # Prediction
    prediction = model.predict(input_df)[0]
    result = "✅ The customer is likely to stay." if prediction == 0 else "⚠️ The customer is likely to churn."

    # Display
    if prediction == 1:
        st.error(result)
    else:
        st.success(result)
