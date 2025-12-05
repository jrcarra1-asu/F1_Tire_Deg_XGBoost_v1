# Databricks notebook source
# MAGIC %pip install xgboost mlflow joblib

# COMMAND ----------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_models():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, encoder, le

numerical_features = ['Throttle', 'Brake', 'Speed', 'Surface_Roughness',
                      'Ambient_Temperature', 'Lateral_G_Force', 'Longitudinal_G_Force',
                      'Tire_Friction_Coefficient', 'Tire_Tread_Depth',
                      'force_on_tire', 'front_surface_temp', 'rear_surface_temp',
                      'front_inner_temp', 'rear_inner_temp']  # Add for validity

categorical_features = ['Tire_Compound', 'Driving_Style', 'Track']

tire_compounds = ['C1', 'C2', 'C3', 'C4', 'C5']  # Match training
driving_styles = ['Aggressive', 'Normal']
tracks = ['Monza', 'Monaco', 'Red Bull Ring']

st.title("F1 Tire Degradation Alert App")

st.header("Input Telemetry Data")
col1, col2 = st.columns(2)

with col1:
    throttle = st.slider("Throttle (0-1)", 0.0, 1.0, 0.5)
    brake = st.slider("Brake (0-1)", 0.0, 1.0, 0.5)
    speed = st.number_input("Speed (km/h)", min_value=0.0, value=200.0)
    surface_roughness = st.number_input("Surface Roughness", min_value=0.0, value=1.0)
    ambient_temp = st.number_input("Ambient Temp (°C)", min_value=-10.0, max_value=50.0, value=25.0)
    lateral_g = st.number_input("Lateral G-Force", min_value=0.0, value=1.5)
    longitudinal_g = st.number_input("Longitudinal G-Force", min_value=0.0, value=1.5)

with col2:
    friction_coeff = st.number_input("Friction Coeff", min_value=0.0, value=0.8)
    tread_depth = st.number_input("Tread Depth (mm)", min_value=0.0, value=5.0)
    force_on_tire = st.number_input("Force on Tire (N)", min_value=0.0, value=1000.0)
    front_surface_temp = st.number_input("Front Surface Temp (°C)", min_value=0.0, value=80.0)
    rear_surface_temp = st.number_input("Rear Surface Temp (°C)", min_value=0.0, value=80.0)
    front_inner_temp = st.number_input("Front Inner Temp (°C)", min_value=0.0, value=90.0)
    rear_inner_temp = st.number_input("Rear Inner Temp (°C)", min_value=0.0, value=90.0)

lap_count = st.number_input("Lap Count", min_value=1, max_value=70, value=10)  # Fix validity
tire_compound = st.selectbox("Tire Compound", tire_compounds)
driving_style = st.selectbox("Driving Style", driving_styles)
track = st.selectbox("Track", tracks)

if tread_depth < 1.6:
    st.error("Illegal tread per FIA—min 1.6mm.")

if st.button("Predict Risk"):
    try:
        model, scaler, encoder, le = load_models()

        num_vals = [throttle, brake, speed, surface_roughness, ambient_temp, lateral_g, longitudinal_g, friction_coeff, tread_depth, force_on_tire, front_surface_temp, rear_surface_temp, front_inner_temp, rear_inner_temp, lap_count]
        input_data = dict(zip(numerical_features, num_vals))
        input_data.update({'Tire_Compound': tire_compound, 'Driving_Style': driving_style, 'Track': track})
        input_df = pd.DataFrame([input_data])

        X_num = input_df[numerical_features]
        X_cat = input_df[categorical_features]
        try:
            X_cat_encoded = encoder.transform(X_cat)
        except ValueError as e:
            st.error(f"Category error: {e}. Trained categories: {encoder.categories_}")
            st.stop()
        X = np.hstack((X_num.values, X_cat_encoded))
        X_scaled = scaler.transform(X)

        pred_encoded = model.predict(X_scaled)[0]
        probs = model.predict_proba(X_scaled)[0]
        risk = le.inverse_transform([pred_encoded])[0]

        st.success(f"Predicted Degradation Risk: {risk}")
        class_map = {label: idx for idx, label in enumerate(le.classes_)}
        st.write(f"Probabilities - Safe: {probs[class_map['safe']]:.2f}, Medium: {probs[class_map['medium']]:.2f}, Critical: {probs[class_map['critical']]:.2f}")

        if risk == 'critical':
            st.warning("Pit Alert: Degradation critical—initiate undercut to gain positions on fresh tires.")
        elif risk == 'medium':
            st.info("Monitor: Consider overcut if rivals pit first; tires holding but watch inner temps.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. Check PKLs are in repo and compatible.")

st.markdown("---")
st.caption("Model trained on simulated F1 data. For demo only.")
