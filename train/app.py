import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved artifacts (assume .pkl files are in the same folder as this app.py)
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
le = joblib.load('label_encoder.pkl')

# Define your features (from the training code)
numerical_features = ['Throttle', 'Brake', 'Speed', 'Surface_Roughness',
                      'Ambient_Temperature', 'Lateral_G_Force', 'Longitudinal_G_Force',
                      'Tire_Friction_Coefficient', 'Tire_Tread_Depth',
                      'force_on_tire', 'front_surface_temp', 'rear_surface_temp',
                      'front_inner_temp', 'rear_inner_temp', 'lap_count']  # New
categorical_features = ['Tire_Compound', 'Driving_Style', 'Track']

# Streamlit app title and description
st.title("F1 Tire Degradation Predictor")
st.write("Enter tire and track conditions to predict degradation risk.")

# Create input form for features
with st.form(key='prediction_form'):
    # Numerical inputs (adjust ranges based on your data)
    throttle = st.slider("Throttle (0-1)", 0.0, 1.0, 0.5)
    brake = st.slider("Brake (0-1)", 0.0, 1.0, 0.0)
    speed = st.number_input("Speed (km/h)", 0.0, 350.0, 200.0)
    surface_roughness = st.number_input("Surface Roughness", 0.0, 1.0, 0.5)
    ambient_temperature = st.number_input("Ambient Temperature (°C)", -10.0, 50.0, 25.0)
    lateral_g_force = st.number_input("Lateral G-Force", -5.0, 5.0, 0.0)
    longitudinal_g_force = st.number_input("Longitudinal G-Force", -5.0, 5.0, 0.0)
    tire_friction_coefficient = st.number_input("Tire Friction Coefficient", 0.0, 2.0, 1.0)
    tire_tread_depth = st.number_input("Tire Tread Depth (mm)", 0.0, 10.0, 5.0)
    force_on_tire = st.number_input("Force on Tire (N)", 0.0, 10000.0, 5000.0)
    front_surface_temp = st.number_input("Front Surface Temp (°C)", 0.0, 200.0, 80.0)
    rear_surface_temp = st.number_input("Rear Surface Temp (°C)", 0.0, 200.0, 80.0)
    front_inner_temp = st.number_input("Front Inner Temp (°C)", 0.0, 200.0, 90.0)
    rear_inner_temp = st.number_input("Rear Inner Temp (°C)", 0.0, 200.0, 90.0)

    # Categorical inputs (adjust options based on your data)
    tire_compound = st.selectbox("Tire Compound", ['Soft', 'Medium', 'Hard'])
    driving_style = st.selectbox("Driving Style", ['Aggressive', 'Normal', 'Conservative'])
    track = st.selectbox("Track", ['Monza', 'Monaco', 'Red Bull Ring'])

    submit_button = st.form_submit_button(label='Predict Degradation Risk')

# Process prediction when form is submitted
if submit_button:
    # Collect inputs into a DataFrame
    input_data = pd.DataFrame({
        'Throttle': [throttle],
        'Brake': [brake],
        'Speed': [speed],
        'Surface_Roughness': [surface_roughness],
        'Ambient_Temperature': [ambient_temperature],
        'Lateral_G_Force': [lateral_g_force],
        'Longitudinal_G_Force': [longitudinal_g_force],
        'Tire_Friction_Coefficient': [tire_friction_coefficient],
        'Tire_Tread_Depth': [tire_tread_depth],
        'force_on_tire': [force_on_tire],
        'front_surface_temp': [front_surface_temp],
        'rear_surface_temp': [rear_surface_temp],
        'front_inner_temp': [front_inner_temp],
        'rear_inner_temp': [rear_inner_temp],
        'Tire_Compound': [tire_compound],
        'Driving_Style': [driving_style],
        'Track': [track]
    })

    # Preprocess: Split num/cat, encode, scale
    X_num = input_data[numerical_features]
    X_cat = input_data[categorical_features]
    X_cat_encoded = encoder.transform(X_cat)
    X = np.hstack((X_num.values, X_cat_encoded))
    X_scaled = scaler.transform(X)

    # Predict
    pred_class_encoded = xgb_model.predict(X_scaled)[0]
    pred_class = le.inverse_transform([pred_class_encoded])[0]
    pred_probs = xgb_model.predict_proba(X_scaled)[0]

    # Display results
    st.success(f"Predicted Degradation Risk: **{pred_class}**")
    st.write(f"Probabilities: Safe={pred_probs[0]:.2f}, Medium={pred_probs[1]:.2f}, Critical={pred_probs[2]:.2f}")
