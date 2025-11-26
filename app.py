import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("resnet_eeg_model.h5")
scaler = joblib.load("scaler_eeg.pkl")

st.title("Epileptic Seizure Detection (1D ResNet)")

uploaded_file = st.file_uploader("Upload EEG CSV (178 values)", type=["csv"])
input_data = None

# -----------------------
# LOAD CSV
# -----------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    
    if df.shape[1] != 178:
        st.error(f"Invalid input shape: {df.shape}. Expected 1 x 178 CSV.")
    else:
        eeg_values = df.values.reshape(1, -1)
        input_data = eeg_values
        st.write("EEG Loaded:", input_data)

else:
    # -----------------------
    # PASTE VALUES
    # -----------------------
    eeg_text = st.text_area("Or paste 178 comma-separated values")
    if eeg_text:
        try:
            eeg_values = np.array([float(x) for x in eeg_text.split(",")])
            if len(eeg_values) != 178:
                st.error("You must enter exactly 178 values.")
            else:
                input_data = eeg_values.reshape(1, -1)
                st.write("EEG Loaded:", input_data)
        except:
            st.error("Invalid numbers. Check your input.")

# -----------------------
# PREDICT
# -----------------------
if input_data is not None:
    # Apply scaler
    scaled = scaler.transform(input_data)

    # Reshape for ResNet → (1, 178, 1)
    scaled = scaled.reshape(1, 178, 1)

    # Predict
    prob = model.predict(scaled)[0][0]

    label = "SEIZURE" if prob >= 0.5 else "NON-SEIZURE"

    st.subheader(f"Prediction: {label}")
    st.subheader(f"Probability: {prob:.4f}")
