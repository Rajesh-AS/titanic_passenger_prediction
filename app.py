import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")

# -------------------------------
# Safe file paths (VERY IMPORTANT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# -------------------------------
# Load model & encoders safely
# -------------------------------
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    le_sex, le_embarked = pickle.load(open(ENCODER_PATH, "rb"))
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# -------------------------------
# User Inputs
# -------------------------------
p_class = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sib_sp = st.number_input("Siblings / Spouses", 0, 10, 0)
parch = st.number_input("Parents / Children", 0, 10, 0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]

input_data = np.array([[p_class, sex_encoded, age, sib_sp, parch, fare, embarked_encoded]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("💀 Passenger Did Not Survive")
