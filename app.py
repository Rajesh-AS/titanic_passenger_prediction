import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("titanic_model.pkl", "rb"))
le_sex, le_embarked = pickle.load(open("encoders.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")

p_class = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sib_sp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Encode inputs
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]

input_data = np.array([[p_class, sex_encoded, age, sib_sp, parch, fare, embarked_encoded]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("💀 Passenger Did Not Survive")
