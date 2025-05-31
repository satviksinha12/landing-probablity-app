import streamlit as st
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

st.title("✈️ Aircraft Landing Probability Predictor (Bayesian Model)")

# Sample training dataset
data = pd.read_csv("Generated_Flight_Dataset.csv")


X = data.drop("Outcome", axis=1)
y = data["Outcome"]

encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
X_encoded = np.column_stack([encoders[col].transform(X[col]) for col in X.columns])

model = CategoricalNB()
model.fit(X_encoded, y)

# Input fields
aircraft = st.selectbox("Aircraft Type", data["Aircraft"].unique())
runway = st.selectbox("Runway Length", data["Runway_Length"].unique())
visibility = st.selectbox("Visibility", data["Visibility"].unique())
wind = st.selectbox("Wind", data["Wind"].unique())
load = st.selectbox("Load", data["Load"].unique())
weather = st.selectbox("Weather", data["Weather"].unique())

# Predict button
if st.button("Predict Landing Outcome"):
    input_dict = {
        "Aircraft": aircraft,
        "Runway_Length": runway,
        "Visibility": visibility,
        "Wind": wind,
        "Load": load,
        "Weather": weather
    }

    input_encoded = np.array([
        encoders[col].transform([input_dict[col]])[0] for col in X.columns
    ]).reshape(1, -1)

    probs = model.predict_proba(input_encoded)[0]
    class_labels = model.classes_

    st.subheader("🧠 Prediction Probabilities:")
    for label, prob in zip(class_labels, probs):
        st.write(f"**{label}**: {prob:.2%}")
