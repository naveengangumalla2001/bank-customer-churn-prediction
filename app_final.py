import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    model_path = "model.pkl"  # Change if your model file name is different

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    return joblib.load(model_path)

model = load_model()

# Title
st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the bank or stay.")

# Input Fields
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.slider("Tenure", 0, 10, 5)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

card_type = st.selectbox(
    "Card Type",
    ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]
)

complain = st.selectbox(
    "Customer Complaint",
    ["No", "Yes"]
)

points = st.number_input(
    "Point Earned",
    min_value=0,
    max_value=1000,
    value=100
)

# Preprocess Input
def preprocess_input():
    data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_crcard == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Complain": 1 if complain == "Yes" else 0,
        "Satisfaction Score": satisfaction,
        "Point Earned": points,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Card Type_GOLD": 1 if card_type == "GOLD" else 0,
        "Card Type_PLATINUM": 1 if card_type == "PLATINUM" else 0,
        "Card Type_SILVER": 1 if card_type == "SILVER" else 0
    }

    return pd.DataFrame([data])

# Prediction
if st.button("Predict Churn"):

    try:
        input_df = preprocess_input()

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = 0.0

        st.divider()

        st.subheader("Customer Details Sent To Model")
        st.dataframe(input_df)

        if prediction == 1:
            st.error(
                f"⚠️ High Churn Risk\n\n"
                f"The customer is likely to leave the bank.\n\n"
                f"Churn Probability: {probability:.2%}"
            )
        else:
            st.success(
                f"✅ Low Churn Risk\n\n"
                f"The customer is likely to stay with the bank.\n\n"
                f"Retention Probability: {(1 - probability):.2%}"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.info(
    "This prediction is generated using the trained machine learning model."
)
