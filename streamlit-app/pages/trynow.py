import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Machine Learning-Based Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.title("Machine Learning-Based Customer Churn Prediction")
st.divider()


# ================== Load All Model =======================4

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

dt_model_path = BASE_DIR / "models" / "DecisionTreeModel.pkl"
lr_model_path = BASE_DIR / "models" / "Logistic_Regression.pkl"
rf_model_path = BASE_DIR / "models" / "RandomForest.pkl"
knn_model_path = BASE_DIR / "models" / "KNN.pkl"

Dt_model = joblib.load(dt_model_path)
KNN_model = joblib.load(knn_model_path)
Logistic_Regression_model = joblib.load(lr_model_path)
Random_Forest_model = joblib.load(rf_model_path)

# ================== User Form ============================
with st.form("Try Now Form"):
    Name = st.text_input("Enter Your Name")
    gender = st.selectbox(
        "What is your gender?",
        ["Male", "Female"],
        index=None,
        placeholder="Select Gender"
    )

    SeniorCitizen = st.selectbox(
        "Are you a senior citizen (60+ years)?",
        [0, 1],
        index=None,
        placeholder="Select Option"
    )

    Partner = st.selectbox(
        "Do you have a partner/spouse?",
        ["Yes", "No"],
        index=None,
        placeholder="Select Option"
    )

    tenure = st.number_input(
        "How many months have you been using the service?",
        min_value=0,
        max_value=100,
        step=1,
        value=None,
        placeholder="Enter number of months"
    )
    
    Dependents = st.selectbox(
        "Do you have dependents?",
        ["Yes", "No"],
        index=None,
        placeholder="Select Option"
    )

    PhoneService = st.selectbox(
        "Do you have phone service?",
        ["Yes", "No"],
        index=None,
        placeholder="Select Option"
    )

    MultipleLines = st.selectbox(
        "Do you use multiple phone lines?",
        ["Yes", "No", "No phone service"],
        index=None,
        placeholder="Select Option"
    )

    InternetService = st.selectbox(
        "What type of internet service do you use?",
        ["DSL", "Fiber optic", "No"],
        index=None,
        placeholder="Select Internet Service"
    )

    OnlineSecurity = st.selectbox(
        "Do you have online security service?",
        ["Yes", "No", "No internet service"],
        index=None,
        placeholder="Select Option"
    )

    OnlineBackup = st.selectbox(
        "Do you have online backup service?",
        ["Yes", "No", "No internet service"],
        index=None,
        placeholder="Select Option"
    )

    DeviceProtection = st.selectbox(
        "Do you have device protection service?",
        ["Yes", "No", "No internet service"],
        index=None,
        placeholder="Select Option"
    )

    TechSupport = st.selectbox(
        "Do you have technical support service?",
        ["Yes", "No", "No internet service"],
        index=None,
        placeholder="Select Option"
    )

    StreamingTV = st.selectbox(
        "Do you use TV streaming service?",
        ["Yes", "No", "No internet service"],
        index=None,
        placeholder="Select Option"
    )

    StreamingMovies = st.selectbox(
        "Do you use movie streaming service?",
        ["Yes", "No", "No internet service"],
        index=None,
        placeholder="Select Option"
    )

    Contract = st.selectbox(
        "What type of contract do you have?",
        ["Month-to-month", "One year", "Two year"],
        index=None,
        placeholder="Select Contract Type"
    )

    PaperlessBilling = st.selectbox(
        "Do you receive bills electronically (paperless billing)?",
        ["Yes", "No"],
        index=None,
        placeholder="Select Option"
    )

    PaymentMethod = st.selectbox(
        "Which payment method do you use?",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        index=None,
        placeholder="Select Payment Method"
    )

    MonthlyCharges = st.number_input(
        "What is your monthly bill amount (₹)?",
        min_value=0.0,
        step=0.01,
        value=None,
        placeholder="Enter monthly charges"
    )

    algorithms = st.selectbox(
        "Select an Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "K Nearest Neibours",
            "Random Forest",
        ],
        index=None,
        placeholder="Select Model"
    )

    btn = st.form_submit_button("Submit")




required_fields = [
    gender, SeniorCitizen, Partner, Dependents,
    PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod, algorithms,
    Name, MonthlyCharges, tenure
]

if btn:

    if any(x is None for x in required_fields):
        st.error("Please fill all fields before prediction.")
        st.stop()

    # Prediction

    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": tenure * MonthlyCharges
    })

    if algorithms == "Decision Tree":
        prediction = Dt_model.predict(input_df)[0]
        probability = Dt_model.predict_proba(input_df)[0][1]

        st.success(
            f"Churn Probability: {probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is likely to stay.")

    elif algorithms == "K Nearest Neibours":
        prediction = KNN_model.predict(input_df)[0]
        probability = KNN_model.predict_proba(input_df)[0][1]

        st.success(
            f"Churn Probability: {probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is likely to stay.")

    elif algorithms == "Random Forest":
        prediction = Random_Forest_model.predict(input_df)[0]
        probability = Random_Forest_model.predict_proba(input_df)[0][1]

        st.success(
            f"Churn Probability: {probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is likely to stay.")

    else:
        prediction = Logistic_Regression_model.predict(input_df)[0]
        probability = Logistic_Regression_model.predict_proba(input_df)[0][1]

        st.success(
            f"Churn Probability: {probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is likely to stay.")