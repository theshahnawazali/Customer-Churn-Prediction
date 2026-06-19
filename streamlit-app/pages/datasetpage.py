import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Machine Learning-Based Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Machine Learning-Based Customer Churn Prediction")
st.divider()


BASE_PATH = Path(__file__).resolve().parents[2]
df = pd.read_csv(BASE_PATH / "data" / "Churn-Prediction-dataset.csv")

st.write(df)