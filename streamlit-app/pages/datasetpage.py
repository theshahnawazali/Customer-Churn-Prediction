import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Machine Learning-Based Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Machine Learning-Based Customer Churn Prediction")
st.divider()

df = pd.read_csv("../data/Churn-Prediction-dataset.csv")

st.write(df)