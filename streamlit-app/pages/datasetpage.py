import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Coustomer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Coustomer Churn Prediction")
st.divider()

df = pd.read_csv("../data/Churn-Prediction-dataset.csv")

st.write(df)