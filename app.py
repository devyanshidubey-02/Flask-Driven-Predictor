import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import PredictPipeline, CustomData  # Import the classes

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Store Sales Prediction App", layout="centered")

# Title of the app
st.title("Store Sales Prediction App")

# Description for the app
st.write("""
This app predicts the store sales based on the following parameters:
- Store Area
- Items Available
- Daily Customer Count
""")

# Input fields for the custom data
store_area = st.number_input("Store Area (sq. ft)", min_value=0.0, value=1000.0, step=0.1)
items_available = st.number_input("Items Available", min_value=0, value=1000, step=1)
daily_customer_count = st.number_input("Daily Customer Count", min_value=0, value=100, step=1)

# Create a button to trigger prediction
if st.button("Predict Store Sales"):
    custom_data = CustomData(store_area, items_available, daily_customer_count)
    data = custom_data.get_data_as_dataframe()
    prediction_pipeline = PredictPipeline()
    prediction = prediction_pipeline.predict(data)
    st.write(f"**Predicted Store Sales:** {prediction[0]:.2f}")