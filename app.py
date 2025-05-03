import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Import custom pipeline code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student Exam Performance", layout="centered")
st.title("Student Exam Performance Indicator")

with st.form("prediction_form"):
    gender = st.selectbox("Gender", ['female', 'male'])  # Lowercase
    ethnicity = st.selectbox("Race or Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
    parental_education = st.selectbox("Parental Level of Education", [
        "associate's degree", "bachelor's degree", "high school",
        "master's degree", "some college", "some high school"
    ])
    lunch = st.selectbox("Lunch Type", ['standard', 'free/reduced'])  # Exact match
    test_course = st.selectbox("Test Preparation Course", ['none', 'completed'])
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)
    
    submit = st.form_submit_button("Predict Math Score")

if submit:
    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()
        st.write("Input DataFrame:", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.success(f"Predicted Math Score: {results[0]}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
