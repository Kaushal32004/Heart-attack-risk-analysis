import streamlit as st
import joblib
import pandas as pd

# Load saved models
model = joblib.load("heart_attack_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Set custom background and styles
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://png.pngtree.com/background/20230417/original/pngtree-medical-light-blue-background-picture-image_2446893.jpg");
        background-size: cover;
        background-attachment: fixed;
    }

    .main-title {
        font-size: 48px;
        color: crimson;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
        text-shadow: 2px 2px 5px white;
    }

    .input-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    label, .stSelectbox label, .stNumberInput label {
        color: #222 !important;
        font-weight: bold !important;
    }

    .result {
        font-size: 32px;
        color: white;
        background-color: rgba(255, 105, 180, 0.85);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>Heart Attack Risk Prediction</div>", unsafe_allow_html=True)

# Horizontal input layout using columns inside styled boxes
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        age = st.number_input("**Age**", min_value=1, max_value=120, value=45)
        smoking = st.selectbox("**Smoking**", [0, 1])
        bmi = st.number_input("**BMI**", min_value=10.0, max_value=50.0, value=25.0)
        diabetes = st.selectbox("**Diabetes**", [0, 1])
        cholesterol = st.number_input("**Cholesterol Level**", min_value=100.0, max_value=300.0, value=200.0)
        gender = st.selectbox("**Gender**", label_encoders['Gender'].classes_)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        alcohol = st.selectbox("**Alcohol Consumption**", [0, 1])
        hypertension = st.selectbox("**Hypertension**", [0, 1])
        resting_bp = st.number_input("**Resting BP**", min_value=80, max_value=200, value=120)
        heart_rate = st.number_input("**Heart Rate**", min_value=40, max_value=200, value=80)
        family_history = st.selectbox("**Family History**", [0, 1])
        physical_activity = st.selectbox("**Physical Activity Level**", label_encoders['Physical_Activity_Level'].classes_)
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        stress_level = st.selectbox("**Stress Level**", label_encoders['Stress_Level'].classes_)
        chest_pain = st.selectbox("**Chest Pain Type**", label_encoders['Chest_Pain_Type'].classes_)
        thalassemia = st.selectbox("**Thalassemia**", label_encoders['Thalassemia'].classes_)
        fasting_blood_sugar = st.selectbox("**Fasting Blood Sugar**", [0, 1])
        ecg_results = st.selectbox("**ECG Results**", label_encoders['ECG_Results'].classes_)
        max_heart_rate = st.number_input("**Max Heart Rate Achieved**", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("**Exercise Induced Angina**", [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)

# Encode categorical fields
gender_encoded = label_encoders['Gender'].transform([gender])[0]
physical_activity_encoded = label_encoders['Physical_Activity_Level'].transform([physical_activity])[0]
stress_level_encoded = label_encoders['Stress_Level'].transform([stress_level])[0]
chest_pain_encoded = label_encoders['Chest_Pain_Type'].transform([chest_pain])[0]
thalassemia_encoded = label_encoders['Thalassemia'].transform([thalassemia])[0]
ecg_results_encoded = label_encoders['ECG_Results'].transform([ecg_results])[0]

# Prepare input for model
sample_data = {
    'Age': age,
    'Gender': gender_encoded,
    'Smoking': smoking,
    'Alcohol_Consumption': alcohol,
    'Physical_Activity_Level': physical_activity_encoded,
    'BMI': bmi,
    'Diabetes': diabetes,
    'Hypertension': hypertension,
    'Cholesterol_Level': cholesterol,
    'Resting_BP': resting_bp,
    'Heart_Rate': heart_rate,
    'Family_History': family_history,
    'Stress_Level': stress_level_encoded,
    'Chest_Pain_Type': chest_pain_encoded,
    'Thalassemia': thalassemia_encoded,
    'Fasting_Blood_Sugar': fasting_blood_sugar,
    'ECG_Results': ecg_results_encoded,
    'Exercise_Induced_Angina': exercise_angina,
    'Max_Heart_Rate_Achieved': max_heart_rate
}

# Predict button
if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([sample_data])
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    prediction = model.predict(input_scaled)
    result = label_encoders['Heart_Attack_Risk'].inverse_transform(prediction)[0]

    st.markdown(f"<div class='result'>🧠 Predicted Heart Attack Risk: <strong>{result}</strong></div>", unsafe_allow_html=True)

    # Chatbot-like recommendation if risk is high
    if result.lower() in ['high', 'very high']:
        with st.expander("💬 Doctor Recommendation Chatbot"):
            st.markdown("⚠️ Based on your risk level, it's recommended to consult a cardiologist. Here are some nearby specialists you can consider:")
            st.markdown("""
            **🩺 Dr. Rajeev Kumar**  
            *Cardiologist, Apollo Hospital, Delhi*  
            Contact: +91-9876543210

            **🩺 Dr. Neha Sharma**  
            *Senior Consultant, Fortis Heart Institute, Gurgaon*  
            Contact: +91-9123456789

            **🩺 Dr. Sameer Bansal**  
            *Interventional Cardiologist, AIIMS Delhi*  
            Contact: +91-9988776655
            """)
            st.success("✅ It's crucial to seek medical advice. Stay healthy!")
