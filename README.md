# Predicting-Customer-Churn-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.drop_duplicates()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'customerID':
            df[col] = le.fit_transform(df[col])
    
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']
    return X, y
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Customer Churn Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

# Preprocess manually
gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0

input_data = pd.DataFrame([[gender, senior, partner, tenure, monthly, total]],
                          columns=['gender', 'SeniorCitizen', 'Partner', 'tenure', 'MonthlyCharges', 'TotalCharges'])

pred = model.predict(input_data)[0]
st.subheader("Prediction:")
st.write("Churn" if pred == 1 else "No Churn")
# Customer Churn Prediction

This project uses machine learning to predict whether a customer will churn based on historical data.

## 💡 Features
- Data preprocessing and transformation
- Multiple model training and evaluation
- Deployment using Streamlit Cloud

## 📦 Setup
```bash
pip install -r requirements.txt

