#[python -m streamlit run app.py] RUN THIS IN TERMINAL FOR THE STREAMLIT LINK

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


# Load model and feature columns
model = joblib.load('xgb_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Used Bike Price Predictor")

# Input fields
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2023, value=2018)
kms_driven = st.number_input("Kilometers Driven", min_value=100, max_value=200000, value=15000)
mileage = st.number_input("Mileage (kmpl)", min_value=10.0, max_value=100.0, value=45.0)
power = st.number_input("Power (bhp)", min_value=5.0, max_value=100.0, value=15.0)
cc = st.number_input("Engine CC", min_value=50, max_value=2000, value=150)

owner = st.selectbox("Ownership", ['first owner', 'second owner', 'third owner', 'fourth owner or more'])
location = st.selectbox("Location", [col.replace("location_", "") for col in model_columns if col.startswith("location_")])
brand = st.selectbox("Brand", [col.replace("brand_", "") for col in model_columns if col.startswith("brand_")])

# Feature engineering
bike_age = datetime.now().year - year
brand_col = f"brand_{brand}"
loc_col = f"location_{location}"
owner_col = f"owner_{owner}"

# Build input DataFrame
input_dict = {
    'kms_driven': kms_driven,
    'mileage': mileage,
    'power': power,
    'cc': cc,
    'bike_age': bike_age,
}

# One-hot columns
for col in model_columns:
    if col.startswith("brand_") or col.startswith("location_") or col.startswith("owner_"):
        input_dict[col] = 1 if col in [brand_col, loc_col, owner_col] else 0

# Ensure all model columns are present
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Price"):
    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)
    st.success(f" Estimated Price: ₹{int(predicted_price):,}")
