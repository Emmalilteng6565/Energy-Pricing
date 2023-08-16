import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib


# Load the trained models
gb_demand_model = joblib.load('gb_demand_model.joblib')
gb_price_model = joblib.load('gb_price_model.joblib')

# Load the Scalers
with open('scaler_demand.joblib', 'rb') as f:
    scaler_demand = joblib.load(f)
with open('scaler_price.joblib', 'rb') as f:
    scaler_price = joblib.load(f)

# Define the Streamlit app
def main():
    # Title
    st.title("Energy Demand and Price Prediction")

    # Input features
    rainfall = st.slider('Rainfall (mm)', min_value=0.0, max_value=50.0, step=0.1)
    solar_exposure = st.slider('Solar Exposure (MJ/m^2)', min_value=0.0, max_value=50.0, step=0.1)
    school_day = st.selectbox('School Day (1: Yes, 0: No)', options=[1, 0])
    holiday = st.selectbox('Holiday (1: Yes, 0: No)', options=[1, 0])
    day = st.selectbox('Day of the Month', options=list(range(1,32)))
    month = st.selectbox('Month', options=list(range(1,13)))
    year = st.selectbox('Year', options=list(range(2020, 2030)))

    # Create a dataframe from the inputs
    input_data = {'rainfall': [rainfall], 'solar_exposure': [solar_exposure], 'school_day': [school_day], 
                  'holiday': [holiday], 'day': [day], 'month': [month], 'year': [year]}
    input_df = pd.DataFrame(input_data)

    if st.button('Predict Demand and Price'):
        # Scale the input data for demand prediction
        input_df_scaled_demand = scaler_demand.transform(input_df)

        # Make a demand prediction
        demand_prediction = gb_demand_model.predict(input_df_scaled_demand)[0]
        st.markdown(f"**Predicted Demand: {demand_prediction:.2f} MWh**")

        # Add the demand prediction to the input data for price prediction
        input_df['demand'] = [demand_prediction]

        # Scale the input data for price prediction
        input_df_scaled_price = scaler_price.transform(input_df)

        # Make a price prediction
        price_prediction = gb_price_model.predict(input_df_scaled_price)[0]
        st.markdown(f"**Predicted Price: {price_prediction:.2f} AUD/MWh**")

if __name__ == "__main__":
    main()
