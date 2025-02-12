import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load encoders and models
with open('ohe_dest.pkl', 'rb') as file:
    ohe_dest = pickle.load(file)

with open('ohe_origin.pkl', 'rb') as file:
    ohe_origin = pickle.load(file)

with open('ohe_unique.pkl', 'rb') as file:
    ohe_unique = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Flight Delay Prediction App")

Month = st.number_input('Provide the Month', min_value=1, max_value=12, step=1)
DayofMonth = st.number_input('Provide the Date of Month', min_value=1, max_value=31, step=1)
DepTime = st.number_input("Provide the Departure Time", min_value=0, max_value=2359, step=1)
unique_carrier = st.selectbox("Select the unique carrier:", ohe_unique.categories_[0])
origin = st.selectbox('Select the Origin Airport', ohe_origin.categories_[0])
dest = st.selectbox("Select the Destination Airport", ohe_dest.categories_[0])
distance = st.number_input("Provide the distance to destination:", min_value=1.0, step=1.0)

# Create input data dictionary
input_data = pd.DataFrame([{
    'Month': Month,
    'DayofMonth': DayofMonth,  # Fixed typo here
    'DepTime': DepTime,
    'Distance': distance
}])

# Encoding categorical variables
unique_df = ohe_unique.transform([[unique_carrier]]).toarray()
unique_pd = pd.DataFrame(unique_df, columns=ohe_unique.get_feature_names_out())

origin_df = ohe_origin.transform([[origin]]).toarray()
origin_pd = pd.DataFrame(origin_df, columns=ohe_origin.get_feature_names_out())

dest_df = ohe_dest.transform([[dest]]).toarray()
dest_pd = pd.DataFrame(dest_df, columns=ohe_dest.get_feature_names_out())

# Combine all data into a final DataFrame
final_df = pd.concat([input_data,unique_pd,origin_pd, dest_pd], axis=1)

# Scale the data
final_scaler = scaler.transform(final_df)

# Prediction Button
if st.button('Predict'):
    prob = model.predict(final_scaler)
    if prob<0.5:
        
        st.write(f"Yes,Flight is delay")

    else:
        st.write('No,Flight is not delay')