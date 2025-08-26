import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

#### Load the model
model = tf.keras.models.load_model('model.h5')
### Load the one-hot encoder
with open('onehot_encoder_geo.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)
### Load the lebal_encoder_gender
with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)
### Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

## Streamlit App
st.title("Customer Churn Prediction")

##User Input
geography = st.selectbox("Geography", label_encoder_geo.categories_[0].tolist())
gender = st.selectbox('Gender', gender_encoder.classes_.tolist())
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, value=1000.0)
credit_score = st.slider('Credit Score', 300, 850, 600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=1000000.0)
tenure = st.slider('Tenure (years)', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode the geography
geo_enocded = label_encoder_geo.transform([[geography]]).toarray()
geo_enocded_df = pd.DataFrame(geo_enocded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data, geo_enocded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button('Predict Churn'):
    prediction = model.predict(input_data_scaled)
    churn_prob = prediction[0][0]
    if churn_prob > 0.5:
        st.error(f'The customer is likely to churn with a probability of {churn_prob:.2f}')
    else:
        st.success(f'The customer is unlikely to churn with a probability of {churn_prob:.2f}')