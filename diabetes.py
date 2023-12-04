import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.markdown('<h1 style="font-size: 36px;">Diabetes Prediction</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 style="font-size: 24px;">Model Parameters</h2>', unsafe_allow_html=True)

# Display parameter names with larger font
st.sidebar.markdown('<h3 style="font-size: 18px;">Pregnancies:</h3>', unsafe_allow_html=True)
preg = st.sidebar.number_input('', min_value=0, max_value=20, value=5)

st.sidebar.markdown('<h3 style="font-size: 18px;">Glucose:</h3>', unsafe_allow_html=True)
glu = st.sidebar.number_input('', min_value=10, max_value=200, value=100)

st.sidebar.markdown('<h3 style="font-size: 18px;">BP:</h3>', unsafe_allow_html=True)
bp = st.sidebar.slider('', min_value=10, max_value=200, value=80)

st.sidebar.markdown('<h3 style="font-size: 18px;">Skin Thickness:</h3>', unsafe_allow_html=True)
skin_t = st.sidebar.slider('', min_value=1, max_value=100, value=10)

st.sidebar.markdown('<h3 style="font-size: 18px;">Insulin:</h3>', unsafe_allow_html=True)
ins = st.sidebar.slider('', min_value=10, max_value=900, value=400)

st.sidebar.markdown('<h3 style="font-size: 18px;">BMI:</h3>', unsafe_allow_html=True)
bmi = st.sidebar.slider('', min_value=5.0, max_value=150.0, value=50.0)

st.sidebar.markdown('<h3 style="font-size: 18px;">Diabetes Ped. Func:</h3>', unsafe_allow_html=True)
dia_ped = st.sidebar.slider('', min_value=0.0, max_value=5.0, value=2.0)

st.sidebar.markdown('<h3 style="font-size: 18px;">Age:</h3>', unsafe_allow_html=True)
age = st.sidebar.slider('', min_value=1, max_value=200, value=50)

test_list = [preg, glu, bp, skin_t, ins, bmi, dia_ped, age]

df = pd.DataFrame([test_list], columns=['Pregnancies', 'Glucose', 'BP', 'Skin Thickness', 'Insulin', 'BMI', 'Dia_Ped_Func', 'Age'])


with open('scaler_model.pkl', 'rb') as sc1:
    scaler_model = pickle.load(sc1)

with open('st_pca.pkl', 'rb') as model_1:
    log_model = pickle.load(model_1)

def test_model(model, scaler, entry_list):
    test_array = np.array(entry_list)
    test_array = test_array.reshape(1,-1)
    
    # Apply standard scaler
    test_array_norm = scaler.transform(test_array)
    
    #Make prediction
    prediction = model.predict(test_array_norm)
    if prediction[0]==0:
        result= 'No Diabetes'
    else:
        result = 'Diabetes'
        
    return result

predict_button = st.sidebar.button('Predict Diabetes')

if predict_button:
    result = test_model(model=log_model, scaler=scaler_model, entry_list=test_list)

    st.write(df)
    st.write(result)
