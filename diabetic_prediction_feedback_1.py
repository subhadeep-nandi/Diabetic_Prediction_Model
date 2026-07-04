# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:58:04 2026

@author: SUBHADEEP NANDI
"""



import numpy as np
import pickle
import streamlit as st

# loaded_model = pickle.load(open(r"D:\JOURNEY OF DS\ML\PROJECTS\DIABETES_PRDICTION_PROJECT\trained_model.sav", 'rb'))
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "trained_model.sav"

with open(MODEL_PATH, "rb") as file:
    loaded_model = pickle.load(file)


# create a function for prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    # give a title for our webpage
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from users
    Pregnancies = st.text_input("Number of Pregnancies : ")
    Glucose = st.text_input("Glucose Level : ")
    BloodPressure = st.text_input("Blood Pressure value : ")
    SkinThickness = st.text_input("Skin Thickness value : ")
    Insulin = st.text_input("Insulin Level : ")
    BMI = st.text_input("BMI value : ") 
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value : ")
    Age = st.text_input("Age of the Person : ")
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__ == "__main__":
    main()