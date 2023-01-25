# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 20:32:59 2023

@author: melih
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/melih/Desktop/Data Science(Yapay Zeka)/Deploying machine learning model with spyder/trained_model.sav','rb'))

#Creating a function for prediction
def heart_attack_pred(input_data):

    #changing the input data to numpy array
    np_input_data = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    reshaped_input_data= np_input_data.reshape(1,-1)

    prediction = loaded_model.predict(reshaped_input_data)
    print("Our prediction: ",prediction)

    if (prediction[0]==0):
      return "No risk for heart attack"
    else:
      return "Patient may be exposed heart attack"
  
def main():
    #giving a title
    st.title('Heart Attack Prediction Web App')
    
    # getting the input data from the user
    
    age = st.text_input("Age of the patient")
    sex = st.text_input("Sex of the patient")
    cp = st.text_input("Chest Pain Type (1-Typical Angina, 2-Atypical Angina, 3-Non-anginal pain, 4-asymtomatic")
    trtbps = st.text_input("Resting blood pressure (in mm Hg)")
    chol = st.text_input("Cholestoral in mg/dl fetched via BMI sensor")
    fbs = st.text_input("(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")
    restecg = st.text_input("Resting electrocardiographic results ( 0: normal,1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV,2: showing probable or definite left ventricular hypertrophy by Estes' criteria")
    thalachh = st.text_input("Maximum heart rate achieved")
    exng = st.text_input("Exercise induced angina (1 = yes; 0 = no)")
    oldpeak = st.text_input("ST depression induced by exercise relative to rest")
    slp = st.text_input("The slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)")
    caa = st.text_input("Number of major vessels (0-3)")
    thall = st.text_input("3 = normal; 6 = fixed defect; 7 = reversable defect")
    
    #Code for Prediction
    heartAttack = ''
    
    #Creating a Predict Button
    if st.button('Heart Attack Test Result'):
        heartAttack = heart_attack_pred([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])
    
    st.success(heartAttack)
    

if __name__ == '__main__':
    main()
    
    

    