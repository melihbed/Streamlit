# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users\melih/Desktop/Data Science(Yapay Zeka)/Deploying machine learning model with spyder/trained_model.sav','rb'))


input_data = (18,1 ,0 ,130,131,0,1,115,1,1.2,1,1,3)


#changing the input data to numpy array
np_input_data = np.asarray(input_data)

# reshape the array as we are predicting for one instance
reshaped_input_data= np_input_data.reshape(1,-1)

prediction = loaded_model.predict(reshaped_input_data)
print("Our prediction: ",prediction)

if (prediction[0]==0):
  print("No risk for heart attack")
else:
  print("Patient may be exposed heart attack")