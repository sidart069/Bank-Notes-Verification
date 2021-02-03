# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:02:01 2021

An implementation using Swagger Library in flask
to create an interactive frontend

@author: Sidharth
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger 
from flasgger import Swagger


app= Flask(__name__) #from which point u want to start the application
Swagger(app) #indicating Swagger to genearte UI for our app


pickle_in =open('classifier.pkl' ,'rb')
classifier=pickle.load(pickle_in)


@app.route('/') #root page its a decorator
def welcome():
    return "Welcome to the page"

@app.route('/predict', methods=["GET"])
def predict_note_authen():
    
    """AUTHENTICATING BANK NOTES
    Made by using the docstrings specifications.
    ---
    parameters:
        -  name: variance
           in: query
           type: number
           required: true
        -  name: skewness
           in: query
           type: number
           required: true
        -  name: curtosis
           in: query
           type: number
           required: true
        -  name: entropy
           in: query
           type: number
           required: true    
    responses:
        200:
            description: The output values
    """
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness') 	
    curtosis=request.args.get('curtosis') 	
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is"+ str(prediction)


#we need to externally post these methods then only it can be predicted
@app.route('/predict_file', methods=["POST"] ) 
def predict_note_file():
    
    """AUTHENTICATING BANK NOTES
    Made by using the docstrings specifications.
    ---
    parameters:
        -  name: file
           in: formData
           type: file
           required: true
   
    responses:
        200:
            description: The output values
    """
    
    df_test= pd.read_csv(request.files.get("file")) #file variable has the data of testFile.csv
    prediction=classifier.predict(df_test)
    return "The predicted values of the csv is"+ str(list(prediction))



if __name__=='__main__':
    app.run()

