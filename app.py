# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 01:40:05 2021

@author: SIDHARTH
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app= Flask(__name__) #from which point u want to start the application
pickle_in =open('classifier.pkl' ,'rb')
classifier=pickle.load(pickle_in)


@app.route('/') #root page its a decorator
def welcome():
    return "Welcome to the page"

@app.route('/predict')
def predict_note_authen():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness') 	
    curtosis=request.args.get('curtosis') 	
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is"+ str(prediction)


#we need to externally post these methods then only it can be predicted
@app.route('/predict_file', methods=["POST"] ) 
def predict_note_file():
    df_test= pd.read_csv(request.files.get("file")) #file variable has the data of testFile.csv
    prediction=classifier.predict(df_test)
    return "The predicted values of the csv is"+ str(list(prediction))



if __name__=='__main__':
    app.run()

