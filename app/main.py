import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify


#load model here 
app = Flask(__name__)
model = pickle.load(open('app/model.pkl', 'rb'))
vectorizer = pickle.load(open('app/victorizer.pkl', 'rb'))

@app.route('/') 
def home(): 
    return render_template('home.html') 

@app.route('/predict', methods=['POST']) 
def predict(): 
    '''
    For rendering results on HTML GUI
    '''
    if request.method =='POST':
        message = request.form['message']
        data=[message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = model.predict(vect)
        category = "NA"
        if my_prediction == '1':
            category = "Class A"
        elif my_prediction == '2':
            category = "Class B"
        elif my_prediction == '3':
            category = "Class C"
        elif my_prediction == '4':
            category = "Class D"
        elif my_prediction == '5':
            category = "TBA"
        elif my_prediction == '6':
            category = "NA"

    
    return render_template('result.html', prediction = category) 
