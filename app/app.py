import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify


#load model here 
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('victorizer.pkl', 'rb'))

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
        if my_prediction == '1':
            my_prediction = "Class A"
        if my_prediction == '2':
            my_prediction = "Class B"
        if my_prediction == '3':
            my_prediction = "Class C"
        if my_prediction == '4':
            my_prediction = "Class D"
        if my_prediction == '5':
            my_prediction = "TBA"
        elif my_prediction == '6':
            my_prediction = "NA"

    
    return render_template('result.html', prediction = my_prediction) 

if __name__ == "__main__": 
    app.run(debug=True) 
