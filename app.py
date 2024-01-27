# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:54:00 2024

@author: burha
"""

import pickle
from flask import Flask, request, app, jsonify, render_template, url_for,json
import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')

def home ():
    return render_template('home.html')

@app.route('/predick_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print(new_data) 
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',  methods = ['POST'])

def predict():
    data =[float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0] 
    return render_template("home.html",prediction_text = "the house price prediction is {}".format(output))
    
if __name__ == '__main__':
        app.run(debug=False, port=8002)

    
    