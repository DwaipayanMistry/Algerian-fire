from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import pandas as pd
from flask import Flask,request, jsonify, render_template

application= Flask(__name__)
app=application

try:
    ridge_model = pickle.load(open('./models/ridge.pkl', 'rb'))
    scaler_model = pickle.load(open('./models/scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model files not found. Ensure 'models' directory exists and contains 'ridge.pkl' and 'scaler.pkl'.")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # 1. Retrieve data from form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # 2. Transform using the 9 features your scaler expects
        new_scaled_data = scaler_model.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        
        # 3. Predict
        prediction = ridge_model.predict(new_scaled_data)
        
        # 4. Extract the first value from the prediction array
        result_value = prediction[0]

        # 5. FIX: Pass the result to the template with a key name
        return render_template('predict.html', result=result_value)
    else:
        return render_template('predict.html')


if __name__=='__main__':
    app.run(host='0.0.0.0')
    # all the features ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes',
    #    'Region']
