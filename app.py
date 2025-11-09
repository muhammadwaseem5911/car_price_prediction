from flask import Flask,request,jsonify
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
with open('car_model.pkl','rb') as f:
    model=pickle.load(f)

COLUMNS=[
 'full_name',
 'year',
 'seller_type',
 'km_driven',
 'fuel_type',
 'transmission_type',
 'mileage',
 'engine',
 'max_power',
 'seats']
@app.route('/')
def home():
    return "car price prediction api is running"

@app.route('/predict',methods=['POST'])
def predict():   
    data =request.get_json()
    print('data received',data)
    df = pd.DataFrame([data])
    df=df[COLUMNS]
    prediction = model.predict(df)
    return jsonify({'predicted_selling_price':float(prediction[0])})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
