from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipepline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicted',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Year = request.form.get('Year'),
            Present_Price = request.form.get('Present_Price'),
            Kms_Driven = request.form.get('Kms_Driven'),
            Fuel_Type = request.form.get('Fuel_Type'),
            Seller_Type = request.form.get('Seller_Type'),
            Transmission = request.form.get('Transmission'),
            Owner = request.form.get('Owner')
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipepline()
        selling_price = predict_pipeline.predict(pred_df)
        return render_template('home.html',selling_price=round(selling_price[0],2))
    
if __name__=='__main__':
    app.run(debug=True)