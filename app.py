from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

scaler=pickle.load(open("models/STD_Scaler.pkl", "rb"))
model = pickle.load(open("models/GaussianNB.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('HOME.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        LIMIT_BAL=int(request.form.get("LIMIT_BAL"))
        SEX = float(request.form.get('SEX'))
        EDUCATION = float(request.form.get('EDUCATION'))
        MARRIAGE = float(request.form.get('MARRIAGE'))
        AGE = float(request.form.get('AGE'))
        PAY_0 = float(request.form.get('PAY_0'))
        PAY_2 = float(request.form.get('PAY_2'))
        PAY_3 = float(request.form.get('PAY_3'))
        PAY_4 = float(request.form.get('PAY_4'))
        PAY_5 = float(request.form.get('PAY_5'))
        PAY_6 = float(request.form.get('PAY_6'))
        BILL_AMT1 = float(request.form.get('BILL_AMT1'))
        BILL_AMT2 = float(request.form.get('BILL_AMT2'))
        BILL_AMT3 = float(request.form.get('BILL_AMT3'))
        BILL_AMT4 = float(request.form.get('BILL_AMT4'))
        BILL_AMT5 = float(request.form.get('BILL_AMT5'))
        BILL_AMT6 = float(request.form.get('BILL_AMT6'))
        PAY_AMT1 = float(request.form.get('PAY_AMT1'))
        PAY_AMT2 = float(request.form.get('PAY_AMT2'))
        PAY_AMT3 = float(request.form.get('PAY_AMT3'))
        PAY_AMT4 = float(request.form.get('PAY_AMT4'))
        PAY_AMT5 = float(request.form.get('PAY_AMT5'))
        PAY_AMT6 = float(request.form.get('PAY_AMT6'))

        new_data=scaler.transform([[LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6]])
        predict=model.predict(new_data)
       
        if predict[0] == 1 :
            result = 'FRAUD'
        else:
            result ='NOT A FRAUD'
            
        return render_template('RESULT.html',result=result)

    else:
        return render_template('WELCOME.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)