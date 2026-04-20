from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(features=pred_df)
        #return render_template('home.html', results=prediction[0])
        return render_template('home.html', results=round(float(prediction[0]), 2))

if __name__ == "__main__":
    application.run(host='0.0.0.0')