import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from joblib import dump, load

app = Flask(__name__)
model = load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [data for data in request.form.values()]
    columns = ['experiences_offered', 'room_type', 'accommodates', 'bathrooms',
       'bedrooms', 'beds', 'bed_type', 'security_deposit', 'cleaning_fee',
       'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
       'availability_30', 'availability_90', 'availability_365',
       'requires_license', 'instant_bookable', 'is_business_travel_ready',
       'distance', 'premium', 'prop_type', 'cancellation', 'host_is_foreigner',
       'neighbourhood']
    
    #requests = pd.DataFrame(features, columns=columns)
    
    prediction = model.predict(features)
    
    output = round(np.exp(prediction),3)
    
    return render_template('index.html',prediction_text = 'Listing price should be ${}'.format(output))
    
if __name__ == "__main__":
    app.run(debug=True)
    
