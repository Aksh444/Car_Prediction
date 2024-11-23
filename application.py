from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the trained model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('used_cars.csv')  # Ensure the CSV file is in the same directory

@app.route('/', methods=['GET', 'POST'])
def index():
    # Extract unique values for dropdowns
    companies = sorted(car['brand'].unique())
    years = sorted(car['model_year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')  # Placeholder option for dropdown
    return render_template('index.html', companies=companies, years=years, fuel_types=fuel_types)

@app.route('/get_models', methods=['POST'])
@cross_origin()
def get_models():
    # Handle request to fetch car models for the selected company
    data = request.json
    selected_company = data.get('company')
    if not selected_company or selected_company == 'Select Company':
        return jsonify([])  # Return empty list if no valid company is selected

    # Fetch models for the selected company
    models = sorted(car[car['brand'] == selected_company]['model'].unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    # Get form data
    company = request.form.get('brand')
    car_model = request.form.get('model')
    year = int(request.form.get('model_year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('milage'))

    # Prepare data for prediction
    input_data = pd.DataFrame(
        columns=['model', 'brand', 'model_year', 'milage', 'fuel_type'],
        data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
    )

    # Make prediction
    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
