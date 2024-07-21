#!/usr/bin/env python
"""
Module: deploy.py

This script sets up a Flask web server to deploy a machine learning model. It includes
endpoints for model prediction based on incoming requests. The model is loaded using joblib,
and data preprocessing is handled using StandardScaler and LabelEncoder from scikit-learn.
Caching is implemented using Flask-Caching for improved performance.

Dependencies:
- logging
- time
- pandas
- joblib
- flask (Flask, request, jsonify)
- flask_caching (Cache)
- flasgger (Swagger, swag_from)

Usage:
Start the Flask server by running this script:
    $ python deploy.py

Endpoints:
- /predict: POST endpoint for making predictions using the deployed model.

"""
import sys
import os
import logging
import time
import pandas as pd
import numpy as np
import joblib
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import encode_recognition_type, calculate_url_length, check_referrer_presence, determine_url_type
from flask import Flask, request, jsonify
from flask_caching import Cache
from flasgger import Swagger, swag_from

app = Flask(__name__)
app.config.from_object(__name__)

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure logging for Flask-Caching
cache_logger = logging.getLogger('flask_caching')
cache_logger.setLevel(logging.INFO)

# Create a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it on the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
cache_logger.addHandler(console_handler)

# Get the absolute path to the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Adjust the path to load the model relative to the current script location
model_file = os.path.join(current_dir, '../models/best_model.pkl')
# Load the pre-trained model, scaler, LabelEncoder for target, and encoders for categorical features
model, scaler, le_target, le_region, le_country, columns_list = joblib.load(model_file)
NUM_FEATURES = len(columns_list)

swagger = Swagger(app)

# Function to preprocess input data
def preprocess_input(data):
    """
    Preprocesses the input data for prediction.
    
    Args:
        data (dict): Input data dictionary containing session details.
        
    Returns:
        numpy.ndarray: Preprocessed input data for prediction.
        
    Raises:
        ValueError: If there is a mismatch in the number of features.
    """
    # Initialize new_data with None or an empty DataFrame
    new_data = None
    if not isinstance(data, pd.DataFrame):
        new_data = pd.DataFrame(data, index=[0])

    new_data['url_length'] = new_data['url_without_parameters'].apply(calculate_url_length)
    new_data['url_type'] = new_data['url_without_parameters'].apply(determine_url_type)
    new_data['referrer_present'] = new_data['referrer_without_parameters'].apply(check_referrer_presence)
    new_data['visitor_recognition_type_encoded'] = new_data['visitor_recognition_type'].apply(encode_recognition_type)
    new_data['url_type'] = new_data['url_type'].astype('category').cat.codes

    # Handling unseen labels
    def safe_transform(label_encoder, data):
        seen_classes = set(label_encoder.classes_)
        return [label_encoder.transform([x])[0] if x in seen_classes else -1 for x in data]

    new_data['region_numeric'] = safe_transform(le_region, new_data['region_by_ip_address'])
    new_data['country_numeric'] = safe_transform(le_country, new_data['country_by_ip_address'])

    # Reindex to match model's expected feature columns
    new_data = new_data.reindex(columns=columns_list, fill_value=0)
    print('new_data:',new_data)
    # Scale the features
    new_data_scaled = scaler.transform(new_data)

    if new_data_scaled.shape[1] != NUM_FEATURES:
        raise ValueError(f"Feature shape mismatch, expected: {NUM_FEATURES}, got: {new_data_scaled.shape[1]}")

    return new_data_scaled

# Health Check Endpoint
@app.route('/', methods=['GET'])
def hello():
    """
    Health Check Endpoint
    
    Returns:
        str: Status message indicating API is working.
    """
    return 'NHT Detection API is working!'

# Predict Endpoint
@app.route('/predict', methods=['POST'])
@swag_from({
    'summary': 'Predict NHT',
    'description': 'Predict if the given input is NHT or not.',
    'parameters': [
        {
            'name': 'input_data',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string'},
                    'epoch_ms': {'type': 'number'},
                    'country_by_ip_address': {'type': 'string'},
                    'region_by_ip_address': {'type': 'string'},
                    'url_without_parameters': {'type': 'string'},
                    'referrer_without_parameters': {'type': 'string'},
                    'visitor_recognition_type': {'type': 'string'}
                },
                'example': {
                    'session_id': 'be73c8d1b836170a21529a1b23140f8e',
                    'epoch_ms': 1.520280e+12,
                    'country_by_ip_address': 'US',
                    'region_by_ip_address': 'CA',
                    'url_without_parameters': 'https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/',
                    'referrer_without_parameters': '',
                    'visitor_recognition_type': 'ANONYMOUS'
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Prediction result',
            'schema': {
                'type': 'object',
                'properties': {
                    'prediction': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        },
        '400': {
            'description': 'Invalid input',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    }
})
@cache.memoize(timeout=5)  # Cache results for 5 seconds
def predict():
    """
    Endpoint to predict if the input data is NHT or not.
    
    Returns:
        tuple: JSON response with prediction result or error message, and HTTP status code.
    """
    try:
        start_time = time.time()
        input_data = request.json
        processed_data = preprocess_input(input_data)
        print('processed_data:',processed_data)
        prediction = model.predict(processed_data)
        print('prediction:',prediction)
        predicted_labels = le_target.inverse_transform(prediction)
        print('predicted_labels:',predicted_labels)
        elapsed_time = time.time() - start_time
        app.logger.info('Request processed in %.4f seconds', elapsed_time)
        return jsonify({'prediction': predicted_labels.tolist()}), 200
    except ValueError as ve:
        app.logger.error('ValueError: %s', str(ve))
        return jsonify({'error': str(ve)}), 400
    except KeyError as ke:
        app.logger.error('KeyError: %s', str(ke))
        return jsonify({'error': 'KeyError: Missing expected key'}), 400
    except Exception as e:
        app.logger.error('Unexpected error: %s', str(e))
        return jsonify({'error': 'Unexpected error occurred'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
