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
import logging
import time
import pandas as pd
import joblib

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

# Load the pre-trained model, scaler, LabelEncoder for target, and encoders for categorical features
model, scaler, le_target, columns_list = joblib.load('../models/best_model.pkl')
NUM_FEATURES = len(columns_list)

swagger = Swagger(app)

def determine_url_type(url):
    """
    Determines the type of URL based on its structure.
    Args:
        url (str): The URL to analyze.
    Returns:
        str: 'product' if '/p/' is in the URL, 'category' if '/l/' is in the URL, otherwise 'other'.
    """
    if '/p/' in url:
        return 'product'
    if '/l/' in url:
        return 'category'
    return 'other'

def encode_recognition_type(rec_type):
    """
    Encodes the visitor recognition type into numerical values.
    Args:
        rec_type (str): The visitor recognition type.
    Returns:
        int: Encoded value corresponding to the type, or -1 if type is not found in encoding_map.
    """
    encoding_map = {'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3}
    return encoding_map.get(rec_type, -1)  # Handle unexpected values gracefully

# Assuming new_data['url_without_parameters'] contains URLs as strings
def calculate_url_length(url):
    """
    Calculates the length of the URL.
    
    Args:
        url (str): The URL string.
        
    Returns:
        int: Length of the URL.
    """
    return len(url)

def check_referrer_presence(ref):
    """
    Check if the referrer is present.

    This function checks if the provided referrer string is null or empty,
    and returns 0 if it is, otherwise returns 1.

    Args:
        ref (str or None): The referrer string to check.

    Returns:
        int: 0 if referrer is null or empty, otherwise 1.
    """
    if pd.isnull(ref) or ref == '':
        return 0
    return 1

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
    new_data = pd.get_dummies(data=new_data, columns=['country_by_ip_address', 'region_by_ip_address', 'url_type'], drop_first=True)
    new_data = new_data.reindex(columns=columns_list, fill_value=0)
    new_data_scaled = scaler.transform(new_data)
    if new_data_scaled.shape[1] != NUM_FEATURES:
        raise ValueError(f"Feature shape mismatch, expected: {NUM_FEATURES}, got: {data.shape[1]}")

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
@cache.memoize(timeout=60)  # Cache results for 60 seconds
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
        prediction = model.predict(processed_data)
        predicted_labels = le_target.inverse_transform([prediction])
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
    app.run(debug=True)
