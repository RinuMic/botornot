from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flasgger import Swagger, swag_from

app = Flask(__name__)
app.config.from_object(__name__)

# Load the pre-trained model, scaler, LabelEncoder for target, and encoders for categorical features
# best_xgb, scaler, le_target, columns_list = joblib.load('nht_detection_model_with_label-final.pkl')
model, scaler, le_target, columns_list = joblib.load('nht_detection_model_with_label-final.pkl')
NUM_FEATURES = len(columns_list)

swagger = Swagger(app)

# Function to preprocess input data
def preprocess_input(data):
    # Convert new_data to a DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        new_data = pd.DataFrame(data, index=[0])

    # Feature engineering for the new data
    new_data['url_length'] = new_data['url_without_parameters'].apply(lambda url: len(url))
    new_data['url_type'] = new_data['url_without_parameters'].apply(lambda url: 'product' if '/p/' in url else ('category' if '/l/' in url else 'other'))
    new_data['referrer_present'] = new_data['referrer_without_parameters'].apply(lambda ref: 0 if pd.isnull(ref) or ref == '' else 1)
    new_data['visitor_recognition_type_encoded'] = new_data['visitor_recognition_type'].map({'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3})

    # One-hot encoding for specified features
    new_data = pd.get_dummies(data=new_data, columns=['country_by_ip_address', 'region_by_ip_address', 'url_type'], drop_first=True)

    # Realign columns with original training columns
    new_data = new_data.reindex(columns=columns_list, fill_value=0)

    # Standardize the data using the previously fitted scaler
    new_data_scaled = scaler.transform(new_data)

    if new_data_scaled.shape[1] != NUM_FEATURES:
        raise ValueError(f"Feature shape mismatch, expected: {NUM_FEATURES}, got: {data.shape[1]}")

    return new_data_scaled

@app.route('/', methods=['GET'])
def hello():
    """
    Health Check Endpoint
    ---
    responses:
      200:
        description: API is working
    """
    return 'NHT Detection API is working!'

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
def predict():
    try:
        input_data = request.json
        print('INPUT_DATA:',input_data)
        # Perform preprocessing
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        predicted_labels = le_target.inverse_transform([prediction])
        
        return jsonify({'prediction': predicted_labels.tolist()}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
