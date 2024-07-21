# tests/test_deploy.py

import pytest
from flask import json
from deploy import app, preprocess_input
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'NHT Detection API is working!'

@patch('deploy.model')
@patch('deploy.preprocess_input')
def test_predict(mock_preprocess_input, mock_model, client):
    """Test the /predict endpoint"""
    
    # Mock the preprocess_input function
    mock_preprocess_input.return_value = [[1.0, 2.0, 3.0]]  # Mocked preprocessed data

    # Mock the model prediction
    mock_model.predict.return_value = [1]  # Mocked prediction
    mock_model.predict_proba.return_value = [[0.1, 0.9]]  # Mocked prediction probabilities

    # Mock LabelEncoder inverse_transform
    with patch('deploy.le_target.inverse_transform') as mock_inverse_transform:
        mock_inverse_transform.return_value = ['NHT']
        
        # Example input data
        input_data = {
            "session_id": "be73c8d1b836170a21529a1b23140f8e",
            "epoch_ms": 1.520280e+12,
            "country_by_ip_address": "US",
            "region_by_ip_address": "CA",
            "url_without_parameters": "https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/",
            "referrer_without_parameters": "",
            "visitor_recognition_type": "ANONYMOUS"
        }
        
        response = client.post('/predict', data=json.dumps(input_data), content_type='application/json')
        
        assert response.status_code == 200
        assert response.json == {"prediction": ["NHT"]}
    
@patch('deploy.preprocess_input')
def test_predict_invalid_input(mock_preprocess_input, client):
    """Test the /predict endpoint with invalid input"""
    
    # Mock the preprocess_input function
    mock_preprocess_input.side_effect = ValueError("Feature shape mismatch")
    
    # Example invalid input data
    input_data = {
        "session_id": "be73c8d1b836170a21529a1b23140f8e",
        "epoch_ms": 1.520280e+12,
        "country_by_ip_address": "US",
        "region_by_ip_address": "CA",
        "url_without_parameters": "https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/",
        "referrer_without_parameters": "",
        "visitor_recognition_type": "ANONYMOUS"
    }
    
    response = client.post('/predict', data=json.dumps(input_data), content_type='application/json')
    
    assert response.status_code == 400
    assert response.json == {"error": "Feature shape mismatch"}

@patch('deploy.preprocess_input')
def test_predict_key_error(mock_preprocess_input, client):
    """Test the /predict endpoint with missing key"""
    
    # Mock the preprocess_input function
    mock_preprocess_input.return_value = [[1.0, 2.0, 3.0]]
    
    # Mock KeyError during prediction
    with patch('deploy.model.predict') as mock_predict:
        mock_predict.side_effect = KeyError("Missing expected key")
        
        # Example input data with missing key
        input_data = {
            "session_id": "be73c8d1b836170a21529a1b23140f8e",
            "epoch_ms": 1.520280e+12,
            "country_by_ip_address": "US",
            "region_by_ip_address": "CA",
            "url_without_parameters": "https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/",
            # 'referrer_without_parameters' is missing
            "visitor_recognition_type": "ANONYMOUS"
        }
        
        response = client.post('/predict', data=json.dumps(input_data), content_type='application/json')
        
        assert response.status_code == 400
        assert response.json == {"error": "KeyError: Missing expected key"}

@patch('deploy.preprocess_input')
def test_predict_unexpected_error(mock_preprocess_input, client):
    """Test the /predict endpoint with an unexpected error"""
    
    # Mock the preprocess_input function
    mock_preprocess_input.return_value = [[1.0, 2.0, 3.0]]
    
    # Mock an unexpected error during prediction
    with patch('deploy.model.predict') as mock_predict:
        mock_predict.side_effect = Exception("Unexpected error")
        
        input_data = {
            "session_id": "be73c8d1b836170a21529a1b23140f8e",
            "epoch_ms": 1.520280e+12,
            "country_by_ip_address": "US",
            "region_by_ip_address": "CA",
            "url_without_parameters": "https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/",
            "referrer_without_parameters": "",
            "visitor_recognition_type": "ANONYMOUS"
        }
        
        response = client.post('/predict', data=json.dumps(input_data), content_type='application/json')
        
        assert response.status_code == 500
        assert response.json == {"error": "Unexpected error occurred"}

