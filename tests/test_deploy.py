import pytest
import json
from deploy import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_health_check(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert rv.data == b'NHT Detection API is working!'

def test_predict_endpoint(client):
    input_data = {
        'session_id': 'be73c8d1b836170a21529a1b23140f8e',
        'epoch_ms': 1520280000000,
        'country_by_ip_address': 'US',
        'region_by_ip_address': 'CA',
        'url_without_parameters': 'https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/',
        'referrer_without_parameters': '',
        'visitor_recognition_type': 'ANONYMOUS'
    }
    rv = client.post('/predict', json=input_data)
    
    assert rv.status_code == 200
    response_data = json.loads(rv.data)
    assert 'prediction' in response_data
    assert isinstance(response_data['prediction'], list)
    assert len(response_data['prediction']) == 1
    assert response_data['prediction'][0] in ['Browser', 'Robot']

def test_predict_endpoint_invalid_input(client):
    input_data = {
        'session_id': 'be73c8d1b836170a21529a1b23140f8e',
        'epoch_ms': 1520280000000,
        'country_by_ip_address': 'US',
        'region_by_ip_address': 'CA',
        'url_without_parameters': 'https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/'
        # Missing 'visitor_recognition_type'
    }
    rv = client.post('/predict', json=input_data)
    
    assert rv.status_code == 400
    response_data = json.loads(rv.data)
    assert 'error' in response_data
    assert 'visitor_recognition_type' in response_data['error']

def test_predict_endpoint_unexpected_error(client, monkeypatch):
    def mock_preprocess_input(data):
        raise Exception("Mock unexpected error")
    
    monkeypatch.setattr('deploy.preprocess_input', mock_preprocess_input)
    
    input_data = {
        'session_id': 'be73c8d1b836170a21529a1b23140f8e',
        'epoch_ms': 1520280000000,
        'country_by_ip_address': 'US',
        'region_by_ip_address': 'CA',
        'url_without_parameters': 'https://www.bol.com/nl/l/nieuwe-actie-avontuur-over-prive-detective/N/33590+26931+7289/',
        'referrer_without_parameters': '',
        'visitor_recognition_type': 'ANONYMOUS'
    }
    rv = client.post('/predict', json=input_data)
    
    assert rv.status_code == 500
    response_data = json.loads(rv.data)
    assert 'error' in response_data
    assert response_data['error'] == 'Unexpected error occurred'
