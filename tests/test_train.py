"""
Tests for training module using pytest.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from train import (
    preprocess_data, train_model, evaluate_model, 
    determine_url_type, encode_recognition_type
)

# Sample data for testing
@pytest.fixture
def sample_data():
    """
    Fixture for creating sample data for testing.
    """
    data = {
        'country_by_ip_address': ['US', 'CA', 'US', 'CA'],
        'region_by_ip_address': ['CA', 'CA', 'NY', 'NY'],
        'url_without_parameters': ['/p/product1', '/l/category1', '/p/product2', '/otherpage'],
        'referrer_without_parameters': [None, '/l/category2', '/p/product3', '/l/category3'],
        'visitor_recognition_type': ['', 'ANONYMOUS', 'LOGGEDIN', 'RECOGNIZED'],
        'ua_agent_class': ['Browser', 'Browser', 'Robot', 'Browser']
    }
    return pd.DataFrame(data)

# Test preprocess_data function
def test_preprocess_data(sample_data):
    """
    Test the preprocess_data function.
    """
    X_data, y_data, le_target = preprocess_data(sample_data)
    
    # Check X and y shapes
    assert X_data.shape[0] == sample_data.shape[0]
    assert len(y_data) == sample_data.shape[0]
    
    # Check LabelEncoder
    assert isinstance(le_target, LabelEncoder)
    
    # Check column names in X after preprocessing
    expected_columns = [
        'url_length', 'referrer_present', 'visitor_recognition_type_encoded',
        'country_by_ip_address_CA', 'country_by_ip_address_US',
        'region_by_ip_address_NY', 'url_type_other', 'url_type_product',
        'referrer_without_parameters_/l/category2', 
        'referrer_without_parameters_/p/product3', 'ua_agent_class'
    ]
    assert all(col in X_data.columns for col in expected_columns)

# Test determine_url_type function
def test_determine_url_type():
    """
    Test the determine_url_type function.
    """
    assert determine_url_type('/p/product1') == 'product'
    assert determine_url_type('/l/category1') == 'category'
    assert determine_url_type('/otherpage') == 'other'

# Test encode_recognition_type function
def test_encode_recognition_type():
    """
    Test the encode_recognition_type function.
    """
    assert encode_recognition_type('') == 0
    assert encode_recognition_type('ANONYMOUS') == 1
    assert encode_recognition_type('LOGGEDIN') == 2
    assert encode_recognition_type('RECOGNIZED') == 3
    assert encode_recognition_type('UNKNOWN') == -1

# Sample data for model training and evaluation
@pytest.fixture
def sample_train_test_data():
    """
    Fixture for creating sample train and test data for model training and evaluation.
    """
    np.random.seed(0)
    X_data = np.random.rand(100, 5)
    y_data = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Test train_model function
def test_train_model(sample_train_test_data):
    """
    Test the train_model function.
    """
    X_train, X_test, y_train, y_test = sample_train_test_data
    model = train_model(X_train, y_train)
    assert isinstance(model, XGBClassifier)

# Test evaluate_model function
def test_evaluate_model(sample_train_test_data):
    """
    Test the evaluate_model function.
    """
    X_train, X_test, y_train, y_test = sample_train_test_data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    le_target = LabelEncoder()
    le_target.fit(y_train)
    evaluate_model(model, X_test, y_test, le_target)
