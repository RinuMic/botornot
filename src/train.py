"""
Module: train.py

This script trains a machine learning model to classify user agents. The script includes
functions for preprocessing data, training the model using grid search, evaluating the model,
and saving the best model.

Dependencies:
- pandas
- numpy
- sklearn (train_test_split, GridSearchCV, StandardScaler, LabelEncoder)
- xgboost (XGBClassifier)
- joblib

Usage:
Run the script to train the model and save the best model:
    $ python train.py

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import joblib
from xgboost import XGBClassifier

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

def preprocess_data(df):
    """
    Preprocesses the input data for training.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data.
        
    Returns:
        tuple: Processed feature matrix (X), target vector (y), 
        and fitted LabelEncoder for the target.
    """
    df = df.replace(np.nan, '', regex=True)
    df = df.replace('Unknown', '', regex=True)
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Browser Webview', 'Browser')
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Robot Mobile', 'Robot')
    df['url_length'] = df['url_without_parameters'].apply(calculate_url_length)
    df['url_type'] = df['url_without_parameters'].apply(determine_url_type)
    df['referrer_present'] = df['referrer_without_parameters'].apply(check_referrer_presence)
    df['visitor_recognition_type_encoded'] = df['visitor_recognition_type'].apply(encode_recognition_type)
    x = pd.get_dummies(
        data=df[['country_by_ip_address', 'region_by_ip_address', 'url_length', 'url_type',
                 'referrer_present', 'visitor_recognition_type_encoded']],
        drop_first=True
    )
    y = df['ua_agent_class']
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    return x, y, le_target

def train_model(x_train, y_train):
    """
    Trains the model using GridSearchCV.
    
    Args:
        x_train (np.ndarray): The training feature matrix.
        y_train (np.ndarray): The training target vector.
        
    Returns:
        XGBClassifier: The best estimator found by GridSearchCV.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, x_test_data, y_test_data, le_target_model, scaler_val, x_columns):
    """
    Evaluates the trained model and prints the evaluation metrics.
    Saves the model, scaler, target encoder, and column names to a joblib file.
    
    Args:
        model (XGBClassifier): The trained model.
        X_test (np.ndarray): The test feature matrix.
        y_test (np.ndarray): The test target vector.
        le_target (LabelEncoder): The LabelEncoder fitted on the target variable.
        scaler (StandardScaler): The fitted StandardScaler instance.
        x_columns (list): List of column names used in the feature matrix X.
    """
    y_pred = model.predict(x_test_data)
    print("Confusion Matrix:\n", confusion_matrix(y_test_data, y_pred))
    print("Classification Report:\n", classification_report(y_test_data, y_pred))
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred, average='weighted')
    recall = recall_score(y_test_data, y_pred, average='weighted')
    f1 = f1_score(y_test_data, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    joblib.dump((model, scaler_val, le_target_model, x_columns), '../models/best_model.pkl')

if __name__ == "__main__":
    df = pd.read_csv('../data/clickdata.csv')
    x, y, le_target = preprocess_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    best_model = train_model(x_train, y_train)
    evaluate_model(best_model, x_test, y_test, le_target, scaler, x.columns.tolist())
