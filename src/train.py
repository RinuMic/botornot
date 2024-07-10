"""
Module: train.py

This script contains functions for preprocessing data, training an XGBoost classifier
using hyperparameter tuning with GridSearchCV, and evaluating the model.

Functions:
- preprocess_data: Preprocesses input data, encoding categorical variables and splitting
  into training and testing sets.
- train_model: Trains an XGBoost classifier using GridSearchCV for hyperparameter tuning.
- evaluate_model: Evaluates the trained model using various metrics like accuracy, precision,
  recall, and F1-score.

Dependencies:
- pandas
- numpy
- sklearn (train_test_split, GridSearchCV, StandardScaler, LabelEncoder)
- xgboost (XGBClassifier)
- sklearn.metrics (classification_report, confusion_matrix, accuracy_score, precision_score,
  recall_score, f1_score)
- joblib

Usage:
To train a model:
    $ python train.py

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

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
    elif '/l/' in url:
        return 'category'
    else:
        return 'other'

def encode_recognition_type(type):
    """
    Encodes the visitor recognition type into numerical values.
    Args:
        type (str): The visitor recognition type.
    Returns:
        int: Encoded value corresponding to the type, or -1 if type is not found in encoding_map.
    """
    encoding_map = {'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3}
    return encoding_map.get(type, -1)  # Handle unexpected values gracefully

# Assuming new_data['url_without_parameters'] contains URLs as strings
def calculate_url_length(url):
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
    else:
        return 1

def preprocess_data(input_df):
    """
    Preprocesses the input DataFrame `input_df`.
    Performs data cleaning, feature engineering, and encoding.
    Args:
        input_df (pd.DataFrame): Input DataFrame containing raw data.  
    Returns:
        pd.DataFrame: Processed features (x) after transformations and one-hot encoding.
        np.ndarray: Encoded target variable (y_data).
        LabelEncoder: LabelEncoder object fit on the target variable.
    """
    # Apply preprocessing on the data
    input_df = input_df.replace(np.nan, '', regex=True)
    input_df = input_df.replace('Unknown', '', regex=True)
    # Reduce the amount of detail in classes by merging all different Human types
    input_df['ua_agent_class'] = input_df['ua_agent_class'].str.replace('Browser Webview', 'Browser')
    input_df['ua_agent_class'] = input_df['ua_agent_class'].str.replace('Robot Mobile', 'Robot')
    # Feature engineering 
    input_df['url_length'] = input_df['url_without_parameters'].apply(calculate_url_length) 
    input_df['url_type'] = input_df['url_without_parameters'].apply(determine_url_type)
    input_df['referrer_present'] = input_df['referrer_without_parameters'].apply(check_referrer_presence)
    input_df['visitor_recognition_type_encoded'] = input_df['visitor_recognition_type'].apply(encode_recognition_type)
    # One-hot encoding for specified features
    x = pd.get_dummies(
        data=input_df[['country_by_ip_address', 'region_by_ip_address', 
                 'url_length', 'url_type', 'referrer_present', 
                 'visitor_recognition_type_encoded']],
        drop_first=True
    )
    # Separate target variable
    y_data = input_df['ua_agent_class']
    # Encode target variable
    le_target = LabelEncoder()
    y_data = le_target.fit_transform(y_data)
    return x, y_data, le_target

def train_model(X_train, y_train):
    """
    Trains an XGBoost classifier using Grid Search for hyperparameter tuning.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training target labels.  
    Returns:
        best_model (XGBClassifier): Best trained XGBoost model.
    """
    # Define the model and hyperparameter grid
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    # Perform Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, le_target):
    """
    Evaluates the trained model on the test data and prints evaluation metrics.
    Args:
        model (XGBClassifier): Trained XGBoost model.
        X_test (pd.DataFrame): Test features.
        y_test (np.ndarray): Test target labels.
        le_target (LabelEncoder): LabelEncoder object used to encode target labels.
    """
    y_pred = model.predict(X_test)
    # Generate evaluation metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # Save the best model
    joblib.dump((model, scaler, le_target, x.columns.tolist()), '../models/best_model.pkl')

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('../data/clickdata.csv')
    # Preprocess the data
    x, y_data, le_target = preprocess_data(df)
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(x, y_data, test_size=0.2, random_state=42)
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Train the model
    best_model = train_model(X_train, y_train)
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test, le_target)
