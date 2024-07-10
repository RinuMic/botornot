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
    df['url_length'] = df['url_without_parameters'].apply(lambda url: len(url))
    df['url_type'] = df['url_without_parameters'].apply(
        lambda url: 'product' if '/p/' in url else ('category' if '/l/' in url else 'other')
    )
    df['referrer_present'] = df['referrer_without_parameters'].apply(
        lambda ref: 0 if pd.isnull(ref) or ref == '' else 1
    )
    df['visitor_recognition_type_encoded'] = df['visitor_recognition_type'].map(
        {'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3}
    )
    X = pd.get_dummies(
        data=df[['country_by_ip_address', 'region_by_ip_address', 'url_length', 'url_type',
                 'referrer_present', 'visitor_recognition_type_encoded']],
        drop_first=True
    )
    y = df['ua_agent_class']
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    return X, y, le_target

def train_model(X_train, y_train):
    """
    Trains the model using GridSearchCV.
    
    Args:
        X_train (np.ndarray): The training feature matrix.
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
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, le_target, scaler, X_columns):
    """
    Evaluates the trained model and prints the evaluation metrics.
    Saves the model, scaler, target encoder, and column names to a joblib file.
    
    Args:
        model (XGBClassifier): The trained model.
        X_test (np.ndarray): The test feature matrix.
        y_test (np.ndarray): The test target vector.
        le_target (LabelEncoder): The LabelEncoder fitted on the target variable.
        scaler (StandardScaler): The fitted StandardScaler instance.
        X_columns (list): List of column names used in the feature matrix X.
    """
    y_pred = model.predict(X_test)
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

    joblib.dump((model, scaler, le_target, X_columns), '../models/best_model.pkl')

if __name__ == "__main__":
    df = pd.read_csv('../data/clickdata.csv')
    X, y, le_target = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    best_model = train_model(X_train, y_train)
    evaluate_model(best_model, X_test, y_test, le_target, scaler, X.columns.tolist())
