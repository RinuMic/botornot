"""
Module: train.py

This script trains a machine learning model to classify user agents. The script includes
functions for preprocessing data, training the model using grid search, evaluating the model,
and saving the best model.

Dependencies:
- pandas
- numpy
- sklearn (train_test_split, GridSearchCV, StandardScaler, LabelEncoder)
- RandomForestClassifier, GradientBoostingClassifier
- joblib

Usage:
Run the script to train the model and save the best model:
    $ python train.py

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
from utils import encode_recognition_type, calculate_url_length, check_referrer_presence, determine_url_type

def data_exploration(df):
    # Basic Statistics
    print("Basic Statistics:")
    print(df.describe(include='all'))
    # Data Types and Memory Usage
    print("\nData Types:")
    print(df.dtypes)
    print("\nMemory Usage:")
    print(df.memory_usage(deep=True))
    # Missing Values Analysis
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    print("\nPercentage of Missing Values:")
    missing_percentage = (df.isnull().mean() * 100).round(2)
    print(missing_percentage[missing_percentage > 0])
    # Visualizing Missing Values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()
    # Value Counts for Categorical Features
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        print(f"\nValue Counts for {column}:")
        print(df[column].value_counts())
    # Unique Values Check
    unique_values = {}
    for column in df.columns:
        unique_values[column] = df[column].nunique()
    print("\nNumber of Unique Values per Column:")
    print(unique_values)

def preprocess_data(df, le_region=None, le_country=None):
    """
    Preprocesses the input data for training.
    Args:
        df (pd.DataFrame): The input DataFrame containing the raw data.
        le_region (LabelEncoder, optional): LabelEncoder for region_by_ip_address. Defaults to None.
        le_country (LabelEncoder, optional): LabelEncoder for country_by_ip_address. Defaults to None.

    Returns:
        tuple: Processed feature matrix (X), target vector (y),
        and fitted LabelEncoders for categorical variables.
    """
    # 1. Data Exploration
    data_exploration(df)

    # 2. Handling Missing and Unknown Values
    # Replace NaN and 'Unknown' with empty strings
    df = df.replace(np.nan, '', regex=True)
    df = df.replace('Unknown', '', regex=True)

    # 3. Data Cleaning
    # Perform replacements for ua_agent_class
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Browser Webview', 'Browser')
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Robot Mobile', 'Robot')

    # 4. Encoding Categorical Features
    # Apply LabelEncoder for region and country if not provided
    if le_region is None:
        le_region = LabelEncoder()
        df['region_numeric'] = le_region.fit_transform(df['region_by_ip_address'])
    else:
        df['region_numeric'] = le_region.transform(df['region_by_ip_address'])

    if le_country is None:
        le_country = LabelEncoder()
        df['country_numeric'] = le_country.fit_transform(df['country_by_ip_address'])
    else:
        df['country_numeric'] = le_country.transform(df['country_by_ip_address'])
    # Print mappings for region and country encodings
    print("Region Classes:", le_region.classes_)
    print("Country Classes:", le_country.classes_)
    print("\nTransformed DataFrame:")
    print(df)

    # 5. Grouping Target Variable Classes
    # Group classes into NHT and HT
    class_mapping = {
        'Browser': 'HT',
        'Robot': 'NHT',
        'Hacker': 'NHT',
        'Special': 'NHT',
        'Mobile App': 'NHT',
        'Cloud Application': 'NHT'
    }
    # Apply the class mapping
    df['ua_agent_class_grouped'] = df['ua_agent_class'].map(class_mapping)
    # Check the unique values in the 'ua_agent_class_grouped' column
    print("Unique grouped classes:")
    print(df['ua_agent_class_grouped'].unique())
    # 6. Class Distribution Analysis
    # Compute class frequencies
    class_counts = df['ua_agent_class_grouped'].value_counts()
    total_samples = len(df)
    class_frequencies = class_counts / total_samples
    print("\nClass distribution:")
    print(class_counts)
    print("\nClass frequencies:")
    print(class_frequencies)
    # 7. Feature Engineering
    # Determine other features
    df['url_type'] = df['url_without_parameters'].apply(determine_url_type)
    df['url_length'] = df['url_without_parameters'].apply(calculate_url_length)
    df['referrer_present'] = df['referrer_without_parameters'].apply(check_referrer_presence)
    df['visitor_recognition_type_encoded'] = df['visitor_recognition_type'].apply(encode_recognition_type)
    # Convert categorical columns to numeric if needed
    df['url_type'] = df['url_type'].astype('category').cat.codes
    # Select relevant features for training
    feature_columns = ['country_numeric', 'region_numeric', 'url_length', 'url_type', 
                       'referrer_present', 'visitor_recognition_type_encoded']
    # Check for missing columns
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Columns {missing_columns} not found in DataFrame.")
    # Ensure all selected features are numeric
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    # Drop rows with NaN values (if any)
    df.dropna(inplace=True)
    # Feature and Target Separation
    x = df[feature_columns]
    # 8. Encoding Target Variable
    y = df['ua_agent_class_grouped']
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Return processed data and encoders
    return x, y, le_target, le_region, le_country


def train_model(X_train, y_train):
    """
    Trains multiple models and performs grid search for hyperparameter tuning.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target. 
    Returns:
        dict: Dictionary containing the best estimator for each model.
    """
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    best_estimators = {}
    for clf_name in classifiers:
        print(f"Running Grid Search for {clf_name}...")
        # Define the undersampler
        undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        # Resample the training data
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=classifiers[clf_name], param_grid=param_grids[clf_name], cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_resampled, y_train_resampled)
        # Store the best estimator
        best_estimators[clf_name] = grid_search.best_estimator_
        print(f"Best Parameters for {clf_name}: {grid_search.best_params_}")

    return best_estimators[clf_name]

def evaluate_model(model, x_test_data, y_test_data, le_target_model, le_region_model, le_country_model, scaler_val, x_columns):
    """
    Evaluates the trained model and prints the evaluation metrics.
    Saves the model, scaler, target encoder, and column names to a joblib file.
    Args:
        model (RandomForestClassifier): The trained model.
        x_test_data (np.ndarray): The test feature matrix.
        y_test_data (np.ndarray): The test target vector.
        le_target_model (LabelEncoder): The LabelEncoder fitted on the target variable.
        le_region_model (LabelEncoder): The LabelEncoder fitted on the region variable.
        le_country_model (LabelEncoder): The LabelEncoder fitted on the country variable.
        scaler_val (StandardScaler): The fitted StandardScaler instance.
        x_columns (list): List of column names used in the feature matrix X.
    """
    # Step 1: Prediction and Evaluation Metrics
    y_pred = model.predict(x_test_data)
    print("Confusion Matrix:\n", confusion_matrix(y_test_data, y_pred))
    print("Classification Report:\n", classification_report(y_test_data, y_pred))
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred, average='weighted')
    recall = recall_score(y_test_data, y_pred, average='weighted')
    f1 = f1_score(y_test_data, y_pred, average='weighted')
    '''Accuracy - proportion of correct predictions (both true positives and true negatives) among all predictions made'''
    print(f"Accuracy: {accuracy:.4f}")
    '''PRECISION -proportion of predicted non-human traffic (positive predictions for NHT) that is actually non-human'''
    print(f"Precision: {precision:.4f}")
    '''Recall - proportion of actual non-human traffic that is correctly identified by the model'''
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # Step 2: Saving Model and Associated Objects
    joblib.dump((model, scaler_val, le_target_model, le_region_model, le_country_model, x_columns), '../models/best_model.pkl')
    # Step 3: Feature Importance Analysis
    importances = model.feature_importances_
    feature_names = x_columns  # Assuming x_columns is correctly passed
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. Feature '{feature_names[indices[f]]}' ({importances[indices[f]]:.4f})")
    # Step 4: Cross-validation Scores
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')  # Assuming x_train and y_train are defined
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean()}")

if __name__ == "__main__":
    df = pd.read_csv('../data/clickdata.csv')
    # Preprocess the data
    x, y, le_target, le_region, le_country = preprocess_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    best_model = train_model(x_train, y_train)
    evaluate_model(best_model, x_test, y_test, le_target,le_region, le_country, scaler, x.columns.tolist())
    