# Importing necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib

def initial_exploration(df):
    """
    Perform initial exploration of the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    """
    print(df.info())
    print('------------------------------------------------------------')
    print(df.describe())
    print('------------------------------------------------------------')
    print(df.head())
    print(df.groupby(['ua_agent_class', 'visitor_recognition_type']).count())

    # Printing unique values
    unique_values = {}
    for column in df.columns:
        unique_values[column] = df[column].unique()

    for column, values in unique_values.items():
        print(f"Unique values for {column}:")
        print(values)
        print()

def data_cleaning(df):
    """
    Perform data cleaning on the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.replace(np.nan, '', regex=True)
    df = df.replace('Unknown', '', regex=True)

    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Browser Webview', 'Browser')
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Robot Mobile', 'Robot')
    print(df['ua_agent_class'].unique())

    df['url_length'] = df['url_without_parameters'].apply(lambda url: len(url))
    df['url_type'] = df['url_without_parameters'].apply(lambda url: 'product' if '/p/' in url else ('category' if '/l/' in url else 'other'))
    df['referrer_present'] = df['referrer_without_parameters'].apply(lambda ref: 0 if pd.isnull(ref) or ref == '' else 1)
    df['visitor_recognition_type_encoded'] = df['visitor_recognition_type'].map({'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3})
    df['referrer_present'] = df['referrer_without_parameters'].apply(lambda ref: 0 if pd.isnull(ref) or ref == '' else 1)

    return df

def preprocess_data(df):
    """
    Preprocesses the data for modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        tuple: X (features), y (target), LabelEncoder for target.
    """
    X = pd.get_dummies(data=df[['country_by_ip_address', 'region_by_ip_address', 'url_length', 'url_type', 'referrer_present', 'visitor_recognition_type_encoded']], drop_first=True)
    y = df['ua_agent_class']
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    return X, y, le_target

def train_models(X_train, y_train):
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
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
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
        },
        'XGB': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    best_estimators = {}
    for clf_name in classifiers:
        print(f"Running Grid Search for {clf_name}...")
        grid_search = GridSearchCV(estimator=classifiers[clf_name], param_grid=param_grids[clf_name], cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_estimators[clf_name] = grid_search.best_estimator_
        print(f"Best Parameters for {clf_name}: {grid_search.best_params_}")

    return best_estimators

def evaluate_models(best_estimators, X_test, y_test):
    """
    Evaluates the performance of each best model on the test data.
    
    Args:
        best_estimators (dict): Dictionary containing the best estimator for each model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    for clf_name, clf in best_estimators.items():
        print(f"\nEvaluating {clf_name}...")
        y_pred = clf.predict(X_test)
        
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

def save_best_model(best_model, scaler, le_target, columns_list, file_name):
    """
    Saves the best model along with scaler, LabelEncoder, and feature columns to a file.
    
    Args:
        best_model: Best model object.
        scaler: StandardScaler object.
        le_target: LabelEncoder object.
        columns_list (list): List of feature columns.
        file_name (str): File path to save the model.
    """
    joblib.dump((best_model, scaler, le_target, columns_list), file_name)
    print(f"Best model saved as {file_name}")

def main():
    # Load the data
    df = pd.read_csv('../data/clickdata.csv')

    # Initial exploration
    initial_exploration(df)

    # Data cleaning
    df = data_cleaning(df)

    # Preprocess data
    X, y, le_target = preprocess_data(df)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    best_estimators = train_models(X_train, y_train)

    # Save the best model
    save_best_model(best_estimators['XGB'], scaler, le_target, X.columns.tolist(), '../models/best_model.pkl')

    # Evaluate models
    evaluate_models(best_estimators, X_test, y_test)

if __name__ == "__main__":
    main()
