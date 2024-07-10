# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_pipeline(df):
    # Load saved preprocessing objects
    scaler, le_target, columns = joblib.load('models/preprocessing_objects.pkl')
    
    # Apply preprocessing on the data
    df = df.replace(np.nan, '', regex=True)
    df = df.replace('Unknown', '', regex=True)
    
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Browser Webview','Browser')
    df['ua_agent_class'] = df['ua_agent_class'].str.replace('Robot Mobile','Robot')
    
    df['url_length'] = df['url_without_parameters'].apply(lambda url: len(url))
    df['url_type'] = df['url_without_parameters'].apply(lambda url: 'product' if '/p/' in url else ('category' if '/l/' in url else 'other'))
    df['referrer_present'] = df['referrer_without_parameters'].apply(lambda ref: 0 if pd.isnull(ref) or ref == '' else 1)
    df['visitor_recognition_type_encoded'] = df['visitor_recognition_type'].map({'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3})
    df['referrer_present'] = df['referrer_without_parameters'].apply(lambda ref: 0 if pd.isnull(ref) or ref == '' else 1)
    
    X = pd.get_dummies(data=df[['country_by_ip_address', 'region_by_ip_address','url_length', 'url_type', 'referrer_present', 'visitor_recognition_type_encoded']], drop_first=True)
    y = df['ua_agent_class']
    y = le_target.transform(y)
    
    X = scaler.transform(X)
    
    # Load the best model
    best_model = joblib.load('models/best_model.pkl')
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training on the best model
    best_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = best_model.predict(X_test)
    
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

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('data/clickdata.csv')
    train_pipeline(df)
