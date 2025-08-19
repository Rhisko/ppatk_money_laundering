import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    print(df.isnull().sum())
    print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
    df = df.dropna()
    print(f"After dropping NaN values: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Feature Engineering: Extract Time
    df['Date'] = pd.to_datetime(df['Date'])
    df['weekday'] = df['Date'].dt.weekday
    df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    
    print("After feature engineering:")
    print(df.head())
    
    # Drop unnecessary columns
    df = df.drop(['Time', 'Date'], axis=1)
    print(f"After dropping 'Time' and 'Date': {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    
    # Log transformation for 'Amount'
    df['log_amount'] = np.log1p(df['Amount'])
    df = df.drop(['Amount'], axis=1)
    print("After log transformation of 'Amount':")
    print(df.head())

    categorical_cols = [
        'Sender_account', 'Receiver_account',
        'Payment_currency', 'Received_currency',
        'Sender_bank_location', 'Receiver_bank_location',
        'Payment_type'
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    print("After encoding categorical variables:")
    print(df.head())
    
    # Feature selection
    X = df[
        [
            'Sender_account', 'Receiver_account',
            'Payment_currency', 'Received_currency',
            'Sender_bank_location', 'Receiver_bank_location',
            'Payment_type', 'weekday', 'hour', 'log_amount'
        ]
    ]
    y = df['Is_laundering'].astype(int)

    return X, y  

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Print count of data in X_train and y_train
    print(f"X_train count: {X_train.shape[0]}")
    print(f"y_train count: {y_train.shape[0]}")
    # Scaling: fit scaler di train, transform to train & test
    # scaler = StandardScaler()
    # X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    # X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # (Opsional) Save scaler 

    # joblib.dump(scaler, "trained_models/scaler.pkl")

    # # (Opsional) Save preprocess result & split
    # X_train_scaled.to_csv('data/processed/X_train_scaled.csv', index=False)
    # X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test

