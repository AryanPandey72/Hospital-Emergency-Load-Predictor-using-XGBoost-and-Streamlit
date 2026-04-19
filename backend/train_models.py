import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def add_time_features(df):
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df

def train():
    print("Loading dataset.csv...")
    try:
        df = pd.read_csv('dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Error loading dataset: {e}. Ensure data_engineering.py completed.")
        return

    df = add_time_features(df)
    
    # Target 1: Patient Volume
    features = ['Hour', 'DayOfWeek', 'Month', 'Temperature', 'Flu_Trend']
    X = df[features]
    y_volume = df['Patient_Volume']
    y_wait = df['Wait_Time_Mins']
    
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_volume, test_size=0.2, shuffle=False)
    
    print("Training Patient Inflow Model...")
    model_volume = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    model_volume.fit(X_train_v, y_train_v)
    
    pred_v = model_volume.predict(X_test_v)
    print(f"Patient Volume MAE: {mean_absolute_error(y_test_v, pred_v):.2f}")
    
    # Target 2: Wait Time Model
    X_wait = X.copy()
    X_wait['Current_Volume'] = y_volume
    
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wait, y_wait, test_size=0.2, shuffle=False)
    
    print("Training Wait Time Model...")
    model_wait = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    model_wait.fit(X_train_w, y_train_w)
    
    pred_w = model_wait.predict(X_test_w)
    print(f"Wait Time MAE: {mean_absolute_error(y_test_w, pred_w):.2f}")
    
    # Save models
    with open('model_volume.pkl', 'wb') as f:
        pickle.dump(model_volume, f)
    with open('model_wait.pkl', 'wb') as f:
        pickle.dump(model_wait, f)
        
    print("Models saved successfully (`model_volume.pkl`, `model_wait.pkl`)!")

if __name__ == "__main__":
    train()
