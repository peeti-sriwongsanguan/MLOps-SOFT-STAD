import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])


def preprocess_data(df):
    # Convert IsHoliday to numeric
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # Create time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Handle missing values
    df = df.fillna(method='ffill')

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


def prepare_data_for_model(df):
    target = 'Weekly_Sales'
    features = [col for col in df.columns if col not in ['Date', 'Weekly_Sales']]

    X = df[features]
    y = df[target]

    return X, y


if __name__ == "__main__":
    df = load_data("data/walmart_cleaned.csv")
    preprocessed_df = preprocess_data(df)
    X, y = prepare_data_for_model(preprocessed_df)
    print("Data preprocessing completed.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")