import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, preprocess_data, prepare_data_for_model


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Date': pd.date_range(start='2010-01-01', periods=10, freq='W'),
        'Store': [1] * 10,
        'Dept': [1] * 10,
        'Weekly_Sales': np.random.rand(10) * 1000,
        'IsHoliday': [False] * 10,
        'Temperature': np.random.rand(10) * 30,
        'Fuel_Price': np.random.rand(10) * 5,
        'CPI': np.random.rand(10) * 200,
        'Unemployment': np.random.rand(10) * 10,
    })


def test_load_data(tmp_path):
    # Create a temporary CSV file
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    # Test loading the data
    loaded_df = load_data(file_path)
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (3, 2)


def test_preprocess_data(sample_data):
    preprocessed_df = preprocess_data(sample_data)

    assert 'IsHoliday' in preprocessed_df.columns
    assert preprocessed_df['IsHoliday'].dtype == int
    assert 'Year' in preprocessed_df.columns
    assert 'Month' in preprocessed_df.columns
    assert 'Day' in preprocessed_df.columns
    assert 'DayOfWeek' in preprocessed_df.columns


def test_prepare_data_for_model(sample_data):
    preprocessed_df = preprocess_data(sample_data)
    X, y = prepare_data_for_model(preprocessed_df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'Weekly_Sales' not in X.columns
    assert y.name == 'Weekly_Sales'
    assert X.shape[0] == y.shape[0]