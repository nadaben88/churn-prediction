import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import patch, mock_open
from src.preprocess import (
    load_raw_data,
    clean_data,
    build_preprocessor,
    save_preprocessor,
    preprocess_data,
)
from src.utils import load_config, map_yes_no, map_gender

# --- Fixtures ---
@pytest.fixture
def mock_config():
    """Mock config for testing."""
    return {
        "data": {
            "raw": "data/raw/telco_churn.csv",
            "processed": "data/processed/processed.csv",
        },
        "preprocessing": {
            "target_column": "Churn",
            "binary_categorical_columns": [
                "Partner",
                "Dependents",
                "PhoneService",
                "PaperlessBilling",
            ],
            "nominal_categorical_columns": [
                "SeniorCitizen",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaymentMethod",
            ],
            "numerical_columns": ["tenure", "MonthlyCharges", "TotalCharges"],
            "drop_columns": ["customerID"],
        },
        "artifacts": {
            "model_dir": "artifacts/",
            "preprocessor_name": "preprocessor.joblib",
        },
        "logging": {
            "log_file": "logs/app.log",
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
    }

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "tenure": [1, 2, 3],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No"],  # <-- Add missing columns
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["No", "Yes", "No"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "Yes", "No"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Credit card", "Electronic check"],
        "MonthlyCharges": [50.0, 60.0, 70.0],
        "TotalCharges": [50.0, 120.0, 140.0],
        "Churn": ["Yes", "No", "Yes"],
    })

# --- Tests ---
def test_load_raw_data(mock_config, mocker):
    """Test if raw data is loaded correctly."""
    mock_data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    mocker.patch("pandas.read_csv", return_value=mock_data)
    df = load_raw_data(mock_config)
    assert df.equals(mock_data), "Data not loaded correctly"

def test_clean_data(sample_data, mock_config):
    """Test data cleaning: missing values, target encoding, and TotalCharges calculation."""
    df = clean_data(sample_data.copy(), mock_config)
    # Check TotalCharges imputation
    assert not df["TotalCharges"].isna().any(), "Missing values in TotalCharges not handled"
    # Check target encoding
    assert set(df["Churn"].unique()) == {0, 1}, "Target column not encoded correctly"
    # Check dropped columns
    assert "customerID" not in df.columns, "customerID not dropped"

def test_build_preprocessor(mock_config):
    """Test if the preprocessor is built correctly."""
    preprocessor = build_preprocessor(mock_config)
    assert preprocessor is not None, "Preprocessor not built"
    assert len(preprocessor.transformers) == 4, "Incorrect number of transformers"

def test_save_preprocessor(mock_config, mocker):
    """Test if the preprocessor is saved correctly."""
    mock_preprocessor = "dummy_preprocessor"
    mocker.patch("joblib.dump")
    mocker.patch("pathlib.Path.mkdir")
    save_preprocessor(mock_preprocessor, mock_config)
    joblib.dump.assert_called_once()

def test_preprocess_data(mock_config, mocker, sample_data):
    """Test the main preprocessing pipeline."""
    mocker.patch("pandas.read_csv", return_value=sample_data)
    mocker.patch("joblib.dump")
    mocker.patch("pathlib.Path.mkdir")
    mocker.patch("pandas.DataFrame.to_csv")
    preprocess_data(mock_config)
    # Assertions
    assert pd.read_csv.called, "Data not loaded"
    assert joblib.dump.called, "Preprocessor or feature names not saved"
    assert pd.DataFrame.to_csv.called, "Processed data not saved"

def test_map_yes_no():
    """Test the map_yes_no utility function."""
    df = pd.DataFrame({"col": ["Yes", "No", "Yes"]})
    result = map_yes_no(df)
    assert result.equals(pd.DataFrame({"col": [1, 0, 1]})), "Yes/No mapping failed"

def test_map_gender():
    """Test the map_gender utility function."""
    df = pd.DataFrame({"gender": ["Male", "Female", "Male"]})
    result = map_gender(df)
    assert result.equals(pd.DataFrame({"gender": [1, 0, 1]})), "Gender mapping failed"
