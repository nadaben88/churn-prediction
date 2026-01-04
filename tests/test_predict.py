import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.predict import (
    init_artifacts,
    predict,
    _normalize_input_to_df,
    _format_results,
    GLOBAL_CONFIG,
    GLOBAL_MODEL,
    GLOBAL_PREPROCESSOR,
    load_metadata_from_disk,
    load_preprocessor_from_disk
)
from src.utils import load_config ,load_model

# --- Fixtures ---
@pytest.fixture
def mock_config():
    """Mock config for testing."""
    return {
        "data": {
            "features": [
                "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
                "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                "MonthlyCharges", "TotalCharges",
            ],
            "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
        },
        "artifacts": {
            "model_dir": "artifacts/",
            "model_name": "model.joblib",
            "preprocessor_name": "preprocessor.joblib",
            "metadata_name": "metadata.json",
        },
        "serving": {
            "threshold": 0.5,
        },
        "model_version": "1.0",
    }

@pytest.fixture
def sample_input():
    """Sample input data for testing."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 53.85,
        "TotalCharges": 53.85,
    }

@pytest.fixture
def sample_batch_input(sample_input):
    """Sample batch input data for testing."""
    return [sample_input, {**sample_input, "tenure": 10, "MonthlyCharges": 99.99}]

@pytest.fixture
def mock_model_single():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 1 sample
    model.predict.return_value = np.array([1])
    return model

@pytest.fixture
def mock_model_batch():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])  # 2 samples
    model.predict.return_value = np.array([1, 0])
    return model

@pytest.fixture
def mock_model(mock_model_single):
    return mock_model_single



@pytest.fixture
def mock_preprocessor():
    """Mock preprocessor for testing."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[1, 2, 3], [4, 5, 6]])
    return preprocessor

# --- Tests ---
def test_normalize_input_to_df(mock_config, sample_input):
    """Test input normalization to DataFrame."""
    df = _normalize_input_to_df(sample_input, mock_config)
    assert isinstance(df, pd.DataFrame), "Output is not a DataFrame"
    assert list(df.columns) == mock_config["data"]["features"], "Columns do not match expected features"
    assert df.shape[0] == 1, "DataFrame should have 1 row"

def test_normalize_input_to_df_batch(mock_config, sample_batch_input):
    """Test batch input normalization to DataFrame."""
    df = _normalize_input_to_df(sample_batch_input, mock_config)
    assert isinstance(df, pd.DataFrame), "Output is not a DataFrame"
    assert df.shape[0] == 2, "DataFrame should have 2 rows"

def test_format_results(mock_config):
    """Test result formatting."""
    df_inputs = pd.DataFrame([{"tenure": 1, "MonthlyCharges": 50}])
    probs = np.array([0.8])
    labels = np.array([1])
    results = _format_results(df_inputs, probs, labels, mock_config)
    assert "pred_prob" in results.columns, "Missing pred_prob column"
    assert "pred_label" in results.columns, "Missing pred_label column"
    assert "churn_risk" in results.columns, "Missing churn_risk column"
    assert results["churn_risk"].iloc[0] == "High", "churn_risk not computed correctly"

def test_predict_single(mock_config, sample_input, mock_model, mock_preprocessor):
    """Test single prediction."""
    with patch("src.predict.GLOBAL_MODEL", mock_model), \
         patch("src.predict.GLOBAL_PREPROCESSOR", mock_preprocessor), \
         patch("src.predict.GLOBAL_CONFIG", mock_config):
        result = predict(sample_input)
        assert isinstance(result, dict), "Single prediction should return a dict"
        assert result["pred_label"] in {0, 1}, "pred_label should be 0 or 1"
        assert 0 <= result["pred_prob"] <= 1, "pred_prob should be between 0 and 1"

def test_predict_batch(mock_config, sample_batch_input, mock_model, mock_preprocessor):
    """Test batch prediction."""
    with patch("src.predict.GLOBAL_MODEL", mock_model), \
         patch("src.predict.GLOBAL_PREPROCESSOR", mock_preprocessor), \
         patch("src.predict.GLOBAL_CONFIG", mock_config):
        result = predict(sample_batch_input)
        assert isinstance(result, list), "Batch prediction should return a list"
        assert len(result) == 2, "Batch prediction should return 2 results"
        assert all(isinstance(r, dict) for r in result), "Each result should be a dict"

def test_predict_return_df(mock_config, sample_batch_input, mock_model, mock_preprocessor):
    """Test prediction with return_df=True."""
    with patch("src.predict.GLOBAL_MODEL", mock_model), \
         patch("src.predict.GLOBAL_PREPROCESSOR", mock_preprocessor), \
         patch("src.predict.GLOBAL_CONFIG", mock_config):
        result = predict(sample_batch_input, return_df=True)
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        assert "pred_prob" in result.columns, "Missing pred_prob column"
        assert "pred_label" in result.columns, "Missing pred_label column"

def test_predict_missing_features(mock_config, mock_model, mock_preprocessor):
    """Test prediction with missing features."""
    incomplete_input = {
        "gender": "Male",
        "tenure": 2,
        "MonthlyCharges": 53.85,
    }
    with patch("src.predict.GLOBAL_MODEL", mock_model), \
         patch("src.predict.GLOBAL_PREPROCESSOR", mock_preprocessor), \
         patch("src.predict.GLOBAL_CONFIG", mock_config):
        result = predict(incomplete_input)
        assert isinstance(result, dict), "Result should be a dict"
        assert "pred_label" in result, "Missing pred_label in result"

def init_artifacts(config_path):
    global GLOBAL_CONFIG, GLOBAL_MODEL, GLOBAL_PREPROCESSOR, GLOBAL_METADATA

    cfg = load_config(config_path)
    GLOBAL_CONFIG = cfg

    GLOBAL_PREPROCESSOR = load_preprocessor_from_disk(
        cfg["artifacts"]["model_dir"], cfg["artifacts"]["preprocessor_name"]
    )

    GLOBAL_MODEL = load_model(
        cfg["artifacts"]["model_dir"], cfg["artifacts"]["model_name"]
    )

    GLOBAL_METADATA = load_metadata_from_disk(
        cfg["artifacts"]["model_dir"], cfg["artifacts"]["metadata_name"]
    )



