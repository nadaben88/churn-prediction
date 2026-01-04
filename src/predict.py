import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
import joblib
import numpy as np
import pandas as pd
import yaml
from .utils import load_config , setup_logging ,load_model


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Module-level singletons (populated by init_artifacts)
GLOBAL_CONFIG: Optional[Dict] = None
GLOBAL_MODEL = None
GLOBAL_PREPROCESSOR = None
GLOBAL_METADATA: Optional[Dict] = None



def load_preprocessor_from_disk(config: Dict):
    """Load the preprocessing artifact from disk (path taken from config)."""
    model_dir = Path(config["artifacts"]["model_dir"])
    preproc_path = model_dir / config["artifacts"]["preprocessor_name"]
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {preproc_path.resolve()}")
    logger.info(f"Loading preprocessor from {preproc_path}")
    preprocessor = joblib.load(preproc_path)
    return preprocessor


def load_metadata_from_disk(config: Dict) -> Optional[Dict]:
    """Load model metadata if present (optional)."""
    model_dir = Path(config["artifacts"]["model_dir"])
    metadata_name = config["artifacts"].get("metadata_name")
    if not metadata_name:
        return None
    metadata_path = model_dir / metadata_name
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found at {metadata_path}; continuing without metadata.")
        return None
    try:
        metadata = joblib.load(metadata_path) if metadata_path.suffix in {".joblib", ".pkl"} else None
        if metadata is None:
            # try yaml/json
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
    except Exception:
        try:
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as exc:
            logger.warning(f"Unable to parse metadata file {metadata_path}: {exc}")
            metadata = None
    return metadata


def init_artifacts(config_path: str = "config.yaml"):
    """
    Initialize module-level artifacts by loading config, model, preprocessor and metadata.
    This should be called once at API startup (e.g., FastAPI startup event).
    """
    global GLOBAL_CONFIG, GLOBAL_MODEL, GLOBAL_PREPROCESSOR, GLOBAL_METADATA

    cfg = load_config(config_path)
    GLOBAL_CONFIG = cfg

    GLOBAL_PREPROCESSOR = load_preprocessor_from_disk(cfg)
    GLOBAL_MODEL = load_model(cfg)
    GLOBAL_METADATA = load_metadata_from_disk(cfg)

    logger.info("Artifacts initialized: model, preprocessor, metadata (if present).")


def _normalize_input_to_df(
    input_data: Union[Dict, List[Dict], pd.DataFrame],
    config: Dict,
) -> pd.DataFrame:
    """
    Convert input payload (dict, list of dicts, or DataFrame) to a DataFrame
    and ensure columns match the expected feature list from config.
    Does not apply model preprocessor; just prepares raw dataframe.

    - Fills missing expected features with NaN
    - Coerces numeric columns to numeric dtype (safe coercion)
    - Reorders columns to expected order
    """
    # Convert to DataFrame
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        raise TypeError("input_data must be dict, list of dicts, or pandas.DataFrame")

    expected_features: List[str] = config["data"]["features"]
    numeric_features: List[str] = config["data"].get("numeric_features", [])

    # Fill missing expected features
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = np.nan

    # Coerce numeric features
    for numc in numeric_features:
        if numc in df.columns:
            df[numc] = pd.to_numeric(df[numc], errors="coerce")

    # Optionally warn about unexpected extra columns
    extra_cols = [c for c in df.columns if c not in expected_features]
    if extra_cols:
        logger.debug(f"Input contains extra columns not used by model: {extra_cols}")

    # Reorder to expected features
    df = df[expected_features].copy()

    return df


def _format_results(df_inputs: pd.DataFrame, probs: np.ndarray, labels: np.ndarray, config: Dict):
    """
    Build a DataFrame of results aligned with df_inputs index.
    """
    threshold = float(config.get("serving", {}).get("threshold", 0.5))
    results = pd.DataFrame(index=df_inputs.index)
    results["pred_prob"] = probs
    results["pred_label"] = (probs >= threshold).astype(int)
    # If user passes a model output label (labels param), prefer labels if provided and consistent
    if labels is not None:
        # ensure integer labels
        try:
            labels = np.asarray(labels).astype(int)
            results["pred_label_model"] = labels
        except Exception:
            # fallback: ignore labels if type mismatch
            results["pred_label_model"] = np.nan

    # churn_risk text label
    results["churn_risk"] = results["pred_label"].apply(lambda p: "High" if p == 1 else "Low")

    # include model version if available
    model_version = (config.get("artifacts", {}).get("model_name") or "")  

    results["model_version"] = config.get("model_version", model_version)
    return results


def predict(
    input_data: Union[Dict, List[Dict], pd.DataFrame],
    config: Optional[Dict] = None,
    model: Optional[object] = None,
    preprocessor: Optional[object] = None,
    return_df: bool = False,
) -> Union[Dict, List[Dict], pd.DataFrame]:
    """
    Run inference for single or batch inputs.

    Parameters
    ----------
    input_data : dict | list[dict] | pd.DataFrame
        Incoming raw features (not yet transformed).
    config : dict, optional
        If not provided, the module-level GLOBAL_CONFIG must be initialized via init_artifacts().
    model : optional
        Optional model object (sklearn-like) to use. If not provided, GLOBAL_MODEL must be set.
    preprocessor : optional
        Optional preprocessor object (scikit-learn transformer). If not provided, GLOBAL_PREPROCESSOR must be set.
    return_df : bool
        If True, return a pandas.DataFrame with inputs + predictions. Otherwise, return list/dict.

    Returns
    -------
    If input was a single record (dict) and return_df False -> dict
    If input was a batch (list or DataFrame) and return_df False -> list[dict]
    If return_df True -> pandas.DataFrame (indexed same as processed df)
    """
    # Resolve config and artifacts
    cfg = config or GLOBAL_CONFIG
    if cfg is None:
        raise RuntimeError("Configuration not provided and GLOBAL_CONFIG is None. Call init_artifacts() or pass config.")

    mdl = model or GLOBAL_MODEL
    pproc = preprocessor or GLOBAL_PREPROCESSOR
    if mdl is None or pproc is None:
        raise RuntimeError("Model and preprocessor must be provided either as arguments or via init_artifacts().")

    # Normalize input to dataframe with expected columns
    df_raw = _normalize_input_to_df(input_data, cfg)
    if df_raw.empty:
        raise ValueError("No input records provided after normalization.")

    # Apply preprocessor.transform safely
    try:
        X_trans = pproc.transform(df_raw)
    except Exception as exc:
        # Helpful error message: commonly OneHotEncoder with unknown categories causes ValueError
        err_msg = (
            f"Preprocessor.transform failed: {exc}. "
            "Common causes: unseen categories with OneHotEncoder configured without handle_unknown='ignore', "
            "or type mismatches. Inspect the incoming payload values and ensure training preprocessor matches."
        )
        logger.exception(err_msg)
        raise RuntimeError(err_msg) from exc

    # Predict probabilities; use predict_proba if available else fallback to decision_function or predict
    probs = None
    labels = None
    try:
        if hasattr(mdl, "predict_proba"):
            probs = mdl.predict_proba(X_trans)[:, 1]
            labels = mdl.predict(X_trans)  # model's own label decision
        elif hasattr(mdl, "decision_function"):
    
            scores = mdl.decision_function(X_trans)
            probs = 1 / (1 + np.exp(-scores))
            labels = (probs >= cfg.get("serving", {}).get("threshold", 0.5)).astype(int)
        else:
            # fallback to predict only
            labels = mdl.predict(X_trans)
            probs = labels.astype(float)  # not ideal but fallback
    except Exception as exc:
        logger.exception(f"Model prediction failed: {exc}")
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    results_df = _format_results(df_raw, probs, labels, cfg)

    # Merge inputs and predictions in return if requested
    if return_df:
        merged = pd.concat([df_raw.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        return merged

    # Otherwise return list/dict structure
    output_records = []
    for idx, row in results_df.reset_index(drop=True).iterrows():
        rec = {
            "pred_label": int(row["pred_label"]),
            "pred_prob": float(row["pred_prob"]),
            "churn_risk": row["churn_risk"],
            "model_version": row["model_version"],
        }
        output_records.append(rec)

    # If input was a single dict, return single dict
    if isinstance(input_data, dict):
        return output_records[0]
    return output_records


# Example main for local debugging (do not use this in production API)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage: must ensure config.yaml is present and artifacts exist
    try:
        init_artifacts("config.yaml")
    except Exception as exc:
        logger.exception(f"Failed to initialize artifacts: {exc}")
        raise

    example_input = {
        "gender": "Male", 

        "SeniorCitizen": 0 ,
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

    # Single prediction
    pred = predict(example_input)
    print("Single prediction:", pred)

    # Batch prediction
    batch = [example_input, {**example_input, "tenure": 10, "MonthlyCharges": 99.99, "TotalCharges":185}]
    preds = predict(batch)
    print("Batch predictions:", preds)
