import yaml
import logging
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score,accuracy_score


 
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.basicConfig(
        filename=config["logging"]["log_file"],
        level=config["logging"]["level"],
        format=config["logging"]["format"]
    )

def load_model(config):
    """Load the trained model from disk."""
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_name"]
    model = joblib.load(model_path)
    logging.info("Model loaded from %s", model_path)
    return model

def evaluate_model(model, X, y, set_name="Validation"):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = {
        "Recall": recall_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "AUC-ROC": roc_auc_score(y, y_proba),
        "Accuracy":accuracy_score(y, y_pred),
    }
    print(f"\n{set_name} Set Metrics:")
    print(classification_report(y, y_pred))
    return metrics , y_pred, y_proba

def map_yes_no(X):
    return X.map(lambda val: 1 if val == "Yes" else 0)

def map_gender(X):
    return X.map(lambda val: 1 if val == "Male" else 0)

