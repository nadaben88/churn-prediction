import pandas as pd
import joblib
import yaml
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,  roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from .utils import load_config, setup_logging,load_model ,evaluate_model



def load_test_data(config):
    """Load test data from disk."""
    test_data_path = Path(config["data"]["test"])
    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=[config["preprocessing"]["target_column"]])
    y_test = test_df[config["preprocessing"]["target_column"]]
    logging.info("Test data loaded from %s", test_data_path)
    return X_test, y_test


def save_metrics(metrics, config):
    """Save metrics to a JSON file."""
    metrics_dir = Path(config["artifacts"]["model_dir"])
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Metrics saved to %s", metrics_path)

def plot_class_distribution(y_test, config):
    """Plot and save class distribution."""
    fig, ax = plt.subplots()
    y_test.value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    fig_path = Path("Figures") / "class_distribution.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info("Class distribution plot saved to %s", fig_path)

def plot_confusion_matrix(y_test, y_pred, config):
    """Plot and save normalized confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", ax=ax, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix")
    fig_path = Path("Figures") / "confusion_matrix.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info("Confusion matrix plot saved to %s", fig_path)

def plot_roc_curve(y_test, y_proba, config):
    """Plot and save ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig_path = Path("Figures") / "roc_curve.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info("ROC curve plot saved to %s", fig_path)

def plot_feature_importance(model, config):
    """Plot and save feature importance."""
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots()
        # Load the saved feature names
        feature_names = joblib.load(Path(config["artifacts"]["model_dir"]) / "feature_names.joblib")
        importance = pd.Series(model.feature_importances_, index=feature_names)
        importance.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Feature Importance")
        fig_path = Path("Figures") / "feature_importance.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logging.info("Feature importance plot saved to %s", fig_path)
    else:
        logging.warning("Model does not support feature importance.")


def plot_precision_recall_curve(y_test, y_proba, config):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig_path = Path("Figures") / "precision_recall_curve.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info("Precision-recall curve plot saved to %s", fig_path)

def plot_heatmap(X_test, config):
    """Plot and save correlation heatmap of test features."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(X_test.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig_path = Path("Figures") / "heatmap.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logging.info("Heatmap plot saved to %s", fig_path)

def main():
    config = load_config()
    setup_logging()

    # Load model and test data
    model = load_model(config)
    X_test, y_test = load_test_data(config)

    # Evaluate on test set
    test_metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, "Test")
    logging.info("Test Metrics: %s", test_metrics)

    # Save metrics
    save_metrics(test_metrics, config)

    # Generate and save plots
    plot_class_distribution(y_test, config)
    plot_confusion_matrix(y_test, y_pred, config)
    plot_roc_curve(y_test, y_proba, config)
    plot_feature_importance(model, config)
    plot_precision_recall_curve(y_test, y_proba, config)
    plot_heatmap(X_test, config)

if __name__ == "__main__":
    main()
