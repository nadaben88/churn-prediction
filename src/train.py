import pandas as pd
import joblib
import yaml
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,classification_report, roc_auc_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from .utils import map_yes_no, map_gender , load_config , setup_logging ,evaluate_model


def load_data(config):
    """Load processed data."""
    processed_data_path = config["data"]["processed"]
    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=[config["preprocessing"]["target_column"]])
    y = df[config["preprocessing"]["target_column"]]
    return X, y

def load_preprocessor(config):
    """Load preprocessor from disk."""
    preprocessor_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["preprocessor_name"]
    preprocessor = joblib.load(preprocessor_path)
    print("Preprocessor loaded successfully.")  
    return preprocessor

def train_model(X_train, y_train, config):
    """Train the model using the specified configuration with hyperparameter tuning."""
    model_type = config["training"]["model_type"]
    if model_type == "XGBoost":
        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'scale_pos_weight': [1, 2.7, 5]
        }
        # Define multiple scorers
        scoring = {
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=config["training"]["random_state"]
            ),
            param_grid=param_grid,
            scoring=scoring,
            refit='roc_auc',  # Best model selected based on roc_auc
            cv=3,
            n_jobs=-1,
            verbose=2
        )
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        # Get the best model
        model = grid_search.best_estimator_
        # Print best parameters
        print("Best parameters found: ", grid_search.best_params_)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


def save_model(model, config):
    """Save the trained model to disk."""
    model_dir = Path(config["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config["artifacts"]["model_name"]
    joblib.dump(model, model_path)

def save_test_data(X_test, y_test, feature_names, config):
    """Save the test data to disk with aligned indices."""
    
    # Get directory and create it if needed
    test_data_path = Path(config["data"]["test"])
    test_data_path.parent.mkdir(parents=True, exist_ok=True)

    #Reset index alignment
    X_test_df = pd.DataFrame(X_test, columns=feature_names).reset_index(drop=True)
    y_test_df = pd.Series(y_test).reset_index(drop=True)

    #Add the target column AFTER alignment
    target_col = config["preprocessing"]["target_column"]
    X_test_df[target_col] = y_test_df

    # Save
    X_test_df.to_csv(test_data_path, index=False)
    print(f"Test dataset saved at: {test_data_path}")


def main():
    config = load_config()
    setup_logging()
    logging.info("Loading data from: %s", config["data"]["raw"])
    # Load data and preprocessor
    X, y = load_data(config)
    print("Features in X:", X.columns.tolist())  
    preprocessor = load_preprocessor(config)
    print("Is preprocessor fitted?", hasattr(preprocessor, 'named_transformers_'))
          
    
    # Transform data 
    X_preprocessed = preprocessor.transform(X)
    print("Transformed data shape:", X_preprocessed.shape)  
    # Load feature names (saved during preprocessing)
    feature_names = joblib.load(Path(config["artifacts"]["model_dir"]) / "feature_names.joblib")

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2,
        random_state=config["training"]["random_state"], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25,
        random_state=config["training"]["random_state"], stratify=y_train_val
    )
    # Train model
    model = train_model(X_train, y_train, config)
    # Evaluate on validation set
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    print("Validation Metrics:", val_metrics)
    y_test = y_test.reset_index(drop=True)

    
    # Save model and test data
    save_model(model, config)
    save_test_data(X_test, y_test, feature_names, config)


if __name__ == "__main__":
    main()

