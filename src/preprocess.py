import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from pathlib import Path
from .utils import map_yes_no, map_gender , load_config ,setup_logging


def load_raw_data(config):
    """Load raw data from the specified path."""
    raw_data_path = config["data"]["raw"]
    df = pd.read_csv(raw_data_path)
    return df

def clean_data(df, config):
    """Clean the data: handle missing values, drop columns, and create new features."""
    # Convert TotalCharges to numeric and impute missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # For missing values, we use the  calculation: MonthlyCharges * tenure
    df.loc[:, "TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df['TotalCharges'] = df.apply(
        lambda row: row['MonthlyCharges'] * (row['tenure'] - 1)
        if row['Churn'] == 'Yes'
        else row['MonthlyCharges'] * row['tenure'],
        axis=1
    )


    # Drop customerID
    df.drop(columns=config["preprocessing"]["drop_columns"], inplace=True)
    target_col = config["preprocessing"]["target_column"]
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})



    return df


def build_preprocessor(config):
    binary_categorical_cols = config["preprocessing"]["binary_categorical_columns"]
    nominal_categorical_cols = config["preprocessing"]["nominal_categorical_columns"]
    numerical_cols = config["preprocessing"]["numerical_columns"]

    # Preprocessing for numeric columns
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for nominal categorical columns
    nominal_categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Preprocessing for binary categorical columns (Yes/No → 0/1)
    binary_categorical_transformer = Pipeline(
        steps=[
            ("mapper", FunctionTransformer(map_yes_no, validate=False)),

        ]
    )

    # Preprocessing for gender (Male/Female → 0/1)
    gender_transformer = Pipeline(
        steps=[
            ("mapper", FunctionTransformer(map_gender, validate=False)),

        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("nominal_cat", nominal_categorical_transformer, nominal_categorical_cols),
            ("binary_cat", binary_categorical_transformer, binary_categorical_cols),
            ("gender", gender_transformer, ["gender"]),  
        ]
    )
    return preprocessor


def save_preprocessor(preprocessor, config):
    """Save the preprocessor to disk."""
    preprocessor_dir = Path(config["artifacts"]["model_dir"])
    preprocessor_dir.mkdir(parents=True, exist_ok=True)  
    preprocessor_path = preprocessor_dir / config["artifacts"]["preprocessor_name"]
    joblib.dump(preprocessor, preprocessor_path)

def preprocess_data(config):
    """Main function to preprocess data and save the preprocessor and feature names."""
    df = load_raw_data(config)
    df = clean_data(df, config)

    # Extract features (exclude target column)
    features = [col for col in df.columns if col != config["preprocessing"]["target_column"]]
    X = df[features]

    # Build and fit preprocessor
    preprocessor = build_preprocessor(config)
    preprocessor.fit(X)

    # Extract feature names after preprocessing
    numerical_cols = config["preprocessing"]["numerical_columns"]
    nominal_categorical_cols = config["preprocessing"]["nominal_categorical_columns"]
    binary_categorical_cols = config["preprocessing"]["binary_categorical_columns"]

    # Get one-hot encoded feature names for nominal categorical columns
    nominal_cat_transformer = preprocessor.named_transformers_["nominal_cat"]
    ohe_feature_names = nominal_cat_transformer.named_steps["onehot"].get_feature_names_out(nominal_categorical_cols)

    # Combine all feature names
    feature_names = numerical_cols + binary_categorical_cols + ["gender"] + list(ohe_feature_names)


    # Save feature names
    feature_names_dir = Path(config["artifacts"]["model_dir"])
    feature_names_dir.mkdir(parents=True, exist_ok=True)
    feature_names_path = feature_names_dir / "feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)

    # Save preprocessor
    save_preprocessor(preprocessor, config)

    # Save processed data
    processed_data_dir = Path(config["data"]["processed"].rsplit('/', 1)[0])
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_path = Path(config["data"]["processed"])
    df.to_csv(processed_data_path, index=False)


if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)
    setup_logging()
    logging.info("Loading data from: %s", config["data"]["raw"])


