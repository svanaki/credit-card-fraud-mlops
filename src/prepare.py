"""
Data preprocessing script for the Credit Card Fraud Detection MLOps project.

This script:
- Loads the raw dataset
- Removes duplicate rows
- Splits data into train, validation, and test sets
- Scales Time and Amount using RobustScaler
- Saves processed datasets
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import load_config

RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DATA_DIR = Path("data/processed")

TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def load_data(path: Path) -> pd.DataFrame:
    """Load raw credit card fraud dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates().reset_index(drop=True)


def split_data(
    df,
    target_column,
    test_size,
    validation_size,
    random_state,
):
    """Split dataset into stratified train, validation, and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    validation_ratio = validation_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, columns_to_scale):
    """Fit RobustScaler on training data and transform all data splits."""
    scaler = RobustScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(
        X_train_scaled[columns_to_scale]
    )

    X_val_scaled[columns_to_scale] = scaler.transform(X_val_scaled[columns_to_scale])

    X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_scaler(scaler, scaler_path: Path) -> None:
    """Save the fitted feature scaler for inference."""
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)


def save_processed_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    processed_data_dir,
    target_column,
):
    """Save processed train, validation, and test datasets."""
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    val = X_val.copy()
    test = X_test.copy()

    train[target_column] = y_train.values
    val[target_column] = y_val.values
    test[target_column] = y_test.values

    train.to_csv(processed_data_dir / "train.csv", index=False)
    val.to_csv(processed_data_dir / "val.csv", index=False)
    test.to_csv(processed_data_dir / "test.csv", index=False)


def main():
    """Run the preprocessing pipeline."""
    config = load_config()

    raw_data_path = Path(config["data"]["raw_path"])
    processed_data_dir = Path(config["data"]["processed_dir"])
    target_column = config["data"]["target_column"]

    model_dir = Path(config["paths"]["model_dir"])
    scaler_name = config["paths"]["scaler_name"]
    scaler_path = model_dir / scaler_name

    preprocessing_config = config["preprocessing"]
    remove_duplicate_rows = preprocessing_config["remove_duplicates"]
    scale_columns = preprocessing_config["scale_columns"]
    test_size = preprocessing_config["test_size"]
    validation_size = preprocessing_config["validation_size"]
    random_state = preprocessing_config["random_state"]

    print("Loading raw dataset...")
    df = load_data(raw_data_path)

    print(f"Original dataset shape: {df.shape}")

    if remove_duplicate_rows:
        print("Removing duplicate rows...")
        df = remove_duplicates(df)
        print(f"Dataset shape after removing duplicates: {df.shape}")

    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_column,
        test_size,
        validation_size,
        random_state,
    )

    print("Scaling selected columns...")
    X_train, X_val, X_test, scaler = scale_features(
        X_train,
        X_val,
        X_test,
        scale_columns,
    )

    print("Saving processed datasets...")
    save_processed_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        processed_data_dir,
        target_column,
    )

    print("Saving fitted scaler...")
    save_scaler(scaler, scaler_path)

    print(f"Scaler saved to: {scaler_path}")

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
