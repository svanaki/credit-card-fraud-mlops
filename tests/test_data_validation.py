"""
Tests for data preprocessing and validation.
"""

import pandas as pd

from src.prepare import (
    remove_duplicates,
    scale_features,
    split_data,
)


def create_sample_dataset():
    """Create a small synthetic dataset for preprocessing tests."""
    data = {
        "Time": list(range(20)),
        "Amount": [100 + i * 10 for i in range(20)],
        "V1": [0.1 * i for i in range(20)],
        "V2": [0.2 * i for i in range(20)],
        "Class": [0] * 10 + [1] * 10,
    }
    return pd.DataFrame(data)


def test_remove_duplicates():
    df = create_sample_dataset()

    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = remove_duplicates(duplicated)

    assert len(cleaned) == len(df)


def test_split_data():
    df = create_sample_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_column="Class",
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_scaled_columns_exist():
    df = create_sample_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_column="Class",
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
    )

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_val,
        X_test,
        ["Time", "Amount"],
    )

    assert "Time" in X_train_scaled.columns
    assert "Amount" in X_train_scaled.columns

    assert X_train_scaled.shape == X_train.shape
    assert X_val_scaled.shape == X_val.shape
    assert X_test_scaled.shape == X_test.shape
