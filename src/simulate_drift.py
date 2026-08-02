"""
Create a synthetic current dataset with intentional feature drift.

This file is for monitoring and retraining demonstrations only.
It must not be treated as real production data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path("data/processed/test.csv")
DEFAULT_OUTPUT_PATH = Path("data/monitoring/simulated_drift.csv")
TARGET_COLUMN = "Class"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the source dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Input dataset is empty: {path}")

    return dataframe


def simulate_drift(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic distribution shifts to selected features."""
    drifted_data = dataframe.copy()
    rng = np.random.default_rng(RANDOM_STATE)

    # Shift Time and Amount distributions.
    drifted_data["Time"] = (
        drifted_data["Time"] * 1.40
        + rng.normal(0, 0.25, len(drifted_data))
    )

    drifted_data["Amount"] = (
        drifted_data["Amount"] * 1.80
        + rng.normal(0, 0.30, len(drifted_data))
    )

    # Shift enough PCA features for dataset-level drift detection.
    drift_columns = [f"V{i}" for i in range(1, 19)]

    for index, column in enumerate(drift_columns, start=1):
        location_shift = 0.75 + (index * 0.04)
        scale_factor = 1.20 + (index * 0.01)

        drifted_data[column] = (
            drifted_data[column] * scale_factor
            + location_shift
            + rng.normal(0, 0.10, len(drifted_data))
        )

    # Keep the target unchanged.
    if TARGET_COLUMN in dataframe.columns:
        drifted_data[TARGET_COLUMN] = dataframe[TARGET_COLUMN]

    return drifted_data


def save_data(dataframe: pd.DataFrame, path: Path) -> None:
    """Save the synthetic drift dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a synthetic dataset with intentional drift."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Source dataset used to create the simulated current data.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the generated drift dataset.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the drift simulation workflow."""
    arguments = parse_arguments()

    print(f"Loading source data: {arguments.input}")
    source_data = load_data(arguments.input)

    print(f"Source shape: {source_data.shape}")
    print("Applying synthetic feature drift...")

    drifted_data = simulate_drift(source_data)
    save_data(drifted_data, arguments.output)

    print("Synthetic drift dataset created successfully.")
    print(f"Output path: {arguments.output}")
    print(
        "Shifted features: Time, Amount, and V1-V18. "
        "The Class target was not modified."
    )


if __name__ == "__main__":
    main()