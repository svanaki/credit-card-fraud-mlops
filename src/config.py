from pathlib import Path

import yaml


def load_config(config_path: Path = Path("params.yaml")) -> dict:
    """Load project configuration from params.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)