"""Shared pytest setup: import paths and small fixtures used by several test files."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "demo"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

MAIN_CONFIG = ROOT / "configs" / "train_rtx5060.yaml"
DATASET = ROOT / "data" / "raw" / "sentences.csv"
SAMPLE_DIRS = [ROOT / "demo" / "examples", ROOT / "demo" / "test_files"]
EXPECTED_LABELS = ROOT / "demo" / "examples" / "expected_labels.json"


@pytest.fixture(scope="session")
def main_config():
    import yaml

    with open(MAIN_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def label_list(main_config):
    return main_config["data"]["label_list"]


@pytest.fixture(scope="session")
def dataset():
    import pandas as pd

    # dtype=str + keep_default_na=False: read every cell as written (no "TRUE" -> bool, no "" -> NaN)
    return pd.read_csv(DATASET, dtype=str, keep_default_na=False)
