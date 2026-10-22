import os
from pathlib import Path

import pytest

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "acts.yaml").exists():
        pytest.skip(f"data directory not found: {DATA_DIR}")
    return DATA_DIR
