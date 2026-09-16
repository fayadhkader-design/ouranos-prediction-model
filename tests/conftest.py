import pytest
from src.pipeline import train_default


@pytest.fixture(scope="session")
def model():
    return train_default()
