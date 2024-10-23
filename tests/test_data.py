from pathlib import Path

import pandas as pd
import pytest

from reservations.config import DataConfig
from reservations.data import DataLoader


@pytest.fixture
def sample_df():
    data = {
        "age": [25, 30, 22, None, 28],
        "income": [50000, 60000, None, 80000, 45000],
        "city": ["New York", "Los Angeles", "New York", "Chicago", None],
        "canceled": ["Canceled", "Not_Canceled", "Not_Canceled", "Canceled", "Canceled"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def config():
    return DataConfig(
        target="canceled",
        numerical_variables=["age", "income"],
        categorical_variables=["city"],
        test_size=0.2,
        random_state=42,
    )


@pytest.fixture
def data_loader(tmpdir, sample_df, config):
    csv_path = Path(tmpdir) / "sample_data.csv"
    sample_df.to_csv(csv_path, index=False)
    return DataLoader(path=csv_path, config=config)


# Test initialization
def test_data_loader_init(data_loader, config, sample_df):
    assert data_loader.path.exists()
    assert isinstance(data_loader.df, pd.DataFrame)
    assert data_loader.config == config
    pd.testing.assert_frame_equal(data_loader.df, sample_df)


# Test data splitting
def test_split_data(data_loader):
    X_train, X_test, y_train, y_test = data_loader._split_data()
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 4
    assert y_test.shape[0] == 1


# Test preprocessor creation
def test_create_preprocessor(data_loader):
    data_loader._create_preprocessor()
    assert "numerical" in data_loader.preprocessor.transformers[0][0]
    assert "categorical" in data_loader.preprocessor.transformers[1][0]


# Test target encoding
def test_encode_target(data_loader, sample_df):
    y_encoded = data_loader._encode_target(sample_df["canceled"])
    expected = pd.Series([1, 0, 0, 1, 1], name="canceled")
    pd.testing.assert_series_equal(y_encoded, expected)


# Test feature encoding
def test_encode_features(data_loader, sample_df):
    data_loader._create_preprocessor()
    X_train, _, _, _ = data_loader._split_data()
    X_encoded = data_loader._encode_features(X_train)

    # Verify feature transformation results
    assert isinstance(X_encoded, pd.DataFrame)
    assert X_encoded.shape[1] == 4
    assert "age" in X_encoded.columns
    assert "income" in X_encoded.columns
    assert any("city" in col for col in X_encoded.columns)
