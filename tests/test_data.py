import numpy as np
import pandas as pd
import pytest

from reservations.config import DataConfig
from reservations.data import DataLoader, DataPreprocessor


@pytest.fixture
def sample_df():
    data = {
        "age": [25, 30, 22, np.nan, 28],
        "income": [50000, 60000, np.nan, 80000, 45000],
        "city": ["New York", "Los Angeles", "New York", "Chicago", np.nan],
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
        catalog_name="catalog",
        schema_name="schema",
        volume_name="volume",
    )


# Tests for DataLoader
def test_data_loader_split_data_without_target(sample_df, config):
    data_loader = DataLoader(config=config)
    X_train, X_test = data_loader.split_data(sample_df.drop(columns=config.target))

    # Test the length of train and test sets
    assert len(X_train) == 4  # 80% of 5 samples
    assert len(X_test) == 1  # 20% of 5 samples


def test_data_loader_split_data_with_target(sample_df, config):
    data_loader = DataLoader(config=config)
    X_train, X_test, y_train, y_test = data_loader.split_data(
        sample_df.drop(columns=config.target), sample_df[config.target]
    )

    # Test the length of train and test sets
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert len(y_train) == 4
    assert len(y_test) == 1


# Tests for DataPreprocessor
def test_data_preprocessor_encode_target(sample_df, config):
    preprocessor = DataPreprocessor(config=config)
    y_encoded = preprocessor._encode_target(sample_df[config.target])

    # Check if the target is correctly encoded
    expected_encoded = pd.Series([1, 0, 0, 1, 1], name="canceled")
    pd.testing.assert_series_equal(y_encoded, expected_encoded)


def test_data_preprocessor_preprocess_data(sample_df, config):
    preprocessor = DataPreprocessor(config=config)
    X_encoded, y_encoded = preprocessor.preprocess_data(sample_df, target=config.target)

    # Check if y_encoded is as expected
    expected_y = pd.Series([1, 0, 0, 1, 1], name="canceled")
    pd.testing.assert_series_equal(y_encoded, expected_y)

    # Check the shape of X_encoded (should have two numerical and three one-hot encoded columns)
    assert X_encoded.shape == (5, 5)

    # Check if the column names are correct
    expected_columns = ["age", "income", "city_Chicago", "city_Los Angeles", "city_New York"]
    assert all(col in X_encoded.columns for col in expected_columns)
