import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from reservations.model import DecisionTreeModel, Model, RandomForestModel


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


# Test if Model class cannot be instantiated directly (due to ABC)
def test_abstract_model_instantiation():
    with pytest.raises(TypeError):
        Model()


def test_random_forest_model_init():
    model = RandomForestModel(n_estimators=50, max_depth=3)
    assert isinstance(model.model, RandomForestClassifier)
    assert model.model.n_estimators == 50
    assert model.model.max_depth == 3


def test_decision_tree_model_init():
    model = DecisionTreeModel(max_depth=4)
    assert isinstance(model.model, DecisionTreeClassifier)
    assert model.model.max_depth == 4


def test_random_forest_model_fit_predict(sample_data):
    X, y = sample_data
    model = RandomForestModel()
    model.fit(X, y)

    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape


def test_decision_tree_model_fit_predict(sample_data):
    X, y = sample_data
    model = DecisionTreeModel()
    model.fit(X, y)

    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == y.shape


def test_random_forest_model_predict_shape(sample_data):
    X, y = sample_data
    model = RandomForestModel()
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (100,)


def test_decision_tree_model_predict_shape(sample_data):
    X, y = sample_data
    model = DecisionTreeModel()
    model.fit(X, y)

    predictions = model.predict(X)
    assert predictions.shape == (100,)
