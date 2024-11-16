from abc import ABC, abstractmethod

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Model(ABC, mlflow.pyfunc.PythonModel):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass


class RandomForestModel(Model):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class DecisionTreeModel(Model):
    def __init__(
        self,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
