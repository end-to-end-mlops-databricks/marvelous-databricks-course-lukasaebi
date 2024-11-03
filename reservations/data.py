from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reservations.config import DataConfig


class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config

    def split_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> tuple:
        if y is None:
            return train_test_split(X, test_size=self.config.test_size, random_state=self.config.random_state)
        return train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state)


class DataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self._create_preprocessor()

    def _create_preprocessor(self) -> None:
        numerical_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, self.config.numerical_variables),
                ("categorical", categorical_pipeline, self.config.categorical_variables),
            ]
        )

    def preprocess_data(self, X: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
        X_encoded = self._encode_features(X.drop(target, axis=1))
        y_encoded = self._encode_target(X[target])
        return X_encoded, y_encoded

    def _encode_target(self, y: pd.Series) -> pd.Series:
        return y.map({"Canceled": 1, "Not_Canceled": 0})

    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = self.preprocessor.fit_transform(X)
        feature_names = [name.split("__")[-1] for name in self.preprocessor.get_feature_names_out()]
        id = feature_names.pop()
        return pd.DataFrame(np.c_[X_encoded[:, -1], X_encoded[:, :-1]], columns=[id, *feature_names])
