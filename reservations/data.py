from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from reservations.config import DataConfig


class DataLoader:
    def __init__(self, df: str | Path | pd.DataFrame, config: DataConfig):
        self.df = df
        self.config = config
        self.preprocesser = None

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: str | Path | pd.DataFrame):
        if isinstance(df, (str, Path)):
            self._df = pd.read_csv(df)
        elif isinstance(df, pd.DataFrame):
            self._df = df
        else:
            raise ValueError("Data must be a path to a csv file or a pandas DataFrame.")

    def preprocess_data(self) -> tuple:
        X_train, X_test, y_train, y_test = self._split_data()
        self._create_preprocessor()
        y_train = self._encode_target(y_train)
        y_test = self._encode_target(y_test)
        X_train = self._encode_features(X_train)
        X_test = self._encode_features(X_test)
        return X_train, X_test, y_train, y_test

    def _split_data(self) -> list:
        return train_test_split(
            self.df.drop(self.config.target, axis=1),
            self.df[self.config.target],
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

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

    def _encode_target(self, y: pd.Series) -> pd.Series:
        return y.map({"Canceled": 1, "Not_Canceled": 0})

    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = self.preprocessor.fit_transform(X)
        feature_names = [name.split("__")[-1] for name in self.preprocessor.get_feature_names_out()]
        return pd.DataFrame(X_encoded, columns=feature_names)
