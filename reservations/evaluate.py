from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Metric(ABC):
    @classmethod
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class Accuracy(Metric):
    @classmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)


class Precision(Metric):
    @classmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred)


class Recall(Metric):
    @classmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred)
