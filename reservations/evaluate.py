from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Metric(ABC):
    @staticmethod
    @abstractmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class Accuracy(Metric):
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)


class Precision(Metric):
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred)


class Recall(Metric):
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred)
