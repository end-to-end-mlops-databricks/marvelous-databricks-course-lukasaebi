import numpy as np

from reservations.evaluate import Accuracy, Precision, Recall

y_true = np.array([0, 1, 1, 0, 0, 1])
y_pred = np.array([0, 0, 1, 0, 1, 1])


def test_accuracy():
    accuracy = Accuracy()

    acc = accuracy.calculate(y_true, y_pred)
    assert acc == 0.6666666666666666


def test_precision():
    precision = Precision()

    prec = precision.calculate(y_true, y_pred)
    assert prec == 0.6666666666666666


def test_recall():
    recall = Recall()

    rec = recall.calculate(y_true, y_pred)
    assert rec == 0.6666666666666666
