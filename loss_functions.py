import numpy as np


class MeanSquaredError:
    def loss(self, y_hat, y):
        return np.sum((y_hat - y) ** 2, axis=1, keepdims=True)

    def derivative(self, y_hat, y):
        return np.sum(2 * (y_hat - y), axis=1, keepdims=True)


class MeanAbsoluteError:
    def loss(self, y_hat, y):
        return np.sum(np.absolute(y_hat - y), axis=1, keepdims=True)

    def derivative(self, y_hat, y):
        return np.sum(np.where(y_hat > y, 1, -1), axis=1, keepdims=True)


class CrossEntropy:
    def loss(self, y_hat, y):
        return -np.sum(y * np.log(y_hat), axis=1, keepdims=True)

    def derivative(self, y_hat, y):
        return -np.sum(y / y_hat - ((1 - y) / (1 - y_hat)), axis=1, keepdims=True)


LOSS_FUNCTIONS = {
    "mean_squared_error": MeanSquaredError,
    "mean_absolute_error": MeanAbsoluteError,
    "cross_entropy": CrossEntropy
}
