import numpy as np


class Sigmoid:
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.function(x) * (1 - self.function(x))


class Tanh:
    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU:
    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x < 0, 0, 1)


class ELU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def function(self, x):
        return np.where(x < 0, self.alpha * (np.exp(x) - 1), x)

    def derivative(self, x):
        return np.where(x < 0, self.function(x) + self.alpha, 1)


class LeakyReLU:
    def function(self, x):
        return np.where(x < 0, 0.01 * x, x)

    def derivative(self, x):
        return np.where(x < 0, 0.01, 1)


ACTIVATION_FUNCTIONS = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "elu": ELU,
    "leaky_relu": LeakyReLU,
}
