import numpy as np
from scratch.tensor import Tensor
class Linear:
    def __init__(self, in_features, out_features):
        # Xavier initialization
        scale = (1.0 / in_features) ** 0.5
        W = np.random.randn(in_features, out_features) * scale
        b = np.zeros((1, out_features))

        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def __call__(self, x):
        return x @ self.W + self.b
