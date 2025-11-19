import numpy as np
from scratch.tensor import Tensor

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            end = start + self.batch_size
            batch_idx = indices[start:end]

            xb = self.X[batch_idx]
            yb = self.y[batch_idx]
            
            # Flatten y if it's 2D with shape (batch, 1) for compatibility with loss functions
            if yb.ndim == 2 and yb.shape[1] == 1:
                yb = yb.flatten()

            xb = Tensor(xb, requires_grad=False)
            yb = Tensor(yb, requires_grad=False)

            yield xb, yb
