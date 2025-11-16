import numpy as np
from scratch.tensor import Tensor
from scratch.nn.linear import Linear
from scratch.nn.activations import ReLU, Sigmoid
from scratch.nn.loss import Loss
from scratch.nn.optim import Adam

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.history = []
    def fit(self, dataloader, epochs=10):
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            steps = 0

            for x, y in dataloader:

                x = Tensor(x, requires_grad=False)
                y = Tensor(y, requires_grad=False)
                # Forward
                y_pred = self.model(x)

                # Loss
                loss = self.loss_fn(y_pred, y)

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.data
                steps += 1

            avg_loss = total_loss / steps

            if epoch%100==0: print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.6f}")
            self.history.append((epoch, avg_loss))
