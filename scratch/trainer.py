import numpy as np
from loguru import logger
from tqdm import tqdm
from time import monotonic
from scratch.utils import profile_cpu, profile_mem
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

    @profile_cpu()
    @profile_mem()
    def fit(self, dataloader, epochs=10):

        # Count total parameters in the model
        total_params = sum(np.prod(param.data.shape) for param in self.model.parameters())
        logger.info(f"Training model with {total_params} parameters for {epochs} epochs")
        start_time = monotonic()
        for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
            total_loss = 0.0
            steps = 0

            for x, y in dataloader:
                assert isinstance(x, Tensor) and isinstance(y, (Tensor, int))

                # logger.info(f"x: {x.shape()}, y: {y.shape()}")
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

            if epoch%100==0: logger.info(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.6f}")
            self.history.append((epoch, avg_loss))
        end_time = monotonic()
        logger.info(f"Training time: {end_time - start_time:.2f} seconds")