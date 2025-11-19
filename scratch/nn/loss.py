from scratch.tensor import Tensor
import numpy as np
from loguru import logger

class Loss:
    # @staticmethod
    # def cross_entropy(logits, target_index):
    #     # convert tensor → Python int
    #     if isinstance(target_index, Tensor):
    #         target_index = int(target_index.data.item() if target_index.data.ndim > 0 else target_index.data)

    #     log_probs = logits.log_softmax()
    #     return -log_probs[0, target_index]  

    @staticmethod
    def cross_entropy(logits, targets):
        """
        logits: Tensor shape (B, C)
        targets: Tensor shape (B,) containing class indices
        """

        # Convert to numpy integer array
        if isinstance(targets, Tensor):
            targets = targets.data.astype(int)

        # log_softmax over classes
        log_probs = logits.log_softmax()      # (B, C)

        # Pick correct class for each sample → advanced indexing
        batch_indices = np.arange(log_probs.data.shape[0])
        picked = log_probs[batch_indices, targets]    # uses Tensor.__getitem__

        # We want mean loss over batch
        return -(picked).mean()

        
    @staticmethod
    def mse(pred, target):
        diff = pred - target
        return (diff * diff).mean()


    
    @staticmethod
    def binary_cross_entropy(pred, target):
        """
        pred: Tensor, probability after sigmoid
        target: Tensor or scalar 0/1
        """

        eps = 1e-12  # for numerical stability
        pred = pred.clamp(eps, 1 - eps)  # avoid log(0)


        # ensure target is a Tensor
        if not isinstance(target, Tensor):
            target = Tensor(np.array(target), requires_grad=False)

        loss = -(target * pred.log() + (1 - target) * (1 - pred).log())
        return loss.mean()
