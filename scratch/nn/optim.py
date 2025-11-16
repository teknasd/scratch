import numpy as np

class Optim:
    def __init__(self, params, lr):
        self.param_groups = [{
            "params": list(params),
            "lr": lr
        }]
        self.state = {}   # param → state dict

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = np.zeros_like(p.grad)


class SGD(Optim):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
              if p.requires_grad:
                p.data -= group["lr"] * p.grad


class Momentum(Optim):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum

        for p in self.param_groups[0]["params"]:
            self.state[p] = {"velocity": np.zeros_like(p.data)}

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                v = self.state[p]["velocity"]
                v[:] = self.momentum * v + lr * p.grad
                p.data -= v

class Adam(Optim):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        for p in self.param_groups[0]["params"]:
            self.state[p] = {
                "m": np.zeros_like(p.data),
                "v": np.zeros_like(p.data),
                "t": 0
            }

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                st = self.state[p]

                st["t"] += 1
                g = p.grad

                # update biased estimates
                st["m"] = self.b1 * st["m"] + (1 - self.b1) * g
                st["v"] = self.b2 * st["v"] + (1 - self.b2) * (g * g)

                # bias correction
                m_hat = st["m"] / (1 - self.b1 ** st["t"])
                v_hat = st["v"] / (1 - self.b2 ** st["t"])

                # parameter update
                p.data -= lr * m_hat / (np.sqrt(v_hat) + self.eps)