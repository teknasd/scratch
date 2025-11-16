class ReLU:
    def __call__(self, x):
        return x.relu()   # we implement Tensor.relu() next


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return x.leaky_relu(self.alpha)


class Tanh:
    def __call__(self, x):
        return x.tanh()

class Sigmoid:
    def __call__(self, x):
        return x.sigmoid()

class Softmax:
    def __call__(self, x):
        return x.softmax()

class LogSoftmax:
    def __call__(self, x):
        return x.log_softmax()

        