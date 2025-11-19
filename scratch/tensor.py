import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # --- Helper: Unbroadcast gradient to target shape ---
    @staticmethod
    def _unbroadcast(grad, shape):
        """
        Sum grad to match 'shape' after broadcasting.
        grad: np.ndarray (upstream gradient)
        shape: tuple (target shape)
        """
        if grad is None:
            return None
        g = grad
        # sum extra leading dims
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        # sum over dims that were broadcast (shape dim ==1)
        for i, dim in enumerate(shape):
            if dim == 1:
                g = g.sum(axis=i, keepdims=True)
        return g

    # ---- Core ops ----
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = Tensor._unbroadcast(out.grad, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = Tensor._unbroadcast(out.grad, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad * other.data
                grad_self = Tensor._unbroadcast(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = out.grad * self.data
                grad_other = Tensor._unbroadcast(grad_other, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad / other.data
                grad_self = Tensor._unbroadcast(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = -out.grad * self.data / (other.data ** 2)
                grad_other = Tensor._unbroadcast(grad_other, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Tensor(self.data ** power, (self,), f'**{power}')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = (power * (self.data ** (power - 1))) * out.grad
                grad_self = Tensor._unbroadcast(grad_self, self.data.shape)
                self.grad += grad_self

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                # out.grad shape is (m,p), other.data.T shape (p,n) => (m,n)
                grad_self = out.grad @ other.data.T
                # should already match self.data.shape
                grad_self = Tensor._unbroadcast(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                grad_other = Tensor._unbroadcast(grad_other, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    # ---- Unary ops ----
    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, (self,), 'tanh')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += (1 - out_data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, (self,), 'relu')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, (self,), 'exp')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += out_data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out_data = np.log(self.data)
        out = Tensor(out_data, (self,), 'log')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out
        
    # ---- Indexing ----
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'slice')

        def _backward():
            if not self.requires_grad:
                return
            
            # advanced indexing case
            try:
                np.add.at(self.grad, idx, out.grad)
            except (TypeError, IndexError):
                # fallback simple slice
                self.grad[idx] += out.grad

        out._backward = _backward
        return out

    # ---- Reductions ----
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'sum')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)

        # ensure array shape even if scalar
        out_data = np.array(out_data, dtype=np.float32)

        # divisor depends on axis
        if axis is None:
            divisor = self.data.size
        elif isinstance(axis, int):
            divisor = self.data.shape[axis]
        else:
            divisor = np.prod([self.data.shape[a] for a in axis])


        out = Tensor(out_data, (self,), 'mean')

        def _backward():
            if self.requires_grad:
                grad = out.grad / divisor

                # match reduced dimensions
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)

                # expand grad over original shape
                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    
    def clamp(self, min_val, max_val):
        clamped = np.clip(self.data, min_val, max_val)
        out = Tensor(clamped, (self,), 'clamp')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                mask = (self.data >= min_val) & (self.data <= max_val)
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        out_data = np.where(self.data > 0, self.data, alpha * self.data)
        out = Tensor(out_data, (self,), 'leaky_relu')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                grad = np.where(self.data > 0, 1, alpha)
                self.grad += out.grad * grad

        out._backward = _backward
        return out


    def softmax(self):
        exps = (self.data - self.data.max(axis=1, keepdims=True))
        exps = np.exp(exps)
        out_data = exps / exps.sum(axis=1, keepdims=True)
        out = Tensor(out_data, (self,), 'softmax')

        def _backward():
            # optional: implement full softmax backward
            # but real frameworks use log_softmax + nll only
            pass

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        # numerically stable log-softmax
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        sum_exps = np.sum(exps, axis=axis, keepdims=True)

        out_data = shifted - np.log(sum_exps)  # log_softmax outputs

        out = Tensor(out_data, (self,), 'log_softmax')

        def _backward():
            if not self.requires_grad:
                return

            # softmax = exp(log_softmax)
            softmax_output = np.exp(out_data)
            grad_out = out.grad

            # vectorized Jacobian for log_softmax:
            # dL/dx = grad_out - softmax * sum(grad_out)
            sum_grad = np.sum(grad_out, axis=axis, keepdims=True)
            grad_self = grad_out - softmax_output * sum_grad

            self.grad += grad_self

        out._backward = _backward
        return out


    # ---- Backprop ----
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # seed gradient for root (dL/dL = 1)
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    # ---- Shape ----
    def shape(self):
        return self.data.shape