import numpy as np
from scratch.tensor import Tensor

# viz : https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

def im2col(x, kernel_h, kernel_w, stride, padding):
    """
    Converts 4D input tensor to 2D matrix using im2col algorithm.
    This allows convolution to be performed as matrix multiplication.
    
    Args:
        x: input (batch, channels, height, width)
        kernel_h, kernel_w: kernel dimensions
        stride: stride
        padding: padding
    
    Returns:
        col: 2D array of shape (batch * out_h * out_w, in_channels * kernel_h * kernel_w)
    """
    batch, channels, h, w = x.shape
    
    # Apply padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    # Calculate output dimensions
    out_h = (x.shape[2] - kernel_h) // stride + 1
    out_w = (x.shape[3] - kernel_w) // stride + 1
    
    # Create column matrix
    col = np.zeros((batch, channels, kernel_h, kernel_w, out_h, out_w))
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_pos in range(kernel_w):
            x_max = x_pos + stride * out_w
            col[:, :, y, x_pos, :, :] = x[:, :, y:y_max:stride, x_pos:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_h * out_w, -1)
    return col

def col2im(col, input_shape, kernel_h, kernel_w, stride, padding):
    """
    Converts 2D matrix back to 4D tensor (inverse of im2col).
    
    Args:
        col: 2D array
        input_shape: original input shape (batch, channels, height, width)
        kernel_h, kernel_w: kernel dimensions
        stride: stride
        padding: padding
    
    Returns:
        x: 4D tensor of shape input_shape
    """
    batch, channels, h, w = input_shape
    
    # Padded dimensions
    h_padded = h + 2 * padding
    w_padded = w + 2 * padding
    
    # Output dimensions
    out_h = (h_padded - kernel_h) // stride + 1
    out_w = (w_padded - kernel_w) // stride + 1
    
    # Reshape col back
    col = col.reshape(batch, out_h, out_w, channels, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    
    # Create output
    x = np.zeros((batch, channels, h_padded, w_padded))
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x_pos in range(kernel_w):
            x_max = x_pos + stride * out_w
            x[:, :, y:y_max:stride, x_pos:x_max:stride] += col[:, :, y, x_pos, :, :]
    
    # Remove padding
    if padding > 0:
        return x[:, :, padding:-padding, padding:-padding]
    return x

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        2D Convolutional layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (filters)
            kernel_size: Size of the convolutional kernel (int or tuple)
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with He initialization
        k_h, k_w = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * k_h * k_w))
        W = np.random.randn(out_channels, in_channels, k_h, k_w) * scale
        b = np.zeros((out_channels, 1))
        
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)
    
    def parameters(self):
        return [self.W, self.b]
    
    def __call__(self, x):
        """
        Forward pass of Conv2D using im2col for efficiency
        
        Args:
            x: Tensor of shape (batch, channels, height, width)
        
        Returns:
            Tensor of shape (batch, out_channels, out_height, out_width)
        """
        batch, in_c, in_h, in_w = x.data.shape
        k_h, k_w = self.kernel_size
        
        # Calculate output dimensions
        if self.padding > 0:
            h_padded = in_h + 2 * self.padding
            w_padded = in_w + 2 * self.padding
        else:
            h_padded = in_h
            w_padded = in_w
        
        out_h = (h_padded - k_h) // self.stride + 1
        out_w = (w_padded - k_w) // self.stride + 1
        
        # Convert input to column matrix using im2col
        # col shape: (batch * out_h * out_w, in_channels * k_h * k_w)
        col = im2col(x.data, k_h, k_w, self.stride, self.padding)
        
        # Reshape weights for matrix multiplication
        # W shape: (out_channels, in_channels * k_h * k_w)
        W_col = self.W.data.reshape(self.out_channels, -1)
        
        # Perform convolution as matrix multiplication
        # out_col shape: (batch * out_h * out_w, out_channels)
        out_col = col @ W_col.T + self.b.data.T
        
        # Reshape output back to 4D
        out_data = out_col.reshape(batch, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        
        # Store for backward pass
        self._cache = (x, col, W_col)
        
        out = Tensor(out_data, (x, self.W, self.b), 'conv2d')
        
        def _backward():
            if out.grad is None:
                return
            
            # Reshape output gradient
            # dout shape: (batch, out_channels, out_h, out_w)
            dout = out.grad
            dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
            
            # Gradient w.r.t weights
            if self.W.requires_grad:
                # dW = col.T @ dout_reshaped
                dW = col.T @ dout_reshaped
                dW = dW.T.reshape(self.W.data.shape)
                self.W.grad += dW
            
            # Gradient w.r.t bias
            if self.b.requires_grad:
                db = np.sum(dout_reshaped, axis=0, keepdims=True).T
                self.b.grad += db
            
            # Gradient w.r.t input
            if x.requires_grad:
                # dx_col = dout_reshaped @ W_col
                dx_col = dout_reshaped @ W_col
                
                # Convert column back to image
                dx = col2im(dx_col, x.data.shape, k_h, k_w, self.stride, self.padding)
                x.grad += dx
        
        out._backward = _backward
        return out


class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        """
        2D Max Pooling layer
        
        Args:
            kernel_size: Size of the pooling window (int or tuple)
            stride: Stride of the pooling (defaults to kernel_size)
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size
    
    def parameters(self):
        return []
    
    def __call__(self, x):
        """
        Forward pass of MaxPool2D using vectorized operations
        
        Args:
            x: Tensor of shape (batch, channels, height, width)
        
        Returns:
            Tensor of shape (batch, channels, out_height, out_width)
        """
        batch, channels, in_h, in_w = x.data.shape
        k_h, k_w = self.kernel_size
        
        # Calculate output dimensions
        out_h = (in_h - k_h) // self.stride + 1
        out_w = (in_w - k_w) // self.stride + 1
        
        # Use im2col to extract pooling windows
        col = im2col(x.data, k_h, k_w, self.stride, padding=0)
        # col shape: (batch * out_h * out_w, channels * k_h * k_w)
        
        # Reshape to separate channels
        col = col.reshape(batch, out_h, out_w, channels, k_h * k_w)
        col = col.transpose(0, 3, 1, 2, 4)  # (batch, channels, out_h, out_w, k_h * k_w)
        
        # Find max along the last dimension
        out_data = np.max(col, axis=4)
        
        # Store argmax for backward pass
        self._cache = (x, col, np.argmax(col, axis=4))
        
        out = Tensor(out_data, (x,), 'maxpool2d')
        
        def _backward():
            if out.grad is None or not x.requires_grad:
                return
            
            _, col, max_indices = self._cache
            batch, channels, out_h, out_w, pool_size = col.shape
            
            # Create gradient for column
            dout = out.grad  # (batch, channels, out_h, out_w)
            
            # Expand gradient to pool size
            dcol = np.zeros((batch, channels, out_h, out_w, pool_size))
            
            # Distribute gradient only to max positions
            for b in range(batch):
                for c in range(channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            max_idx = max_indices[b, c, i, j]
                            dcol[b, c, i, j, max_idx] = dout[b, c, i, j]
            
            # Reshape back
            dcol = dcol.transpose(0, 2, 3, 1, 4).reshape(batch * out_h * out_w, -1)
            
            # Convert back to image
            dx = col2im(dcol, x.data.shape, k_h, k_w, self.stride, padding=0)
            x.grad += dx
        
        out._backward = _backward
        return out


class Flatten:
    def __init__(self):
        """Flattens input while keeping batch dimension"""
        pass
    
    def parameters(self):
        return []
    
    def __call__(self, x):
        """
        Forward pass of Flatten
        
        Args:
            x: Tensor of shape (batch, channels, height, width)
        
        Returns:
            Tensor of shape (batch, channels * height * width)
        """
        batch = x.data.shape[0]
        self.input_shape = x.data.shape
        
        out_data = x.data.reshape(batch, -1)
        out = Tensor(out_data, (x,), 'flatten')
        
        def _backward():
            if out.grad is None or not x.requires_grad:
                return
            
            x.grad += out.grad.reshape(self.input_shape)
        
        out._backward = _backward
        return out

