# Scratch AI

A deep learning framework built from scratch in Python - no PyTorch, no TensorFlow, just NumPy and pure Python.

## Overview

Scratch AI is a fully functional deep learning library implementing automatic differentiation, neural network layers, optimizers, and training infrastructure from first principles. The goal is to understand how modern ML frameworks work under the hood by building everything from scratch.

## Features

### 🔥 Core Tensor System
- **Custom Tensor class** with automatic differentiation (autograd)
- **Arithmetic operations**: `+`, `-`, `*`, `/`, `**`, `@` (matrix multiplication)
- **Unary operations**: `tanh`, `relu`, `exp`, `log`, `sigmoid`, `leaky_relu`, `clamp`
- **Reduction operations**: `sum`, `mean` (with axis and keepdims support)
- **Advanced operations**: `softmax`, `log_softmax` (numerically stable)
- **Indexing and slicing** with gradient support
- **Broadcasting** with automatic gradient unbroadcasting
- **Backpropagation** through arbitrary computation graphs

### 🧠 Neural Network Components

**Layers:**
- `Linear` - Fully connected layer with Xavier initialization

**Activations:**
- `ReLU` - Rectified Linear Unit
- `LeakyReLU` - Leaky ReLU with configurable alpha
- `Tanh` - Hyperbolic tangent
- `Sigmoid` - Sigmoid activation
- `Softmax` - Numerically stable softmax
- `LogSoftmax` - Log-softmax for numerical stability

**Loss Functions:**
- `MSE` - Mean Squared Error
- `CrossEntropy` - Cross-Entropy Loss for multi-class classification
- `BinaryCrossEntropy` - Binary Cross-Entropy for binary classification

**Optimizers:**
- `SGD` - Stochastic Gradient Descent
- `Momentum` - SGD with momentum
- `Adam` - Adaptive Moment Estimation

**Model Composition:**
- `Sequential` - Stack layers to build deep networks

### 🚀 Training Infrastructure

- **DataLoader**: Batch processing with shuffling
- **Trainer**: High-level training loop with progress tracking
- **Profiling utilities**: CPU and memory profiling decorators
- **Visualization**: Training history plotting

### ✅ Comprehensive Testing

- **93 tests** across 5 test files
- Full coverage of tensor operations (forward and backward passes)
- Tests for layers, activations, loss functions, and model composition
- Edge case and numerical stability testing

## Project Structure

```
scratch/
├── tensor.py          # Core Tensor class with autograd
├── dataloader.py      # DataLoader for batch processing
├── trainer.py         # Training loop infrastructure
├── utils.py           # Profiling and visualization utilities
├── env.py             # Environment configuration
└── nn/
    ├── linear.py      # Linear (fully connected) layer
    ├── activations.py # Activation function modules
    ├── loss.py        # Loss functions
    ├── optim.py       # Optimizers (SGD, Momentum, Adam)
    └── sequential.py  # Sequential model container

tests/                 # Comprehensive test suite
├── test_tensor_ops.py
├── test_layers.py
├── test_activations.py
├── test_loss.py
└── test_sequential.py

demo_*.ipynb          # Example notebooks
```

## Quick Start

### Basic Tensor Operations

```python
from scratch.tensor import Tensor

# Create tensors
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])

# Operations with automatic gradient tracking
z = (x * y).sum()
z.backward()  # Compute gradients

print(x.grad)  # Gradients w.r.t. x
```

### Building a Neural Network

```python
from scratch.nn.linear import Linear
from scratch.nn.activations import ReLU
from scratch.nn.sequential import Sequential
from scratch.nn.loss import Loss
from scratch.nn.optim import Adam
from scratch.trainer import Trainer
from scratch.dataloader import DataLoader

# Build model
model = Sequential(
    Linear(2, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 3)
)

# Setup training
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = Loss.cross_entropy

# Create trainer
trainer = Trainer(model, optimizer, loss_fn)

# Train
dataloader = DataLoader(X_train, y_train, batch_size=32)
trainer.fit(dataloader, epochs=100)
```

## Examples

Check out the demo notebooks for complete examples:

- `demo_scratch_spiral.ipynb` - Multi-class classification on spiral dataset
- `demo_scratch_spiral_2.ipynb` - Advanced spiral classification
- `basics.ipynb` - Basic tensor operations and concepts

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tensor_ops.py -v

# Run with coverage
python -m pytest tests/ --cov=scratch --cov-report=html
```

## Key Implementation Details

### Automatic Differentiation
The framework implements reverse-mode automatic differentiation (backpropagation):
- Each tensor tracks its computational history
- Gradients flow backward through the computation graph
- Supports arbitrary computation graphs with proper gradient accumulation

### Broadcasting
Operations handle NumPy-style broadcasting with automatic gradient unbroadcasting to ensure gradients flow correctly to tensors of different shapes.

### Numerical Stability
Special care is taken for numerically sensitive operations:
- `log_softmax` uses the log-sum-exp trick
- Loss functions include epsilon for stability
- `clamp` operation to prevent overflow/underflow

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- Loguru (for logging)
- tqdm (for progress bars)
- pytest (for testing, optional)

## Purpose

Built to deeply understand how modern deep learning frameworks like PyTorch work under the hood. By implementing everything from scratch, this project reveals the fundamental algorithms and design patterns that power AI systems.

## License

See LICENSE file for details.
