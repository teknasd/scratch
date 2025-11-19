# Test Suite for Scratch Tensor Library

This directory contains comprehensive unit tests for the scratch tensor library.

## Test Coverage

### 1. Tensor Operations (`test_tensor_ops.py`)
Tests all fundamental tensor operations and their gradients:

**Basic Operations:**
- Tensor creation and initialization
- `requires_grad` functionality
- Gradient accumulation

**Arithmetic Operations:**
- Addition (`+`, `__add__`, `__radd__`)
- Subtraction (`-`, `__sub__`, `__rsub__`)
- Multiplication (`*`, `__mul__`, `__rmul__`)
- Division (`/`, `__truediv__`)
- Power (`**`, `__pow__`)
- Negation (`-`, `__neg__`)
- Matrix multiplication (`@`, `__matmul__`)

**Unary Operations:**
- `tanh()` - Hyperbolic tangent
- `relu()` - Rectified Linear Unit
- `exp()` - Exponential
- `log()` - Natural logarithm
- `sigmoid()` - Sigmoid activation
- `leaky_relu()` - Leaky ReLU with custom alpha
- `clamp()` - Clamp values to range

**Reduction Operations:**
- `sum()` - Sum with axis and keepdims support
- `mean()` - Mean with axis and keepdims support

**Indexing:**
- Single element indexing
- Slice indexing
- Multi-dimensional indexing

**Advanced Operations:**
- `softmax()` - Softmax activation
- `log_softmax()` - Numerically stable log-softmax

**Broadcasting:**
- Addition with broadcasting
- Multiplication with broadcasting
- Gradient unbroadcasting

**Complex Computation Graphs:**
- Multi-operation gradient flow
- Network-like computation graphs

### 2. Layers (`test_layers.py`)
Tests neural network layers:

**Linear Layer:**
- Initialization with Xavier scaling
- Forward pass computation
- Backward pass and gradients
- Parameter collection
- Batch processing

### 3. Activations (`test_activations.py`)
Tests all activation function wrappers:

**Activations:**
- `ReLU` - Rectified Linear Unit
- `LeakyReLU` - Leaky ReLU with configurable alpha
- `Tanh` - Hyperbolic tangent
- `Sigmoid` - Sigmoid activation
- `Softmax` - Softmax with numerical stability
- `LogSoftmax` - Log-softmax with numerical stability

Each activation is tested for:
- Forward pass correctness
- Backward pass gradients
- Output range validity
- Numerical stability

### 4. Loss Functions (`test_loss.py`)
Tests loss function implementations:

**Loss Functions:**
- `MSE` - Mean Squared Error
  - Perfect predictions
  - Gradient correctness
  - Matrix inputs
  
- `CrossEntropy` - Cross-Entropy Loss
  - Forward computation
  - Backward gradients
  - Correct vs incorrect predictions
  - Different target classes
  - Tensor and integer targets
  
- `BinaryCrossEntropy` - Binary Cross-Entropy Loss
  - Forward computation
  - Backward gradients
  - Perfect/wrong predictions
  - Numerical stability (edge cases)
  - Scalar and matrix inputs

### 5. Sequential Model (`test_sequential.py`)
Tests layer stacking and model composition:

**Sequential Features:**
- Layer initialization and stacking
- Forward pass through multiple layers
- Backward pass through entire network
- Parameter collection from all layers
- Deep networks (4+ layers)
- Different activation combinations
- Batch processing
- Single layer models
- Mixed layer types (with/without parameters)

**Integration Tests:**
- Complete training step (forward + backward)
- Regression network workflow
- Classification network workflow

## Running Tests

### Run All Tests
```bash
# Using pytest (recommended)
python -m pytest tests/ -v

# Using unittest
python -m unittest discover tests/

# Using the test runner script
python tests/run_all_tests.py
```

### Run Specific Test File
```bash
python -m pytest tests/test_tensor_ops.py -v
python -m pytest tests/test_layers.py -v
python -m pytest tests/test_activations.py -v
python -m pytest tests/test_loss.py -v
python -m pytest tests/test_sequential.py -v
```

### Run Specific Test Class
```bash
python -m pytest tests/test_tensor_ops.py::TestArithmeticOps -v
python -m pytest tests/test_activations.py::TestReLU -v
```

### Run Specific Test
```bash
python -m pytest tests/test_tensor_ops.py::TestArithmeticOps::test_add -v
```

### Run with Coverage
```bash
python -m pytest tests/ --cov=scratch --cov-report=html
```

## Test Statistics

- **Total Tests:** 93
- **Test Files:** 5
- **Test Classes:** 24
- **Coverage Areas:**
  - Tensor operations (forward & backward)
  - Neural network layers
  - Activation functions
  - Loss functions
  - Model composition

## Requirements

The tests require:
- `numpy` - For numerical operations
- `pytest` (optional but recommended) - For running tests
- `loguru` - For logging (used in loss.py)

## Test Design Principles

1. **Comprehensive Coverage:** Every operation tests both forward pass and backward pass (gradients)
2. **Edge Cases:** Tests include edge cases like zeros, negatives, large values
3. **Numerical Stability:** Tests verify operations don't produce NaN or Inf
4. **Shape Correctness:** All tests verify output shapes match expectations
5. **Gradient Correctness:** Backward passes are verified against expected gradients
6. **Integration Tests:** Higher-level tests verify components work together

## Adding New Tests

When adding new functionality to the library:

1. Add corresponding test class in appropriate file
2. Test forward pass computation
3. Test backward pass gradients
4. Test edge cases and error conditions
5. Add integration tests if needed
6. Update this README with new test coverage

## Notes

- All tests use `np.testing.assert_array_almost_equal()` for floating-point comparisons
- Gradients are verified either analytically or by checking expected properties
- Tests use small, hand-computable examples for easy verification
- Random seeds should be set when using random data for reproducibility

