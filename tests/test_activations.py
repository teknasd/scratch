import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratch.tensor import Tensor
from scratch.nn.activations import ReLU, LeakyReLU, Tanh, Sigmoid, Softmax, LogSoftmax


class TestReLU(unittest.TestCase):
    """Test ReLU activation"""
    
    def test_forward(self):
        """Test ReLU forward pass"""
        relu = ReLU()
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        out = relu(x)
        
        expected = [0, 0, 0, 1, 2]
        np.testing.assert_array_almost_equal(out.data, expected)
    
    def test_backward(self):
        """Test ReLU backward pass"""
        relu = ReLU()
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        out = relu(x)
        out.backward()
        
        expected_grad = [0, 0, 0, 1, 1]
        np.testing.assert_array_almost_equal(x.grad, expected_grad)
    
    def test_matrix(self):
        """Test ReLU with matrix input"""
        relu = ReLU()
        x = Tensor([[-1, 2], [3, -4]], requires_grad=True)
        out = relu(x)
        
        expected = [[0, 2], [3, 0]]
        np.testing.assert_array_almost_equal(out.data, expected)


class TestLeakyReLU(unittest.TestCase):
    """Test Leaky ReLU activation"""
    
    def test_forward_default_alpha(self):
        """Test Leaky ReLU forward with default alpha"""
        leaky_relu = LeakyReLU()
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        out = leaky_relu(x)
        
        expected = [-0.02, -0.01, 0, 1, 2]
        np.testing.assert_array_almost_equal(out.data, expected)
    
    def test_forward_custom_alpha(self):
        """Test Leaky ReLU forward with custom alpha"""
        leaky_relu = LeakyReLU(alpha=0.1)
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        out = leaky_relu(x)
        
        expected = [-0.2, -0.1, 0, 1, 2]
        np.testing.assert_array_almost_equal(out.data, expected)
    
    def test_backward(self):
        """Test Leaky ReLU backward pass"""
        leaky_relu = LeakyReLU(alpha=0.1)
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        out = leaky_relu(x)
        out.backward()
        
        expected_grad = [0.1, 0.1, 0.1, 1, 1]
        np.testing.assert_array_almost_equal(x.grad, expected_grad)


class TestTanh(unittest.TestCase):
    """Test Tanh activation"""
    
    def test_forward(self):
        """Test Tanh forward pass"""
        tanh = Tanh()
        x = Tensor([0, 1, -1], requires_grad=True)
        out = tanh(x)
        
        expected = np.tanh([0, 1, -1])
        np.testing.assert_array_almost_equal(out.data, expected)
    
    def test_backward(self):
        """Test Tanh backward pass"""
        tanh = Tanh()
        x = Tensor([0, 1, -1], requires_grad=True)
        out = tanh(x)
        out.backward()
        
        expected_grad = 1 - np.tanh([0, 1, -1]) ** 2
        np.testing.assert_array_almost_equal(x.grad, expected_grad)
    
    def test_range(self):
        """Test Tanh output is in range [-1, 1]"""
        tanh = Tanh()
        x = Tensor([-10, -5, 0, 5, 10], requires_grad=True)
        out = tanh(x)
        
        # All values should be in [-1, 1]
        self.assertTrue(np.all(out.data >= -1))
        self.assertTrue(np.all(out.data <= 1))


class TestSigmoid(unittest.TestCase):
    """Test Sigmoid activation"""
    
    def test_forward(self):
        """Test Sigmoid forward pass"""
        sigmoid = Sigmoid()
        x = Tensor([0, 1, -1], requires_grad=True)
        out = sigmoid(x)
        
        expected = 1 / (1 + np.exp(-np.array([0, 1, -1])))
        np.testing.assert_array_almost_equal(out.data, expected)
    
    def test_backward(self):
        """Test Sigmoid backward pass"""
        sigmoid = Sigmoid()
        x = Tensor([0, 1, -1], requires_grad=True)
        out = sigmoid(x)
        out.backward()
        
        s = 1 / (1 + np.exp(-np.array([0, 1, -1])))
        expected_grad = s * (1 - s)
        np.testing.assert_array_almost_equal(x.grad, expected_grad)
    
    def test_range(self):
        """Test Sigmoid output is in range [0, 1]"""
        sigmoid = Sigmoid()
        x = Tensor([-10, -5, 0, 5, 10], requires_grad=True)
        out = sigmoid(x)
        
        # All values should be in [0, 1]
        self.assertTrue(np.all(out.data >= 0))
        self.assertTrue(np.all(out.data <= 1))
    
    def test_zero_input(self):
        """Test Sigmoid at zero is 0.5"""
        sigmoid = Sigmoid()
        x = Tensor([0], requires_grad=True)
        out = sigmoid(x)
        
        self.assertAlmostEqual(out.data[0], 0.5)


class TestSoftmax(unittest.TestCase):
    """Test Softmax activation"""
    
    def test_forward(self):
        """Test Softmax forward pass"""
        softmax = Softmax()
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = softmax(x)
        
        # Output should sum to 1
        self.assertAlmostEqual(out.data.sum(), 1.0)
        
        # Verify correct computation
        expected = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
        np.testing.assert_array_almost_equal(out.data[0], expected)
    
    def test_batch(self):
        """Test Softmax with batch input"""
        softmax = Softmax()
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        out = softmax(x)
        
        # Each row should sum to 1
        row_sums = out.data.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])
    
    def test_numerical_stability(self):
        """Test Softmax with large values (numerical stability)"""
        softmax = Softmax()
        x = Tensor([[1000, 1001, 1002]], requires_grad=True)
        out = softmax(x)
        
        # Should not overflow
        self.assertFalse(np.any(np.isnan(out.data)))
        self.assertFalse(np.any(np.isinf(out.data)))
        
        # Should still sum to 1
        self.assertAlmostEqual(out.data.sum(), 1.0)


class TestLogSoftmax(unittest.TestCase):
    """Test LogSoftmax activation"""
    
    def test_forward(self):
        """Test LogSoftmax forward pass"""
        log_softmax = LogSoftmax()
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = log_softmax(x)
        
        # Verify correct computation
        shifted = np.array([1, 2, 3]) - 3
        exps = np.exp(shifted)
        expected = shifted - np.log(exps.sum())
        np.testing.assert_array_almost_equal(out.data[0], expected)
    
    def test_backward(self):
        """Test LogSoftmax backward pass"""
        log_softmax = LogSoftmax()
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = log_softmax(x)
        out.backward()
        
        # Gradient should exist and have correct shape
        self.assertEqual(x.grad.shape, (1, 3))
        
        # Gradient along batch dimension should sum to 0
        self.assertAlmostEqual(x.grad.sum(axis=1)[0], 0.0, places=6)
    
    def test_numerical_stability(self):
        """Test LogSoftmax with large values"""
        log_softmax = LogSoftmax()
        x = Tensor([[1000, 1001, 1002]], requires_grad=True)
        out = log_softmax(x)
        
        # Should not overflow
        self.assertFalse(np.any(np.isnan(out.data)))
        self.assertFalse(np.any(np.isinf(out.data)))
    
    def test_equivalence_with_log_of_softmax(self):
        """Test LogSoftmax is equivalent to log(softmax(x))"""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        
        # Method 1: log_softmax
        log_softmax = LogSoftmax()
        out1 = log_softmax(x)
        
        # Method 2: log(softmax(x))
        softmax = Softmax()
        x2 = Tensor(x.data.copy(), requires_grad=True)
        out2_data = np.log(softmax(x2).data)
        
        # Should be approximately equal
        np.testing.assert_array_almost_equal(out1.data, out2_data, decimal=6)


if __name__ == '__main__':
    unittest.main()

