import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratch.tensor import Tensor
from scratch.nn.linear import Linear


class TestLinearLayer(unittest.TestCase):
    """Test Linear layer"""
    
    def test_initialization(self):
        """Test layer initialization"""
        layer = Linear(10, 5)
        
        # Check weight shape
        self.assertEqual(layer.W.data.shape, (10, 5))
        
        # Check bias shape
        self.assertEqual(layer.b.data.shape, (1, 5))
        
        # Check bias is initialized to zeros
        np.testing.assert_array_almost_equal(layer.b.data, np.zeros((1, 5)))
        
        # Check requires_grad is True
        self.assertTrue(layer.W.requires_grad)
        self.assertTrue(layer.b.requires_grad)
    
    def test_forward(self):
        """Test forward pass"""
        layer = Linear(3, 2)
        
        # Set known weights for testing
        layer.W = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        layer.b = Tensor([[0.1, 0.2]], requires_grad=True)
        
        # Input
        x = Tensor([[1, 1, 1]], requires_grad=True)
        
        # Forward pass
        out = layer(x)
        
        # Expected: [1, 1, 1] @ [[1,2], [3,4], [5,6]] + [0.1, 0.2]
        # = [9, 12] + [0.1, 0.2] = [9.1, 12.2]
        expected = [[9.1, 12.2]]
        np.testing.assert_array_almost_equal(out.data, expected)
    
    def test_backward(self):
        """Test backward pass"""
        layer = Linear(2, 1)
        
        # Set known weights
        layer.W = Tensor([[2], [3]], requires_grad=True)
        layer.b = Tensor([[1]], requires_grad=True)
        
        # Input
        x = Tensor([[1, 2]], requires_grad=True)
        
        # Forward and backward
        out = layer(x)
        out.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.W.grad)
        self.assertIsNotNone(layer.b.grad)
        
        # Verify gradient shapes
        self.assertEqual(layer.W.grad.shape, (2, 1))
        self.assertEqual(layer.b.grad.shape, (1, 1))
    
    def test_parameters(self):
        """Test parameters method"""
        layer = Linear(5, 3)
        params = layer.parameters()
        
        self.assertEqual(len(params), 2)
        self.assertIs(params[0], layer.W)
        self.assertIs(params[1], layer.b)
    
    def test_batch_input(self):
        """Test with batch input"""
        layer = Linear(3, 2)
        
        # Batch of 4 samples
        x = Tensor(np.random.randn(4, 3), requires_grad=True)
        
        out = layer(x)
        
        # Output should have batch dimension
        self.assertEqual(out.data.shape, (4, 2))
        
        # Test backward
        out.backward()
        self.assertEqual(x.grad.shape, (4, 3))
    
    def test_xavier_initialization_scale(self):
        """Test that weights are initialized with reasonable scale"""
        # With larger layer, check variance is approximately 1/in_features
        layer = Linear(100, 50)
        
        # Variance should be approximately 1/100 = 0.01
        var = np.var(layer.W.data)
        
        # Check it's in a reasonable range (not too strict due to randomness)
        self.assertLess(var, 0.02)
        self.assertGreater(var, 0.005)


if __name__ == '__main__':
    unittest.main()

