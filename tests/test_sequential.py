import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratch.tensor import Tensor
from scratch.nn.linear import Linear
from scratch.nn.activations import ReLU, Tanh, Sigmoid
from scratch.nn.sequential import Sequential


class TestSequential(unittest.TestCase):
    """Test Sequential model stacking"""
    
    def test_initialization(self):
        """Test Sequential initialization"""
        layer1 = Linear(10, 5)
        layer2 = ReLU()
        layer3 = Linear(5, 2)
        
        model = Sequential(layer1, layer2, layer3)
        
        self.assertEqual(len(model.layers), 3)
        self.assertIs(model.layers[0], layer1)
        self.assertIs(model.layers[1], layer2)
        self.assertIs(model.layers[2], layer3)
    
    def test_forward(self):
        """Test forward pass through Sequential"""
        # Create a simple network
        layer1 = Linear(3, 2)
        relu = ReLU()
        layer2 = Linear(2, 1)
        
        model = Sequential(layer1, relu, layer2)
        
        # Input
        x = Tensor([[1, 2, 3]], requires_grad=True)
        
        # Forward pass
        out = model(x)
        
        # Output should have correct shape
        self.assertEqual(out.data.shape, (1, 1))
    
    def test_backward(self):
        """Test backward pass through Sequential"""
        layer1 = Linear(3, 2)
        relu = ReLU()
        layer2 = Linear(2, 1)
        
        model = Sequential(layer1, relu, layer2)
        
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = model(x)
        out.backward()
        
        # Check gradients exist for all parameters
        self.assertIsNotNone(layer1.W.grad)
        self.assertIsNotNone(layer1.b.grad)
        self.assertIsNotNone(layer2.W.grad)
        self.assertIsNotNone(layer2.b.grad)
        self.assertIsNotNone(x.grad)
    
    def test_parameters(self):
        """Test collecting parameters from Sequential"""
        layer1 = Linear(3, 2)
        relu = ReLU()
        layer2 = Linear(2, 1)
        
        model = Sequential(layer1, relu, layer2)
        params = model.parameters()
        
        # Should have 4 parameters: W1, b1, W2, b2
        self.assertEqual(len(params), 4)
        
        # Check they are the correct parameters
        self.assertIn(layer1.W, params)
        self.assertIn(layer1.b, params)
        self.assertIn(layer2.W, params)
        self.assertIn(layer2.b, params)
    
    def test_parameters_with_activation_only(self):
        """Test parameters when layer has no parameters"""
        relu1 = ReLU()
        relu2 = ReLU()
        
        model = Sequential(relu1, relu2)
        params = model.parameters()
        
        # Should have no parameters
        self.assertEqual(len(params), 0)
    
    def test_deep_network(self):
        """Test deeper network"""
        model = Sequential(
            Linear(10, 8),
            ReLU(),
            Linear(8, 6),
            ReLU(),
            Linear(6, 4),
            ReLU(),
            Linear(4, 2)
        )
        
        x = Tensor(np.random.randn(5, 10), requires_grad=True)
        out = model(x)
        
        # Should produce correct output shape
        self.assertEqual(out.data.shape, (5, 2))
        
        # Should be able to backward
        out.backward()
        
        # All linear layers should have gradients
        params = model.parameters()
        for param in params:
            self.assertIsNotNone(param.grad)
    
    def test_different_activations(self):
        """Test Sequential with different activation functions"""
        model = Sequential(
            Linear(3, 4),
            Tanh(),
            Linear(4, 3),
            Sigmoid(),
            Linear(3, 2)
        )
        
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = model(x)
        
        self.assertEqual(out.data.shape, (1, 2))
    
    def test_batch_processing(self):
        """Test Sequential with batch input"""
        model = Sequential(
            Linear(5, 3),
            ReLU(),
            Linear(3, 1)
        )
        
        # Batch of 10 samples
        x = Tensor(np.random.randn(10, 5), requires_grad=True)
        out = model(x)
        
        # Should maintain batch dimension
        self.assertEqual(out.data.shape, (10, 1))
    
    def test_single_layer(self):
        """Test Sequential with single layer"""
        layer = Linear(3, 2)
        model = Sequential(layer)
        
        x = Tensor([[1, 2, 3]], requires_grad=True)
        out = model(x)
        
        self.assertEqual(out.data.shape, (1, 2))
    
    def test_empty_sequential_parameters(self):
        """Test parameters method doesn't fail with mixed layers"""
        class CustomLayer:
            """Layer without parameters method"""
            def __call__(self, x):
                return x * 2
        
        layer1 = Linear(3, 2)
        custom = CustomLayer()
        layer2 = Linear(2, 1)
        
        model = Sequential(layer1, custom, layer2)
        params = model.parameters()
        
        # Should have parameters from Linear layers only
        self.assertEqual(len(params), 4)


class TestSequentialIntegration(unittest.TestCase):
    """Integration tests for Sequential with complete workflows"""
    
    def test_simple_training_step(self):
        """Test a complete forward-backward pass"""
        # Create model
        model = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 1)
        )
        
        # Input and target
        x = Tensor([[1, 2]], requires_grad=True)
        
        # Forward pass
        pred = model(x)
        
        # Simple loss (MSE with target 0)
        target = Tensor([[0]], requires_grad=False)
        loss = ((pred - target) ** 2).mean()
        
        # Backward pass
        loss.backward()
        
        # All parameters should have gradients
        params = model.parameters()
        for param in params:
            self.assertIsNotNone(param.grad)
            # Gradients should be non-zero (with high probability)
            self.assertTrue(np.any(param.grad != 0))
    
    def test_regression_network(self):
        """Test network for regression task"""
        model = Sequential(
            Linear(5, 10),
            Tanh(),
            Linear(10, 10),
            Tanh(),
            Linear(10, 1)
        )
        
        # Generate some data
        x = Tensor(np.random.randn(32, 5), requires_grad=True)
        
        # Forward pass
        pred = model(x)
        
        # Should produce continuous values
        self.assertEqual(pred.data.shape, (32, 1))
        
        # Should be able to compute loss
        target = Tensor(np.random.randn(32, 1), requires_grad=False)
        from scratch.nn.loss import Loss
        loss = Loss.mse(pred, target)
        
        # Should be able to backward
        loss.backward()
        self.assertIsNotNone(loss.data)
    
    def test_classification_network(self):
        """Test network for classification task"""
        model = Sequential(
            Linear(10, 8),
            ReLU(),
            Linear(8, 3)  # 3 classes
        )
        
        # Input
        x = Tensor(np.random.randn(1, 10), requires_grad=True)
        
        # Forward pass (logits)
        logits = model(x)
        
        self.assertEqual(logits.data.shape, (1, 3))
        
        # Compute loss
        from scratch.nn.loss import Loss
        loss = Loss.cross_entropy(logits, 0)
        
        # Backward
        loss.backward()
        
        self.assertIsNotNone(loss.data)


if __name__ == '__main__':
    unittest.main()

