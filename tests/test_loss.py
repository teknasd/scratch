import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratch.tensor import Tensor
from scratch.nn.loss import Loss


class TestMSELoss(unittest.TestCase):
    """Test Mean Squared Error loss"""
    
    def test_forward(self):
        """Test MSE forward pass"""
        pred = Tensor([1, 2, 3], requires_grad=True)
        target = Tensor([1, 1, 1], requires_grad=False)
        
        loss = Loss.mse(pred, target)
        
        # Expected: mean((1-1)^2 + (2-1)^2 + (3-1)^2) = mean(0, 1, 4) = 5/3
        expected = 5.0 / 3.0
        self.assertAlmostEqual(loss.data, expected)
    
    def test_backward(self):
        """Test MSE backward pass"""
        pred = Tensor([1, 2, 3], requires_grad=True)
        target = Tensor([1, 1, 1], requires_grad=False)
        
        loss = Loss.mse(pred, target)
        loss.backward()
        
        # Gradient of MSE: 2/n * (pred - target)
        expected_grad = 2.0 / 3.0 * np.array([0, 1, 2])
        np.testing.assert_array_almost_equal(pred.grad, expected_grad)
    
    def test_perfect_prediction(self):
        """Test MSE when prediction equals target"""
        pred = Tensor([1, 2, 3], requires_grad=True)
        target = Tensor([1, 2, 3], requires_grad=False)
        
        loss = Loss.mse(pred, target)
        
        self.assertAlmostEqual(loss.data, 0.0)
    
    def test_matrix_input(self):
        """Test MSE with matrix input"""
        pred = Tensor([[1, 2], [3, 4]], requires_grad=True)
        target = Tensor([[0, 0], [0, 0]], requires_grad=False)
        
        loss = Loss.mse(pred, target)
        
        # Expected: mean(1 + 4 + 9 + 16) / 4 = 30/4 = 7.5
        expected = 7.5
        self.assertAlmostEqual(loss.data, expected)


class TestCrossEntropyLoss(unittest.TestCase):
    """Test Cross Entropy loss"""
    
    def test_forward(self):
        """Test cross entropy forward pass"""
        # Logits for 3 classes
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        target = 0  # Correct class is 0
        
        loss = Loss.cross_entropy(logits, target)
        
        # Should be non-negative
        self.assertGreaterEqual(loss.data, 0)
        
        # Should be a scalar
        self.assertEqual(loss.data.shape, ())
    
    def test_backward(self):
        """Test cross entropy backward pass"""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        target = 0
        
        loss = Loss.cross_entropy(logits, target)
        loss.backward()
        
        # Gradient should exist
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, (1, 3))
    
    def test_correct_prediction(self):
        """Test loss when highest logit is correct class"""
        # Class 0 has highest logit
        logits = Tensor([[10.0, 1.0, 0.1]], requires_grad=True)
        target = 0
        
        loss = Loss.cross_entropy(logits, target)
        
        # Loss should be small (but not zero due to other logits)
        self.assertLess(loss.data, 1.0)
    
    def test_incorrect_prediction(self):
        """Test loss when highest logit is wrong class"""
        # Class 2 has highest logit
        logits = Tensor([[0.1, 0.2, 10.0]], requires_grad=True)
        target = 0
        
        loss = Loss.cross_entropy(logits, target)
        
        # Loss should be large
        self.assertGreater(loss.data, 5.0)
    
    def test_tensor_target(self):
        """Test cross entropy with Tensor target"""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        target = Tensor([0], requires_grad=False)
        
        loss = Loss.cross_entropy(logits, target)
        
        # Should work with Tensor target
        self.assertGreaterEqual(loss.data, 0)
    
    def test_different_classes(self):
        """Test cross entropy for different target classes"""
        logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        
        losses = []
        for target in [0, 1, 2]:
            logits_copy = Tensor(logits.data.copy(), requires_grad=True)
            loss = Loss.cross_entropy(logits_copy, target)
            losses.append(loss.data)
        
        # Loss for class 2 (highest logit) should be smallest
        self.assertEqual(losses.index(min(losses)), 2)
        
        # Loss for class 0 (lowest logit) should be largest
        self.assertEqual(losses.index(max(losses)), 0)


class TestBinaryCrossEntropyLoss(unittest.TestCase):
    """Test Binary Cross Entropy loss"""
    
    def test_forward(self):
        """Test BCE forward pass"""
        # Predictions after sigmoid (probabilities)
        pred = Tensor([0.9, 0.1, 0.8, 0.2], requires_grad=True)
        target = Tensor([1, 0, 1, 0], requires_grad=False)
        
        loss = Loss.binary_cross_entropy(pred, target)
        
        # Loss should be non-negative
        self.assertGreaterEqual(loss.data, 0)
    
    def test_backward(self):
        """Test BCE backward pass"""
        pred = Tensor([0.9, 0.1, 0.8], requires_grad=True)
        target = Tensor([1, 0, 1], requires_grad=False)
        
        loss = Loss.binary_cross_entropy(pred, target)
        loss.backward()
        
        # Gradient should exist
        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.shape, pred.data.shape)
    
    def test_perfect_prediction(self):
        """Test BCE with perfect predictions"""
        # Almost perfect predictions (can't be exactly 0 or 1 due to log)
        pred = Tensor([0.999, 0.001, 0.999], requires_grad=True)
        target = Tensor([1, 0, 1], requires_grad=False)
        
        loss = Loss.binary_cross_entropy(pred, target)
        
        # Loss should be very small
        self.assertLess(loss.data, 0.01)
    
    def test_wrong_prediction(self):
        """Test BCE with wrong predictions"""
        pred = Tensor([0.1, 0.9, 0.2], requires_grad=True)
        target = Tensor([1, 0, 1], requires_grad=False)
        
        loss = Loss.binary_cross_entropy(pred, target)
        
        # Loss should be large
        self.assertGreater(loss.data, 1.0)
    
    def test_numerical_stability(self):
        """Test BCE doesn't break with edge values"""
        # Test with values close to 0 and 1
        pred = Tensor([0.0001, 0.9999], requires_grad=True)
        target = Tensor([0, 1], requires_grad=False)
        
        loss = Loss.binary_cross_entropy(pred, target)
        
        # Should not be inf or nan
        self.assertFalse(np.isnan(loss.data))
        self.assertFalse(np.isinf(loss.data))
    
    def test_scalar_target(self):
        """Test BCE with scalar target"""
        pred = Tensor([0.8], requires_grad=True)
        target = 1  # Scalar
        
        loss = Loss.binary_cross_entropy(pred, target)
        
        # Should work with scalar target
        self.assertGreaterEqual(loss.data, 0)
    
    def test_matrix_input(self):
        """Test BCE with 2D input"""
        pred = Tensor([[0.9, 0.1], [0.8, 0.2]], requires_grad=True)
        target = Tensor([[1, 0], [1, 0]], requires_grad=False)
        
        loss = Loss.binary_cross_entropy(pred, target)
        
        # Loss should be scalar (mean over all elements)
        self.assertEqual(loss.data.shape, ())


class TestLossComparison(unittest.TestCase):
    """Test comparing different losses"""
    
    def test_mse_vs_cross_entropy_use_cases(self):
        """Demonstrate appropriate use cases for different losses"""
        # For regression: MSE is appropriate
        pred_regression = Tensor([1.5, 2.3, 3.1], requires_grad=True)
        target_regression = Tensor([1.0, 2.0, 3.0], requires_grad=False)
        mse_loss = Loss.mse(pred_regression, target_regression)
        
        # For classification: Cross-entropy is appropriate
        pred_classification = Tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        target_classification = 0
        ce_loss = Loss.cross_entropy(pred_classification, target_classification)
        
        # Both should compute without errors
        self.assertIsNotNone(mse_loss.data)
        self.assertIsNotNone(ce_loss.data)


if __name__ == '__main__':
    unittest.main()

