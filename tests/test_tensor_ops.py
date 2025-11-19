import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import scratch module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scratch.tensor import Tensor


class TestTensorBasics(unittest.TestCase):
    """Test basic Tensor initialization and properties"""
    
    def test_tensor_creation(self):
        """Test creating tensors with different data types"""
        t1 = Tensor(5)
        self.assertEqual(t1.data.shape, ())
        self.assertEqual(t1.data, 5.0)
        
        t2 = Tensor([1, 2, 3])
        self.assertEqual(t2.data.shape, (3,))
        np.testing.assert_array_equal(t2.data, [1.0, 2.0, 3.0])
        
        t3 = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t3.data.shape, (2, 2))
        
    def test_requires_grad(self):
        """Test gradient requirements"""
        t1 = Tensor([1, 2, 3], requires_grad=True)
        self.assertIsNotNone(t1.grad)
        self.assertEqual(t1.grad.shape, t1.data.shape)
        
        t2 = Tensor([1, 2, 3], requires_grad=False)
        self.assertIsNone(t2.grad)


class TestArithmeticOps(unittest.TestCase):
    """Test arithmetic operations and their gradients"""
    
    def test_add(self):
        """Test addition operation"""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        
        np.testing.assert_array_almost_equal(c.data, [5, 7, 9])
        
        # Test gradients
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [1, 1, 1])
        np.testing.assert_array_almost_equal(b.grad, [1, 1, 1])
    
    def test_add_scalar(self):
        """Test addition with scalar"""
        a = Tensor([1, 2, 3], requires_grad=True)
        c = a + 5
        
        np.testing.assert_array_almost_equal(c.data, [6, 7, 8])
        
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [1, 1, 1])
    
    def test_radd(self):
        """Test right addition"""
        a = Tensor([1, 2, 3], requires_grad=True)
        c = 5 + a
        
        np.testing.assert_array_almost_equal(c.data, [6, 7, 8])
    
    def test_sub(self):
        """Test subtraction operation"""
        a = Tensor([5, 7, 9], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a - b
        
        np.testing.assert_array_almost_equal(c.data, [4, 5, 6])
        
        # Test gradients
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [1, 1, 1])
        np.testing.assert_array_almost_equal(b.grad, [-1, -1, -1])
    
    def test_rsub(self):
        """Test right subtraction"""
        a = Tensor([1, 2, 3], requires_grad=True)
        c = 10 - a
        
        np.testing.assert_array_almost_equal(c.data, [9, 8, 7])
        
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [-1, -1, -1])
    
    def test_mul(self):
        """Test multiplication operation"""
        a = Tensor([2, 3, 4], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        c = a * b
        
        np.testing.assert_array_almost_equal(c.data, [2, 6, 12])
        
        # Test gradients
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [1, 2, 3])
        np.testing.assert_array_almost_equal(b.grad, [2, 3, 4])
    
    def test_mul_scalar(self):
        """Test multiplication with scalar"""
        a = Tensor([1, 2, 3], requires_grad=True)
        c = a * 2
        
        np.testing.assert_array_almost_equal(c.data, [2, 4, 6])
        
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [2, 2, 2])
    
    def test_rmul(self):
        """Test right multiplication"""
        a = Tensor([1, 2, 3], requires_grad=True)
        c = 2 * a
        
        np.testing.assert_array_almost_equal(c.data, [2, 4, 6])
    
    def test_div(self):
        """Test division operation"""
        a = Tensor([6, 8, 10], requires_grad=True)
        b = Tensor([2, 4, 5], requires_grad=True)
        c = a / b
        
        np.testing.assert_array_almost_equal(c.data, [3, 2, 2])
        
        # Test gradients
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [0.5, 0.25, 0.2])
        np.testing.assert_array_almost_equal(b.grad, [-1.5, -0.5, -0.4])
    
    def test_pow(self):
        """Test power operation"""
        a = Tensor([2, 3, 4], requires_grad=True)
        c = a ** 2
        
        np.testing.assert_array_almost_equal(c.data, [4, 9, 16])
        
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [4, 6, 8])
    
    def test_neg(self):
        """Test negation operation"""
        a = Tensor([1, -2, 3], requires_grad=True)
        c = -a
        
        np.testing.assert_array_almost_equal(c.data, [-1, 2, -3])
        
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, [-1, -1, -1])


class TestMatMulOp(unittest.TestCase):
    """Test matrix multiplication operation"""
    
    def test_matmul(self):
        """Test matrix multiplication"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b
        
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradients
        c.backward()
        expected_grad_a = np.array([[11, 15], [11, 15]])
        expected_grad_b = np.array([[4, 4], [6, 6]])
        np.testing.assert_array_almost_equal(a.grad, expected_grad_a)
        np.testing.assert_array_almost_equal(b.grad, expected_grad_b)
    
    def test_matmul_vector(self):
        """Test matrix-vector multiplication"""
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = Tensor([[7], [8], [9]], requires_grad=True)
        c = a @ b
        
        expected = np.array([[50], [122]])
        np.testing.assert_array_almost_equal(c.data, expected)


class TestUnaryOps(unittest.TestCase):
    """Test unary operations and their gradients"""
    
    def test_tanh(self):
        """Test tanh activation"""
        a = Tensor([0, 1, -1], requires_grad=True)
        c = a.tanh()
        
        expected = np.tanh([0, 1, -1])
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = 1 - np.tanh([0, 1, -1]) ** 2
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_relu(self):
        """Test ReLU activation"""
        a = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        c = a.relu()
        
        expected = [0, 0, 0, 1, 2]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = [0, 0, 0, 1, 1]
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_exp(self):
        """Test exponential function"""
        a = Tensor([0, 1, 2], requires_grad=True)
        c = a.exp()
        
        expected = np.exp([0, 1, 2])
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        np.testing.assert_array_almost_equal(a.grad, expected)
    
    def test_log(self):
        """Test logarithm function"""
        a = Tensor([1, 2, 3], requires_grad=True)
        c = a.log()
        
        expected = np.log([1, 2, 3])
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = 1.0 / np.array([1, 2, 3])
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_sigmoid(self):
        """Test sigmoid activation"""
        a = Tensor([0, 1, -1], requires_grad=True)
        c = a.sigmoid()
        
        expected = 1 / (1 + np.exp(-np.array([0, 1, -1])))
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        s = expected
        expected_grad = s * (1 - s)
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_leaky_relu(self):
        """Test Leaky ReLU activation"""
        a = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        c = a.leaky_relu(alpha=0.1)
        
        expected = [-0.2, -0.1, 0, 1, 2]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = [0.1, 0.1, 0.1, 1, 1]
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_clamp(self):
        """Test clamp operation"""
        a = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        c = a.clamp(-1, 1)
        
        expected = [-1, -1, 0, 1, 1]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = [0, 1, 1, 1, 0]
        np.testing.assert_array_almost_equal(a.grad, expected_grad)


class TestReductionOps(unittest.TestCase):
    """Test reduction operations"""
    
    def test_sum_all(self):
        """Test sum over all elements"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        c = a.sum()
        
        self.assertEqual(c.data, 10)
        
        # Test gradient
        c.backward()
        expected_grad = np.ones((2, 2))
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_sum_axis(self):
        """Test sum along specific axis"""
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        c = a.sum(axis=0)
        
        expected = [5, 7, 9]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = np.ones((2, 3))
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_sum_keepdims(self):
        """Test sum with keepdims"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        c = a.sum(axis=1, keepdims=True)
        
        expected = [[3], [7]]
        np.testing.assert_array_almost_equal(c.data, expected)
    
    def test_mean_all(self):
        """Test mean over all elements"""
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        c = a.mean()
        
        self.assertAlmostEqual(c.data, 2.5)
        
        # Test gradient
        c.backward()
        expected_grad = np.ones((2, 2)) * 0.25
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_mean_axis(self):
        """Test mean along specific axis"""
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        c = a.mean(axis=0)
        
        expected = [2.5, 3.5, 4.5]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = np.ones((2, 3)) * 0.5
        np.testing.assert_array_almost_equal(a.grad, expected_grad)


class TestIndexing(unittest.TestCase):
    """Test indexing operations"""
    
    def test_getitem_single(self):
        """Test single element indexing"""
        a = Tensor([1, 2, 3, 4], requires_grad=True)
        c = a[1]
        
        self.assertEqual(c.data, 2)
        
        # Test gradient
        c.backward()
        expected_grad = [0, 1, 0, 0]
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_getitem_slice(self):
        """Test slice indexing"""
        a = Tensor([1, 2, 3, 4, 5], requires_grad=True)
        c = a[1:4]
        
        expected = [2, 3, 4]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradient
        c.backward()
        expected_grad = [0, 1, 1, 1, 0]
        np.testing.assert_array_almost_equal(a.grad, expected_grad)
    
    def test_getitem_2d(self):
        """Test 2D indexing"""
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        c = a[0]
        
        expected = [1, 2, 3]
        np.testing.assert_array_almost_equal(c.data, expected)


class TestSoftmax(unittest.TestCase):
    """Test softmax and log_softmax operations"""
    
    def test_softmax(self):
        """Test softmax operation"""
        a = Tensor([[1, 2, 3]], requires_grad=True)
        c = a.softmax()
        
        # Verify output sums to 1
        self.assertAlmostEqual(c.data.sum(), 1.0)
        
        # Verify correct values
        expected = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()
        np.testing.assert_array_almost_equal(c.data[0], expected)
    
    def test_log_softmax(self):
        """Test log_softmax operation"""
        a = Tensor([[1, 2, 3]], requires_grad=True)
        c = a.log_softmax()
        
        # Verify correct values
        shifted = np.array([1, 2, 3]) - 3
        exps = np.exp(shifted)
        expected = shifted - np.log(exps.sum())
        np.testing.assert_array_almost_equal(c.data[0], expected)
        
        # Test gradient
        c.backward()
        # Gradient should be computed correctly
        self.assertEqual(a.grad.shape, (1, 3))


class TestBroadcasting(unittest.TestCase):
    """Test broadcasting in operations"""
    
    def test_broadcast_add(self):
        """Test addition with broadcasting"""
        a = Tensor([[1, 2, 3]], requires_grad=True)  # (1, 3)
        b = Tensor([[1], [2], [3]], requires_grad=True)  # (3, 1)
        c = a + b
        
        expected = [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        # Test gradients are properly unbroadcasted
        c.backward()
        self.assertEqual(a.grad.shape, (1, 3))
        self.assertEqual(b.grad.shape, (3, 1))
    
    def test_broadcast_mul(self):
        """Test multiplication with broadcasting"""
        a = Tensor([[2, 3, 4]], requires_grad=True)  # (1, 3)
        b = Tensor([[1], [2]], requires_grad=True)  # (2, 1)
        c = a * b
        
        expected = [[2, 3, 4], [4, 6, 8]]
        np.testing.assert_array_almost_equal(c.data, expected)
        
        c.backward()
        self.assertEqual(a.grad.shape, (1, 3))
        self.assertEqual(b.grad.shape, (2, 1))


class TestComplexComputations(unittest.TestCase):
    """Test complex computation graphs"""
    
    def test_simple_network(self):
        """Test a simple computation resembling a neural network"""
        # Input
        x = Tensor([[1, 2]], requires_grad=True)
        
        # Weights
        w1 = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
        b1 = Tensor([[0.1, 0.1]], requires_grad=False)
        
        # Forward pass
        h = (x @ w1 + b1).tanh()
        w2 = Tensor([[1], [1]], requires_grad=True)
        out = h @ w2
        
        # Backward pass
        out.backward()
        
        # Check all gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w1.grad)
        self.assertIsNotNone(w2.grad)
        self.assertIsNone(b1.grad)  # requires_grad=False
    
    def test_multiop_gradient(self):
        """Test gradients through multiple operations"""
        a = Tensor([2], requires_grad=True)
        b = Tensor([3], requires_grad=True)
        
        c = a * b  # 6
        d = c + a  # 8
        e = d ** 2  # 64
        
        e.backward()
        
        # de/da = de/dd * dd/da = 2*d * (b + 1) = 16 * 4 = 64
        # Actually: de/dd = 2*d = 16, dd/da = 1 + b = 4, dd/dc = 1, dc/da = b = 3
        # So de/da = de/dd * (dd/da + dd/dc * dc/da) = 16 * (1 + 3) = 64
        self.assertAlmostEqual(a.grad[0], 64)
        
        # de/db = de/dd * dd/dc * dc/db = 16 * 1 * 2 = 32
        self.assertAlmostEqual(b.grad[0], 32)


if __name__ == '__main__':
    unittest.main()

