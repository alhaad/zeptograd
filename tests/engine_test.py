import math
import unittest
from zeptograd.engine import Scalar

class EngineTest(unittest.TestCase):
    def test_creation(self):
        x = Scalar(10)
        self.assertEqual(x.data, 10)
        self.assertEqual(x.grad, 0)
    
    def test_add(self):
        a = Scalar(10)
        b = Scalar(20)

        c = a + b
        self.assertEqual(c.data, 30)
        self.assertEqual(c.grad, 0)

        c.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_mul(self):
        a = Scalar(10)
        b = Scalar(20)

        c = a * b
        self.assertEqual(c.data, 200)
        self.assertEqual(c.grad, 0)

        c.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, 20)
        self.assertEqual(b.grad, 10)

    
    def test_mul_square(self):
        a = Scalar(10)

        b = a * a
        self.assertEqual(b.data, 100)
        self.assertEqual(b.grad, 0)

        b.backward()
        self.assertEqual(b.grad, 1)
        self.assertEqual(a.grad, 20)

    def test_sub(self):
        a = Scalar(10)
        b = Scalar(20)

        c = a - b
        self.assertEqual(c.data, -10)
        self.assertEqual(c.grad, 0)

        c.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, -1)
    
    def test_pow(self):
        a = Scalar(2)
        b = Scalar(3)

        c = a ** b
        self.assertEqual(c.data, 8)
        self.assertEqual(c.grad, 0)

        c.backward()
        self.assertEqual(c.grad, 1)
        self.assertEqual(a.grad, 12)
        self.assertEqual(b.grad, 8 * math.log(2))

    def test_rtruediv(self):
        a = Scalar(2)

        b = 4 / a
        self.assertEqual(b.data, 2)
        self.assertEqual(b.grad, 0)

        b.backward()
        self.assertEqual(b.grad, 1)
        self.assertEqual(a.grad, -1)

    def test_relu(self):
        a = Scalar(1)
        b = Scalar(-1)
        c = Scalar(0)

        a1 = a.relu()
        self.assertEqual(a1.data, 1)
        b1 = b.relu()
        self.assertEqual(b1.data, 0)
        c1 = c.relu()
        self.assertEqual(c1.data, 0)

        a1.backward()
        self.assertEqual(a.grad, 1)
        b1.backward()
        self.assertEqual(b.grad, 0)
        c1.backward()
        self.assertEqual(c.grad, 0)

    def test_karpathy(self):
        a = Scalar(-4.0)
        b = Scalar(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        self.assertAlmostEqual(g.data, 24.7041, places=3)

        g.backward()
        self.assertAlmostEqual(a.grad, 138.8338, places=3)
        self.assertAlmostEqual(b.grad, 645.5773, places=3)