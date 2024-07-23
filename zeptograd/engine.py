from enum import Enum
import math

class Scalar:
    class Operator(Enum):
        ADD = "+"
        MUL = "*"
        POW = "**"
        RELU = "ReLU"

    def __init__(self, data, _op=None, _children=()):
        self.data = data
        self.grad = 0
        self._op = _op
        self._children = set(_children)

        def _backward():
            pass
        self._backward = _backward

    def __repr__(self):
        return f"Scalar(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, _op=Scalar.Operator.ADD, _children=(self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, _op=Scalar.Operator.MUL, _children=(self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data ** other.data, _op=Scalar.Operator.POW, _children=(self, other))
        def _backward():
            if self.data > 0:
                self.grad += (other.data) * (self.data ** (other.data - 1)) * out.grad
                other.grad += (self.data ** other.data) * (math.log(self.data)) * out.grad
            else:
                TODO: An opportunity to leverage complex analysis.
                self.grad += (other.data) * (self.data ** (other.data - 1)) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Scalar(0 if self.data < 0 else self.data, _op=Scalar.Operator.RELU, _children=(self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __radd__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other + self
    
    def __rmul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other * self
    
    def __sub__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return self + (-other)
    
    def __rsub__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other + (-self)
    
    def __rpow__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other ** self
    
    def __truediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return self * (other ** (-1))
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        return other * (self ** (-1))
    
    def __neg__(self):
        return self * -1