# Packages
import math
import random
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
        #return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad # Because it is a + operation
            other.grad += 1.0 * out.grad # Because it is a + operation
            # '+=' accumulates gradients to account for multivariate chain rule
        out._backward = _backward # We want to store the function, not run it

        return out

    def __radd__(self, other): # other + self
        return self + other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad # Because it is a * operation
            other.grad += self.data * out.grad # Because it is a * operation
            # '+=' accumulates gradients to account for multivariate chain rule
        out._backward = _backward # We want to store the function, not run it

        return out

    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other*(self.data**(other-1))) * out.grad
            # '+=' accumulates gradients to account for multivariate chain rule
        out._backward = _backward # We want to store the function, not run it
        return out

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh') # Tupple of one object self

        def _backward():
            self.grad += (1 - t**2) * out.grad
            # '+=' accumulates gradients to account for multivariate chain rule
        out._backward = _backward # We want to store the function, not run it
        return out

    def relu(self):
        out = Value(max(0, self.data), (self, ), 'relu')

        def _backward():
            self.grad = (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        out = Value((1 / (1 + math.exp(-self.data))), (self, ), 'sigmoid')

        def _backward():
            self.grad = (math.exp(-self.data) / ((math.exp(-self.data) + 1)**2)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp') # Tupple of one object self

        def _backward():
            self.grad += out.data * out.grad
            # '+=' accumulates gradients to account for multivariate chain rule
        out._backward = _backward # We want to store the function, not run it
        return out

    def log(self): # log(self)
        out = Value(math.log(self.data), (self, ), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Topological sort to order all nodes so that the edges go one way only (for example, from left to right)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Backpropagation through reversed topological order
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()