# Importing engine.py
import random
from micrograd_myattempt.engine import Value

class Neuron:

    def __init__(self, nin, non_linearity = 'relu'):
        # nin: number of inputs to the neuron
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # weight
        self.b = Value(random.uniform(-1,1)) # bias
        self.non_linearity = non_linearity

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # zip creates a new iterator over tuples
        # sum has a start parameter -- our bias

        # Introducing non linearity with
        assert self.non_linearity in ['relu', 'tanh', 'sigmoid'], "Only supporting relu/tanh/sigmoid non-linearity for now"
        out = (self.non_linearity == 'relu') * act.relu() # relu
        out += (self.non_linearity == 'tanh') * act.tanh() # tanh
        out += (self.non_linearity == 'sigmoid') * act.sigmoid() # sigmoid

        return out

    # Collecting parameters in one array
    def parameters(self):
        return self.w + [self.b] # concatenated

    def __repr__(self):
        return f"Neuron(non-lin={self.non_linearity}, params={len(self.parameters())})"

class Layer:

    def __init__(self, nin, nout, non_linearity = 'relu'):
    # nin: number of inputs to the layer
    # nout: number of neurons
        self.neurons = [Neuron(nin, non_linearity) for _ in range(nout)]
        self.non_linearity = non_linearity

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    # Layer parameters
    def parameters(self):
        #params = []
        #params.extend(n.parameters() for n in self.neurons)
        #return params
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"

class MLP(Module):

    def __init__(self, nin, nouts, non_lin):
    # nin: number of inputs to the layer
    # nouts: a list of nouts with the sizes of all the layers
    # non_lin: a list of non_linearity strings for all layers. Must have the same dimension as nouts
        assert (len(nouts) == len(non_lin)), "non_lin must be a list with the same size of nouts"
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], non_lin[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # MLP parameters
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP([{', '.join(str(layer) for layer in self.layers)}])"

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []