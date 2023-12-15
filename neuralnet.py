import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random

from graphviz import Digraph, Source

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot


  
class Value:
    def __init__(self, data, _children=(), _op='', label='data'):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda : None
        #print(f"{data, _children, _op, label}")

    def __repr__(self):
        return f"{self.label} = {self.data}"
    
    def backward (self):
        topo = []
        visited = set()
        def build_topo (v):  
            if v not in visited:
                visited.add (v)
                for child in v._prev:
                    build_topo(child)
                topo.append (v)
        
        build_topo (self)
        self.grad = 1
        for node in reversed(topo):
            #print(f"Propagating {node} {node.grad}")
            #pdb.set_trace ()
            node._backward ()
         
    def __add__ (self, other):
        
        other = other if isinstance(other, Value) else Value(other, label=f'{other}')
        out = Value(self.data + other.data, (self, other), '+')

        def _backward ():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__ (self, other):
        return self + other
    
    def __mul__ (self, other):
        other = other if isinstance(other, Value) else Value(other, label=f'{other}')
        out = Value(self.data * other.data, (self, other), '*')
        def _backward ():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__ (self, other):
        return self * other
    
    def __truediv__ (self, other):
       return self * other**-1
    
    def __pow__ (self, other):
        #print(f"Am I here ever {self} {other}")
        assert isinstance (other, (int, float)), "Only supports integer or float for now"
        out = Value (self.data ** other, (self,), f'** {other}')
        def _backward ():
            #print ("Propagating -- Pow -- ", other, self)
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def __neg__ (self):
        return self * (-1)
    
    def __sub__ (self, other):
        return self + (-other)
    
    def exp (self):
        x = self.data
        out = Value (math.exp(x), (self, ), 'exp')

        def _backward ():
            #print ("Propagating -- Exp -- ", self)
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def tanh (self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - (t**2)) * out.grad

        out._backward = _backward
        return out 


class Neuron:
    def __init__ (self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__ (self, x):
        assert (len(x) == len(self.w)), "Diomensions need to match"
        act = sum((xi*wi for xi, wi in zip(x, self.w)), self.b)
        out = act.tanh()
        return out
    
    def parameters (self):
        return self.w + [self.b]

class Layer:
    def __init__ (self, nin, nout):
        self._neurons = [Neuron(nin) for _ in range(nout)]
            
    def __call__ (self, x):
        outs = [neuron(x) for neuron in self._neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters (self):
        return [p for neuron in self._neurons for p in neuron.parameters()]
        
class MLP:
    def __init__ (self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__ (self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]


for i in range(100):
    #Forward pass
    ypred = [n(x) for x in xs]
    loss =sum([(yout-ygt)**2 for ygt, yout in zip (ys, ypred)])
    
    #backward prop
    for p in n.parameters ():
        p.grad  = 0.0
    loss.backward()

    #update
    for p in n.parameters ():
        p.data += -0.5 * p.grad
        
    print (i, f"Loss is {loss.data}")
    print(ypred)
#print(n(x))

#o = n.tanh(); o.label = 'o'
#o.backward ()


#draw_dot(n(x)).render('round-table.gv', view=True)

"""
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
#d = e + c; d.label='d'
d = e + c; d.label='d'
f = Value (-2.0, label='f')
g = d * f; g.label='g'
L = g.tanh(); L.label = 'L'
f._backward ()
d._backward ()
e._backward ()
c._backward ()
b._backward ()
a._backward ()
"""   

