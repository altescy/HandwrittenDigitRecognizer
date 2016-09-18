# -*- coding: utf-8 -*-

from chainer import Chain
import chainer.functions as F
import chainer.links as L

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10),
        )
    
    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)), train=False)
        h2 = F.dropout(F.relu(self.l2(h1)), train=False)
        y = self.l3(h2)

        return y

class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            l1=L.Linear(800, 500),
            l2=L.Linear(500, 10)
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.dropout(F.relu(self.l1(h2)), train=True)
        y = self.l2(h3)
        return y
