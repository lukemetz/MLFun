import theano.tensor as T
import numpy as np

from cuboid.bricks import Flattener, FilterPool, Dropout, BatchNormalization
from cuboid.bricks import Convolutional, LeakyRectifier, BrickSequence
from blocks.bricks.conv import MaxPooling
from blocks.bricks import Linear, Softmax

from blocks.initialization import IsotropicGaussian, Constant

from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate

def conv3(num_filters):
    return [Convolutional(filter_size=(3, 3),
            num_filters=num_filters,
            weights_init=IsotropicGaussian(std=0.05),
            biases_init=Constant(0),
            use_bias=True,
            border_mode="same",
            step=(1,1)),

            LeakyRectifier(0.01)]

def max_pool():
    return MaxPooling(pooling_size=(2, 2),
            step=(2, 2))

def linear(n):
    return Linear(output_dim=n,
            weights_init=IsotropicGaussian(std=0.01),
            biases_init=Constant(0),
            use_bias=True)

class ModelHelper():
    def __init__(self):
        self.X = T.tensor4("features")

        seq = BrickSequence(input_dim = (3, 32, 32), bricks=[
            conv3(10),
            conv3(10),
            max_pool(),
            #conv3(10),
            #conv3(10),
            max_pool(),
            #conv3(10),
            #conv3(10),
            Flattener(),
            linear(10),
            Softmax()
            ])

        seq.initialize()

        self.pred = seq.apply(self.X)
        self.Y = T.imatrix("targets")

        self.cost = CategoricalCrossEntropy().apply(self.Y.flatten(), self.pred)
        self.cost.name = "cost"

        self.accur = 1.0 - MisclassificationRate().apply(self.Y.flatten(), self.pred)
        self.accur.name = "accur"
