from theano import tensor as T
import theano
import numpy as np

from fuel.datasets.mnist import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from blocks.bricks import MLP, Linear, Rectifier, Tanh, Sigmoid, Sequence
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent
from blocks.model import Model

from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint

from blocks.extensions.plot import Plot

from cuboid.algorithms import NAG, AdaM
from cuboid.bricks import Dropout
from blocks.main_loop import MainLoop

from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.filter import VariableFilter, get_brick

class VAModel():
    def __init__(self):
        srng = MRG_RandomStreams(seed=123)

        X = T.matrix('features')
        self.X = X

        #drop = Dropout(p_drop=0.5)
        #o = drop.apply(X)
        o = X
        self.noisy = o

        #n_hidden = 64
        n_hidden = 128
        n_zs = 2
        self.n_zs = n_zs

        self.n_hidden = n_hidden

        l = Linear(input_dim=28*28, output_dim=n_hidden,
                weights_init=IsotropicGaussian(0.01),
                biases_init=Constant(0))
        l.initialize()
        o = l.apply(o)
        o = Tanh().apply(o)


        l = Linear(input_dim=n_hidden, output_dim=n_hidden,
                weights_init=IsotropicGaussian(0.01),
                biases_init=Constant(0))
        l.initialize()
        o = l.apply(o)
        o = Tanh().apply(o)

        l = Linear(input_dim=n_hidden, output_dim=n_zs,
                weights_init=IsotropicGaussian(.101),
                biases_init=Constant(0))
        l.initialize()
        mu_encoder = l.apply(o)

        l = Linear(input_dim=n_hidden, output_dim=n_zs,
                weights_init=IsotropicGaussian(0.1),
                biases_init=Constant(0))
        l.initialize()
        log_sigma_encoder = l.apply(o)

        eps = srng.normal(log_sigma_encoder.shape)

        z = eps * T.exp(log_sigma_encoder) + mu_encoder

        z_to_h1_decode = Linear(input_dim=n_zs, output_dim=n_hidden,
                weights_init=IsotropicGaussian(0.1),
                biases_init=Constant(0))
        z_to_h1_decode.initialize()

        h1_decode_to_h_decode = Linear(input_dim=n_hidden, output_dim=n_hidden,
                weights_init=IsotropicGaussian(0.01),
                biases_init=Constant(0))
        h1_decode_to_h_decode.initialize()

        #o = z_to_h_decode.apply(z)
        #h_decoder = Tanh().apply(o)

        h_decode_produce = Linear(input_dim=n_hidden, output_dim=28*28,
                weights_init=IsotropicGaussian(0.01),
                biases_init=Constant(0),
                name="linear4")
        h_decode_produce.initialize()
        #o = h_decode_produce.apply(h_decoder)

        #self.produced = Sigmoid().apply(o)

        seq = Sequence([z_to_h1_decode.apply, Tanh().apply, h1_decode_to_h_decode.apply, Tanh().apply, h_decode_produce.apply, Sigmoid().apply])
        seq.initialize()

        self.produced = seq.apply(z)

        self.cost = T.sum(T.sqr(self.produced - X))
        #self.cost = T.sum(T.nnet.binary_crossentropy(self.produced, X)) #T.sum(T.sqr(self.produced - X))
        self.cost.name = "cost"

        self.variational_cost = - 0.5 * T.sum(1 + 2*log_sigma_encoder - mu_encoder * mu_encoder\
                - T.exp(2 * log_sigma_encoder)) + self.cost
        self.variational_cost.name = "variational_cost"

        self.Z = T.matrix('z')
        self.sampled = seq.apply(self.Z)


        cg = ComputationGraph([self.variational_cost])
        bricks = [get_brick(var) for var
                in cg.variables + cg.scan_variables if get_brick(var)]
        for i, b in enumerate(bricks):
            b.name = b.name + "_" + str(i)


