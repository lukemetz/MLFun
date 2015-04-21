from theano import tensor as T
import theano
import numpy as np

from fuel.datasets.cifar10 import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from blocks.bricks import MLP, Linear, Rectifier, Tanh, Sigmoid
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent
from blocks.model import Model

from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, Dump

from blocks.extensions.plot import Plot

from cuboid.algorithms import NAG, AdaM
from cuboid.bricks import Dropout
from blocks.main_loop import MainLoop

from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.filter import VariableFilter, get_brick
from cuboid.extensions import ExperimentSaver, EpochProgress
from model import VAModel

m = VAModel()

dataset = CIFAR10('train', sources=['features'])
test_dataset = CIFAR10('test', sources=['features'])

scheme = ShuffledScheme(dataset.num_examples, 128)
datastream = DataStream(dataset, iteration_scheme=scheme)

test_scheme = ShuffledScheme(test_dataset.num_examples, 128)
test_stream = DataStream(test_dataset, iteration_scheme=test_scheme)


cg = ComputationGraph([m.variational_cost])
algorithm = GradientDescent(
        cost=m.variational_cost, params=cg.parameters,
        step_rule=AdaM())
        #step_rule=NAG(lr=0.01, m=0.9))

main_loop = MainLoop(
        algorithm,
        datastream,
        model = Model(m.variational_cost),
        extensions=[
            Timing(),
            TrainingDataMonitoring(
                [m.variational_cost], prefix="train", after_epoch=True)
            , DataStreamMonitoring(
                [m.variational_cost],
                test_stream,
                prefix="test")
            , DataStreamMonitoring(
                [m.cost],
                test_stream,
                prefix="test")
            #, FinishAfter(after_n_epochs=10)
            , Printing()
            #, Dump("out.pkl")
            , ExperimentSaver("../VAOutCIFAR", ".")
            , EpochProgress(dataset.num_examples // 128)
            , Plot('cifar10',
                channels=[['train_variational_cost', 'test_variational_cost', 'test_cost']])
            ])
main_loop.run()

_func_noisy = theano.function([m.X], m.noisy)
_func_produced = theano.function([m.X], m.produced)

batch = test_stream.get_epoch_iterator().next()[0]
out_noise = _func_noisy(batch)
out_produced = _func_produced(batch)
import cv2
for k in range(10):
    print out_noise.shape
    img = np.reshape(out_noise[k, :], (28, 28))
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)

    img = np.reshape(out_produced[k, :], (28, 28))
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('masdf', img)

    cv2.waitKey(0)

