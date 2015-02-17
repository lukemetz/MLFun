import theano
from theano import tensor as T
import blocks
import numpy as np

from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks import Linear, Rectifier, Softmax, Brick
from blocks.bricks.base import lazy, application
from blocks.utils import shared_floatx_zeros
from blocks import config

from blocks.bricks.conv import Convolutional, Flattener, MaxPooling

from blocks.initialization import IsotropicGaussian

from blocks.datasets.mnist import MNIST
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme

from blocks.algorithms import Momentum, GradientDescent, Scale, StepRule, AdaDelta

from blocks.main_loop import MainLoop
from blocks.extensions import Printing, Timing
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import DataStreamMonitoring

from cuboid.algorithms import AdaM
from cuboid.bricks import BatchNormalizationConv, BatchNormalization

X = T.matrix("features")

o = X.reshape((X.shape[0], 1, 28, 28))

l = Convolutional(filter_size=(5, 5),
        num_filters=32,
        num_channels=1,
        image_shape=(28,28),
        weights_init=IsotropicGaussian(std=0.01),
        biases_init=IsotropicGaussian(std=0.01, mean=1.0),
        use_bias=True,
        border_mode="valid",
        step=(1,1))
l.initialize()
o = l.apply(o)

l = BatchNormalizationConv(input_shape=l.get_dim("output"),
        B_init=IsotropicGaussian(std=0.01),
        Y_init=IsotropicGaussian(std=0.01))
l.initialize()
o = l.apply(o)

o = Rectifier().apply(o)


l = MaxPooling(pooling_size=(2, 2),
        step=(2, 2),
        input_dim=l.get_dim("output"))
l.initialize()
o = l.apply(o)

l = Convolutional(filter_size=(3, 3),
        num_filters=32,
        num_channels=l.get_dim("output")[0],
        image_shape=l.get_dim("output")[1:],
        weights_init=IsotropicGaussian(std=0.01),
        biases_init=IsotropicGaussian(std=0.01),
        use_bias=True,
        border_mode="valid",
        step=(1,1))
l.initialize()
o = l.apply(o)

l = BatchNormalizationConv(input_shape=l.get_dim("output"),
        B_init=IsotropicGaussian(std=0.01),
        Y_init=IsotropicGaussian(std=0.01))
l.initialize()
o = l.apply(o)

shape = np.prod(l.get_dim("output"))

o = Flattener().apply(o)

l = Linear(input_dim=shape,
        output_dim=200,
        weights_init=IsotropicGaussian(std=0.01),
        biases_init=IsotropicGaussian(std=0.01))
l.initialize()
o = l.apply(o)

l = BatchNormalization(input_dim=l.get_dim("output"),
        B_init=IsotropicGaussian(std=0.01),
        Y_init=IsotropicGaussian(std=0.01))
l.initialize()
o = l.apply(o)

o = Rectifier().apply(o)

l = Linear(input_dim=l.get_dim("output"),
        output_dim=10,
        weights_init=IsotropicGaussian(std=0.01),
        biases_init=IsotropicGaussian(std=0.01))
l.initialize()
o = l.apply(o)

o = Softmax().apply(o)

Y = T.imatrix(name="targets")

cost = CategoricalCrossEntropy().apply(Y.flatten(), o)
cost.name = "cost"

miss_class = 1.0 - MisclassificationRate().apply(Y.flatten(), o)
miss_class.name = "accuracy"

step_rule = AdaM()
algorithm = GradientDescent(cost=cost, step_rule=step_rule)

print "Loading data"
mnist_train = MNIST("train")
train_stream = DataStream(
    dataset=mnist_train,
    iteration_scheme= SequentialScheme(num_examples= mnist_train.num_examples, batch_size= 128))
    #iteration_scheme= SequentialScheme(num_examples= 1000, batch_size= 128))

mnist_test = MNIST("test")
test_stream = DataStream(
    dataset=mnist_test,
    iteration_scheme= SequentialScheme(num_examples= mnist_test.num_examples, batch_size= 1024))
    #iteration_scheme= SequentialScheme(num_examples= 1000, batch_size= 1024))

monitor_test = DataStreamMonitoring(variables=[cost, miss_class], data_stream=test_stream, prefix="test")
monitor_train = DataStreamMonitoring(variables=[cost, miss_class], data_stream=train_stream, prefix="train")


class ToFile(SimpleExtension):
    def __init__(self, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_every_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(ToFile, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.main_loop.log.to_dataframe().to_csv("tmp.csv")


print "Making main loop"
main_loop = MainLoop(
        model=Y,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[monitor_test, monitor_train, Timing(), Printing(), ToFile()]
        )

print "Starting main"
main_loop.run()

