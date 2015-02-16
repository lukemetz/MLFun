import theano
from theano import tensor as T
import blocks
import numpy as np

from theano.tensor.nnet.conv import conv2d, ConvOp

from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks.conv import Convolutional, Flattener, MaxPooling

from blocks.initialization import IsotropicGaussian

from blocks.datasets.mnist import MNIST
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme

from blocks.algorithms import Momentum, GradientDescent, Scale, StepRule, AdaDelta

from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring

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

o = Rectifier().apply(o)

"""
l = Convolutional(filter_size=(3, 3),
        num_filters=32,
        num_channels=l.get_dim("output")[0],
        image_shape=l.get_dim("output")[1:],
        weights_init=IsotropicGaussian(std=0.01),
        use_bias=True,
        border_mode="valid",
        step=(2,2))
o = l.apply(o)
"""
l = MaxPooling(pooling_size=(2, 2),
        step=(2, 2),
        input_dim=l.get_dim("output"))
l.initialize()
o = l.apply(o)

#o = Rectifier().apply(o)

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

shape = np.prod(l.get_dim("output"))

o = Flattener().apply(o)

l = Linear(input_dim=shape,
        output_dim=200,
        weights_init=IsotropicGaussian(std=0.01),
        biases_init=IsotropicGaussian(std=0.01))
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


class AdaM(StepRule):
    def __init__(self, alpha=0.002, b1=0.1, b2=0.001, eps=1e-8):
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.eps=eps

    def compute_step(self, p, g):
        b1 = self.b1
        b2 = self.b2
        eps = self.eps
        alpha = self.alpha

        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        t = theano.shared(np.array(0.0, dtype=theano.config.floatX))

        t_new = t + 1.0
        m_new = b1 * g + (1.0 - b1) * m
        v_new = b2 * g * g + (1 - b2) * v

        m_hat = m_new / (1.0 - T.pow((1.0 - b1), t_new))
        v_hat = v_new / (1.0 - T.pow((1.0 - b2), t_new))

        step = p - alpha * m_hat / (T.sqrt(v_hat) + eps)

        updates = [(m, m_new), (v, v_new), (t, t_new)]

        return step, updates
#step_rule = AdaDelta()
step_rule = Momentum(momentum=.9, learning_rate=0.01)
#step_rule = AdaM()
algorithm = GradientDescent(cost=cost, step_rule=step_rule)

print algorithm


mnist_train = MNIST("train")
train_stream = DataStream(
    dataset=mnist_train,
    iteration_scheme= SequentialScheme(num_examples= 1000, batch_size= 128))

mnist_test = MNIST("test")
test_stream = DataStream(
    dataset=mnist_test,
    iteration_scheme= SequentialScheme(num_examples= mnist_test.num_examples, batch_size= 1024))

monitor_test = DataStreamMonitoring(variables=[cost], data_stream=test_stream, prefix="test")
monitor_train = DataStreamMonitoring(variables=[cost], data_stream=train_stream, prefix="train")

main_loop = MainLoop(
        model=Y,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[monitor_test, monitor_train, Printing()]
        )

main_loop.run()

