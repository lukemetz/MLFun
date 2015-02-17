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

class BatchNormalizationBase(Brick):

    seed_rng = np.random.RandomState(config.default_seed)

    @lazy
    def __init__(self, B_init, Y_init, epsilon=1e-9, seed=None, **kwargs):
        super(BatchNormalizationBase, self).__init__(**kwargs)
        self.eps = epsilon
        self.seed = seed
        self.B_init = B_init
        self.Y_init = Y_init

    @property
    def seed(self):
        if getattr(self, '_seed', None) is not None:
            return self._seed
        else:
            self._seed = self.seed_rng.randint(np.iinfo(np.int32).max)
            return self._seed

    @seed.setter
    def seed(self, value):
        if hasattr(self, '_seed'):
            raise AttributeError("seed already set")
        self._seed = value

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
                return self._rng
        else:
            return np.random.RandomState(self.seed)

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def _initialize(self):
        B, Y = self.params
        self.B_init.initialize(B, self.rng)
        self.Y_init.initialize(Y, self.rng)


class BatchNormalizationConv(BatchNormalizationBase):
    @lazy
    def __init__(self, input_shape, **kwargs):
        super(BatchNormalizationConv, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.num_channels = input_shape[0]

    def _allocate(self):
        B = shared_floatx_zeros((self.num_channels,))
        self.params.append(B)

        Y = shared_floatx_zeros((self.num_channels,))
        self.params.append(Y)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        Reference: http://arxiv.org/pdf/1502.03167v2.pdf
        """
        minibatch_mean = T.mean(input_, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        minibatch_var = T.var(input_, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        norm_x = (input_ - minibatch_mean) / (T.sqrt(minibatch_var + self.eps))
        B, Y = self.params
        B = B.dimshuffle(('x', 0, 'x', 'x'))
        Y = Y.dimshuffle(('x', 0, 'x', 'x'))
        return norm_x * Y + B

    def get_dim(self, name):
        if name == "input_":
            return self.input_shape
        if name == "output":
            return self.input_shape
        super(BatchNormalizationConv, self).get_dim(name)

class BatchNormalization(BatchNormalizationBase):
    @lazy
    def __init__(self, input_dim, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.input_dim = input_dim

    def _allocate(self):
        B = shared_floatx_zeros((self.input_dim,))
        self.params.append(B)

        Y = shared_floatx_zeros((self.input_dim,))
        self.params.append(Y)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """
        Reference: http://arxiv.org/pdf/1502.03167v2.pdf
        """
        minibatch_mean = T.mean(input_, axis=[0]).dimshuffle('x', 0)
        minibatch_var = T.var(input_, axis=[0]).dimshuffle('x', 0)
        norm_x = (input_ - minibatch_mean) / (T.sqrt(minibatch_var + self.eps))
        B, Y = self.params
        B = B.dimshuffle(('x', 0))
        Y = Y.dimshuffle(('x', 0))
        return norm_x * Y + B

    def get_dim(self, name):
        if name == "input_":
            return self.input_dim
        if name == "output":
            return self.input_dim
        super(BatchNormalizationConv, self).get_dim(name)

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

class AdaM(StepRule):
    def __init__(self, alpha=0.0002, b1=0.1, b2=0.001, eps=1e-8):
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

        step = alpha * m_hat / (T.sqrt(v_hat) + eps)
        updates = [(m, m_new), (v, v_new), (t, t_new)]

        return step, updates

print "Learning rules"

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

monitor_test = DataStreamMonitoring(variables=[cost], data_stream=test_stream, prefix="test")
monitor_train = DataStreamMonitoring(variables=[cost], data_stream=train_stream, prefix="train")


class ToFile(SimpleExtension):
    def __init__(self, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_every_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(ToFile, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.main_loop.log.to_dataframe().to_csv("norm2.csv")


print "Making main loop"
main_loop = MainLoop(
        model=Y,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[monitor_test, monitor_train, Timing(), Printing(), ToFile()]
        )

print "Starting main"
main_loop.run()

