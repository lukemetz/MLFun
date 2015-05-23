from blocks.bricks import Activation
from blocks.bricks.base import application
from theano import tensor

class WeightedSigmoid(Activation):
    """Weighted sigmoid
    f(x) = 1.0/ (1.0 + exp(-a * x))

    Parameters
    ----------
    a : float

    References
    ---------
    .. [1] Qi Lyu, Jun Zhu
           "Revisit LongShort-Term Memory: An Optimization Perspective"
    """
    def __init__(self, a=1.0, **kwargs):
        self.a = a
        super(WeightedSigmoid, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return 1.0 / (1.0 + tensor.exp(- self.a * input_))

