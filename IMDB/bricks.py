# -*- coding: utf-8 -*-
from blocks.bricks import Activation, Initializable
from blocks.bricks.recurrent import GatedRecurrent
from blocks.initialization import Constant
from blocks.bricks.base import application, Brick, lazy

from theano import tensor
import numpy as np

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


class GatedRecurrentFull(Initializable):
    """A wrapper around the GatedRecurrent brick that improves usability.
    It contains:
        * A fork to map to initialize the reset and the update units.
        * Better initialization to initialize the different pieces
    While this works, there is probably a better more elegant way to do this.

    Parameters
    ----------
    hidden_dim : int
        dimension of the hidden state
    activation : :class:`.Brick`
    gate_activation: :class:`.Brick`

    state_to_state_init: object
        Weight Initialization
    state_to_reset_init: object
        Weight Initialization
    state_to_update_init: object
        Weight Initialization

    input_to_state_transform: :class:`.Brick`
        [CvMG14] uses Linear transform
    input_to_reset_transform: :class:`.Brick`
        [CvMG14] uses Linear transform
    input_to_update_transform: :class:`.Brick`
        [CvMG14] uses Linear transform

    References
    ---------
        self.rnn = GatedRecurrent(
                weights_init=Constant(np.nan),
                dim=self.hidden_dim,
                activation=self.activation,
                gate_activation=self.gate_activation)
    .. [CvMG14] Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre,
        Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua
        Bengio, *Learning Phrase Representations using RNN Encoder-Decoder
        for Statistical Machine Translation*, EMNLP (2014), pp. 1724-1734.

    """
    @lazy(allocation=['hidden_dim', 'state_to_state_init', 'state_to_update_init', 'state_to_reset_init'],
            initialization=['input_to_state_transform', 'input_to_update_transform', 'input_to_reset_transform'])
    def __init__(self, hidden_dim, activation=None, gate_activation=None,
        state_to_state_init=None, state_to_update_init=None, state_to_reset_init=None,
        input_to_state_transform=None, input_to_update_transform=None, input_to_reset_transform=None,
        **kwargs):

        super(GatedRecurrentFull, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

        self.state_to_state_init = state_to_state_init
        self.state_to_update_init = state_to_update_init
        self.state_to_reset_init = state_to_reset_init

        self.input_to_state_transform = input_to_state_transform
        self.input_to_update_transform = input_to_update_transform
        self.input_to_reset_transform = input_to_reset_transform
        self.input_to_state_transform.name += "_input_to_state_transform"
        self.input_to_update_transform.name += "_input_to_update_transform"
        self.input_to_reset_transform.name += "_input_to_reset_transform"

        self.rnn = GatedRecurrent(
                weights_init=Constant(np.nan),
                dim=self.hidden_dim,
                activation=activation,
                gate_activation=gate_activation)

        self.children = [self.rnn,
                self.input_to_state_transform, self.input_to_update_transform, self.input_to_reset_transform]

    def initialize(self):
        super(GatedRecurrentFull, self).initialize()

        self.input_to_state_transform.initialize()
        self.input_to_update_transform.initialize()
        self.input_to_reset_transform.initialize()

        self.rnn.initialize()

        weight_shape = (self.hidden_dim, self.hidden_dim)
        state_to_state = self.state_to_state_init.generate(rng=self.rng, shape=weight_shape)
        state_to_update= self.state_to_update_init.generate(rng=self.rng, shape=weight_shape)
        state_to_reset = self.state_to_reset_init.generate(rng=self.rng, shape=weight_shape)

        self.rnn.state_to_state.set_value(state_to_state)
        self.rnn.state_to_gates.set_value(np.hstack((state_to_update, state_to_reset)))

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, mask=None):
        """

        Parameters
        ----------
        inputs_ : :class:`~tensor.TensorVariable`
            sequence to feed into GRU. Axes are mb, sequence, features

        mask : :class:`~tensor.TensorVariable`
            A 1D binary array with 1 or 0 to represent data given available.

        Returns
        -------
        output: :class:`theano.tensor.TensorVariable`
            sequence to feed out. Axes are batch, sequence, features
        """
        states_from_in = self.input_to_state_transform.apply(input_)
        update_from_in = self.input_to_update_transform.apply(input_)
        reset_from_in = self.input_to_reset_transform.apply(input_)

        gate_inputs = tensor.concatenate([update_from_in, reset_from_in], axis=2)
        output = self.rnn.apply(inputs=states_from_in, gate_inputs=gate_inputs, mask=mask)

        return output


if __name__ == "__main__":
    from blocks.bricks import Linear
    import theano
    floatX = theano.config.floatX
    x = tensor.tensor3('input')
    gru = GatedRecurrentFull(
            hidden_dim=11,
            state_to_state_init=Constant(1.0),
            state_to_reset_init=Constant(1.0),
            state_to_update_init=Constant(1.0),

            input_to_state_transform = Linear(
                input_dim = 19,
                output_dim = 11,
                weights_init=Constant(0.0),
                biases_init=Constant(0.0)
                ),

            input_to_update_transform = Linear(
                input_dim = 19,
                output_dim = 11,
                weights_init=Constant(0.0),
                biases_init=Constant(0.0)
                ),

            input_to_reset_transform = Linear(
                input_dim = 19,
                output_dim = 11,
                weights_init=Constant(0.0),
                biases_init=Constant(0.0)
                ),
            )
    gru.initialize()
    out = gru.apply(x)
    res = out.shape.eval({x: np.ones((6, 9, 19), dtype=floatX)})
    print res


