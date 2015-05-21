# Baseline / first draft heavily inspired by
# https://github.com/laurent-dinh/dl_tutorials/blob/master/part_4_rnn/imdb_main.py

import theano
from theano import tensor as T
import numpy as np

from dataset import IMDBText, GloveTransformer

from blocks.initialization import Uniform, Constant, IsotropicGaussian, NdarrayInitialization
from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent
from blocks.bricks.parallel import Fork

from blocks.bricks import Linear, Sigmoid, Tanh, Rectifier

from blocks.extensions import Printing, Timing
from blocks.extensions.monitoring import (DataStreamMonitoring,
        TrainingDataMonitoring)
from blocks.extensions.plot import Plot


from blocks.algorithms import GradientDescent, Adam, Scale, StepClipping, CompositeRule, AdaDelta
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model

from cuboid.algorithms import AdaM
from cuboid.extensions import EpochProgress

from fuel.streams import DataStream
from fuel.transformers import Padding

from fuel.schemes import ShuffledScheme


def main():
    x = T.tensor3('features')
    m = T.matrix('features_mask')
    y = T.imatrix('targets')

    #rnn = SimpleRecurrent(
            #dim = 50,
            #activation=Tanh(),
            #weights_init = Uniform(std=0.01),
            #biases_init = Constant(0.)
        #)

    #rnn = GatedRecurrent(
            #dim = 50,
            #activation=Tanh(),
            #weights_init = Uniform(std=0.01),
            #biases_init = Constant(0.)
        #)

    embedding_size = 300
    #glove_version = "vectors.6B.100d.txt"
    glove_version = "glove.6B.300d.txt"
    #fork = Fork(weights_init=IsotropicGaussian(0.02),
            #biases_init=Constant(0.),
            #input_dim=embedding_size,
            #output_dims=[embedding_size]*3,
            #output_names=['inputs', 'reset_inputs', 'update_inputs']
            #)

    rnn = LSTM(
            dim = embedding_size,
            activation=Tanh(),
            weights_init = IsotropicGaussian(std=0.02),
        )
    rnn.initialize()

    #fork.initialize()
    wstd = 0.02

    score_layer = Linear(
            input_dim = 128,
            output_dim = 1,
            weights_init = IsotropicGaussian(std=wstd),
            biases_init = Constant(0.),
            name="linear2")
    score_layer.initialize()

    gloveMapping = Linear(
            input_dim = embedding_size,
            output_dim = embedding_size,
            weights_init = IsotropicGaussian(std=wstd),
            biases_init = Constant(0.0),
            name="gloveMapping"
            )
    gloveMapping.initialize()
    o = gloveMapping.apply(x)
    o = Rectifier(name="rectivfyglove").apply(o)

    forget_bias = np.zeros((embedding_size*4), dtype=theano.config.floatX)
    forget_bias[embedding_size:embedding_size*2] = 4.0
    toLSTM = Linear(
            input_dim = embedding_size,
            output_dim = embedding_size*4,
            weights_init = IsotropicGaussian(std=wstd),
            biases_init = Constant(forget_bias),
            #biases_init = Constant(0.0),
            name="ToLSTM"
            )
    toLSTM.initialize()


    rnn_states, rnn_cells = rnn.apply(toLSTM.apply(o) * T.shape_padright(m), mask=m)
    #inputs, reset_inputs, update_inputs = fork.apply(x)
    #rnn_states = rnn.apply(inputs=inputs, reset_inputs=reset_inputs, update_inputs=update_inputs, mask=m)

    #rnn_out = rnn_states[:, -1, :]
    rnn_out = (rnn_states * m.dimshuffle(0, 1, 'x')).sum(axis=1) / m.sum(axis=1).dimshuffle(0, 'x')
    #rnn_out = (rnn_states).mean(axis=1)# / m.sum(axis=1)

    hidden = Linear(
        input_dim = embedding_size,
        output_dim = 128,
        weights_init = Uniform(std=0.01),
        biases_init = Constant(0.))
    hidden.initialize()

    o = hidden.apply(rnn_out)
    o = Rectifier().apply(o)
    hidden = Linear(
        input_dim = 128,
        output_dim = 128,
        weights_init = IsotropicGaussian(std=0.02),
        biases_init = Constant(0.),
        name="hiddenmap2")
    hidden.initialize()

    o = hidden.apply(o)
    o = Rectifier(name="rec2").apply(o)

    o = score_layer.apply(o)

    probs = Sigmoid().apply(o)

    cost = - (y * T.log(probs) + (1-y) * T.log(1 - probs)).mean()
    cost.name = 'cost'
    misclassification = (y * (probs < 0.5) + (1-y) * (probs > 0.5)).mean()
    misclassification.name = 'misclassification'

    #print (rnn_states * m.dimshuffle(0, 1, 'x')).sum(axis=1).shape.eval(
            #{x : np.ones((45, 111, embedding_size), dtype=theano.config.floatX),
                #m : np.ones((45, 111), dtype=theano.config.floatX)})
    #print (m).sum(axis=1).shape.eval({
                #m : np.ones((45, 111), dtype=theano.config.floatX)})
    #print (m).shape.eval({
                #m : np.ones((45, 111), dtype=theano.config.floatX)})
    #raw_input()


    # =================

    cg = ComputationGraph([cost])
    params = cg.parameters

    algorithm = GradientDescent(
            cost = cost,
            params=params,
            step_rule = CompositeRule([
                StepClipping(threshold=10),
                AdaM(),
                #AdaDelta(),
                ])

            )


    # ========
    print "setting up data"

    train_dataset = IMDBText('train')
    test_dataset = IMDBText('test')
    batch_size = 16
    n_train = train_dataset.num_examples
    train_stream = DataStream(
            dataset=train_dataset,
            iteration_scheme=ShuffledScheme(
                examples=n_train,
                batch_size=batch_size)
            )
    glove = GloveTransformer(glove_version, data_stream=train_stream)

    train_padded = Padding(
            data_stream=glove,
            mask_sources=('features',)
            #mask_sources=[]
            )


    test_stream = DataStream(
            dataset=test_dataset,
            iteration_scheme=ShuffledScheme(
                examples=n_train,
                batch_size=batch_size)
            )
    glove = GloveTransformer(glove_version, data_stream=test_stream)
    test_padded = Padding(
            data_stream=glove,
            mask_sources=('features',)
            #mask_sources=[]
            )
    print "setting up model"
    #import ipdb
    #ipdb.set_trace()

    lstm_norm = rnn.W_state.norm(2)
    lstm_norm.name = "lstm_norm"

    pre_norm= gloveMapping.W.norm(2)
    pre_norm.name = "pre_norm"

    #======
    model = Model(cost)
    extensions = []
    extensions.append(EpochProgress(batch_per_epoch=train_dataset.num_examples // batch_size + 1))
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification, lstm_norm, pre_norm],
        prefix='train',
        after_epoch=True
        ))

    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        data_stream=test_padded,
        prefix='test',
        after_epoch=True
        ))
    extensions.append(Timing())
    extensions.append(Printing())

    extensions.append(Plot("norms", channels=[['train_lstm_norm', 'train_pre_norm']], after_epoch=True))
    extensions.append(Plot("result", channels=[['train_cost', 'train_misclassification']], after_epoch=True))

    main_loop = MainLoop(
            model=model,
            data_stream=train_padded,
            algorithm=algorithm,
            extensions=extensions)
    main_loop.run()


if __name__ == "__main__":
    main()
