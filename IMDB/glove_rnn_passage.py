# Baseline / first draft heavily inspired by
# https://github.com/laurent-dinh/dl_tutorials/blob/master/part_4_rnn/imdb_main.py

import theano
from theano import tensor as T
import numpy as np

from dataset import IMDBText, GloveTransformer

from blocks.initialization import Uniform, Constant, IsotropicGaussian, NdarrayInitialization, Identity, Orthogonal
from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent
from blocks.bricks.parallel import Fork

from blocks.bricks import Linear, Sigmoid, Tanh, Rectifier
from blocks import bricks

from blocks.extensions import Printing, Timing
from blocks.extensions.monitoring import (DataStreamMonitoring,
        TrainingDataMonitoring)

from blocks.extensions.plot import Plot
from plot import PlotHistogram

from blocks.algorithms import GradientDescent, Adam, Scale, StepClipping, CompositeRule, AdaDelta
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model

from cuboid.algorithms import AdaM, NAG
from cuboid.extensions import EpochProgress

from fuel.streams import DataStream, ServerDataStream
from fuel.transformers import Padding

from fuel.schemes import ShuffledScheme
from Conv1D import Conv1D, MaxPooling1D
from schemes import BatchwiseShuffledScheme
from bricks import WeightedSigmoid, GatedRecurrentFull

from multiprocessing import Process
import fuel
import logging
from initialization import SumInitialization

from passage.layers import GatedRecurrent as GatedRecurrentPassage

def main():
    x = T.tensor3('features')
    m = T.matrix('features_mask')
    y = T.imatrix('targets')

    x = x+m.mean()*0

    embedding_size = 300
    glove_version = "glove.6B.300d.txt"
    #embedding_size = 50
    #glove_version = "vectors.6B.50d.txt"
    wstd = 0.02

    #vaguely normalize
    x = x / 3.0 - .5

    #gloveMapping = Linear(
            #input_dim = embedding_size,
            #output_dim = 128,
            #weights_init = Orthogonal(),
            #biases_init = Constant(0.0),
            #name="gloveMapping"
            #)
    #gloveMapping.initialize()
    #o = gloveMapping.apply(x)
    #o = Rectifier(name="gloveRec").apply(o)

    rnn_in = x.dimshuffle(1, 0, 2)
    class Stub(object):
        def output(self, dropout_active=False):
            return rnn_in

    l_in = Stub()
    l_in.size = 300

    layer = GatedRecurrentPassage(
            size=300,
            gate_activation='sigmoid')
    layer.connect(l_in)
    from blocks.roles import add_role, WEIGHT, INITIAL_STATE
    print layer.params
    [add_role(l, WEIGHT) for l in layer.params]

    rnn_out = layer.output()
    o = rnn_out
    #o = rnn_out[-1, :, :]

    #o = rnn_out[:, -1, :]
    #o = rnn_out.mean(axis=1)

    #print rnn_last_out.eval({
        #x: np.ones((3, 101, 300), dtype=theano.config.floatX), 
        #m: np.ones((3, 101), dtype=theano.config.floatX)})
    #raw_input()
    #o = rnn_out.mean(axis=1)

    score_layer = Linear(
            input_dim = 300,
            output_dim = 1,
            weights_init = IsotropicGaussian(std=wstd),
            biases_init = Constant(0.),
            name="linear2")
    score_layer.initialize()
    o = score_layer.apply(o)
    probs = Sigmoid().apply(o)

    cost = - (y * T.log(probs) + (1-y) * T.log(1 - probs)).mean()
    cost.name = 'cost'
    misclassification = (y * (probs < 0.5) + (1-y) * (probs > 0.5)).mean()
    misclassification.name = 'misclassification'

    #print rnn_in.shape.eval(
            #{x : np.ones((45, 111, embedding_size), dtype=theano.config.floatX),
                #})
    #print rnn_out.shape.eval(
            #{x : np.ones((45, 111, embedding_size), dtype=theano.config.floatX),
                #m : np.ones((45, 111), dtype=theano.config.floatX)})
    #print (m).sum(axis=1).shape.eval({
                #m : np.ones((45, 111), dtype=theano.config.floatX)})
    #print (m).shape.eval({
                #m : np.ones((45, 111), dtype=theano.config.floatX)})
    #raw_input()


    # =================

    cg = ComputationGraph([cost])
    #cg = apply_dropout(cg, variables=dropout_variables, drop_prob=0.5)
    params = cg.parameters
    print params
    print "Len params", len(params)

    algorithm = GradientDescent(
            cost = cg.outputs[0],
            params=params,
            step_rule = CompositeRule([
                StepClipping(threshold=4),
                AdaM(),
                #NAG(lr=0.1, momentum=0.9),
                #AdaDelta(),
                ])

            )

    # ========
    print "setting up data"
    ports = {
            'gpu0_train' : 5557,
            'gpu0_test' : 5558,
            'gpu1_train' : 5559,
            'gpu1_test' : 5560,
            }

    #batch_size = 16
    batch_size = 32
    def start_server(port, which_set):
        fuel.server.logger.setLevel('WARN')
        dataset = IMDBText(which_set, sorted=True)

        n_train = dataset.num_examples
        #scheme = ShuffledScheme(examples=n_train, batch_size=batch_size)
        scheme = BatchwiseShuffledScheme(examples=n_train, batch_size=batch_size)

        stream = DataStream(
                dataset=dataset,
                iteration_scheme=scheme)
        print "loading glove"
        glove = GloveTransformer(glove_version, data_stream=stream)
        padded = Padding(
                data_stream=glove,
                #mask_sources=('features',)
                mask_sources=('features',)
                )

        fuel.server.start_server(padded, port=port, hwm=20)

    train_port = ports[theano.config.device + '_train']
    train_p = Process(target=start_server, args=(train_port, 'train'))
    train_p.start()

    test_port = ports[theano.config.device + '_test']
    test_p = Process(target=start_server, args=(test_port, 'test'))
    test_p.start()
sources
    train_stream = ServerDataStream(('features', 'features_mask', 'targets'), port=train_port)
    test_stream = ServerDataStream(('features', 'features_mask', 'targets'), port=test_port)

    print "setting up model"
    #import ipdb
    #ipdb.set_trace()

    n_examples = 25000
    #======
    model = Model(cost)
    extensions = []
    extensions.append(EpochProgress(batch_per_epoch=n_examples // batch_size + 1))
    extensions.append(TrainingDataMonitoring(
        [
            cost,
            misclassification
            ],
        prefix='train',
        every_n_batches=30
        ))

    #extensions.append(DataStreamMonitoring(
        #[cost, misclassification],
        #data_stream=test_stream,
        #prefix='test',
        #after_epoch=True
        #))
    extensions.append(Timing())
    extensions.append(Printing())

    #extensions.append(Plot("norms", channels=[['train_lstm_norm', 'train_pre_norm']], after_epoch=True))
    #extensions.append(Plot(theano.config.device+"_result", channels=[['test_misclassification', 'train_misclassification']], after_epoch=True))
    extensions.append(PlotHistogram(
        channels=['train_state_to_state'],
        bins=50,
        every_n_batches=30))

    extensions.append(Plot(
        theano.config.device+"_result",
        channels=[['train_cost']],
        every_n_batches=30))


    main_loop = MainLoop(
            model=model,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions)
    main_loop.run()


if __name__ == "__main__":
    main()
