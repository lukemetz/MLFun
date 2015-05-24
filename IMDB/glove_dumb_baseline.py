# Linear model on the sum of glove features
# Gets train cost around .45, .46 and doesn't overfit.
# misclass error is like .17

import theano
from theano import tensor as T
import numpy as np

from dataset import IMDBText, GloveTransformer

from blocks.initialization import Uniform, Constant, IsotropicGaussian, NdarrayInitialization, Identity, Orthogonal
from blocks.bricks.recurrent import LSTM, SimpleRecurrent, GatedRecurrent
from blocks.bricks.parallel import Fork

from blocks.bricks import Linear, Sigmoid, Tanh, Rectifier

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


def main():
    x = T.tensor3('features')
    m = T.matrix('features_mask')
    y = T.imatrix('targets')

    embedding_size = 300
    glove_version = "glove.6B.300d.txt"
    #embedding_size = 50
    #glove_version = "vectors.6B.50d.txt"

    o = x.sum(axis=1) + m.mean() * 0

    score_layer = Linear(
            input_dim = 300,
            output_dim = 1,
            weights_init = IsotropicGaussian(std=0.02),
            biases_init = Constant(0.),
            name="linear2")
    score_layer.initialize()
    o = score_layer.apply(o)
    probs = Sigmoid().apply(o)

    cost = - (y * T.log(probs) + (1-y) * T.log(1 - probs)).mean()
    cost.name = 'cost'
    misclassification = (y * (probs < 0.5) + (1-y) * (probs > 0.5)).mean()
    misclassification.name = 'misclassification'

    # =================
    cg = ComputationGraph([cost])
    params = cg.parameters

    algorithm = GradientDescent(
            cost = cg.outputs[0],
            params=params,
            step_rule = CompositeRule([
                StepClipping(threshold=4),
                AdaM(),
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
    batch_size = 16
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
                mask_sources=('features',)
                )

        fuel.server.start_server(padded, port=port, hwm=20)

    train_port = ports[theano.config.device + '_train']
    train_p = Process(target=start_server, args=(train_port, 'train'))
    train_p.start()

    test_port = ports[theano.config.device + '_test']
    test_p = Process(target=start_server, args=(test_port, 'test'))
    test_p.start()

    train_stream = ServerDataStream(('features', 'features_mask', 'targets'), port=train_port)
    test_stream = ServerDataStream(('features', 'features_mask', 'targets'), port=test_port)

    print "setting up model"

    n_examples = 25000
    #======
    model = Model(cost)
    extensions = []
    extensions.append(EpochProgress(batch_per_epoch=n_examples // batch_size + 1))
    extensions.append(TrainingDataMonitoring(
        [
            cost,
            misclassification,
            ],
        prefix='train',
        after_epoch=True
        ))

    #extensions.append(DataStreamMonitoring(
        #[cost, misclassification],
        #data_stream=test_stream,
        #prefix='test',
        #after_epoch=True
        #))
    extensions.append(Timing())
    extensions.append(Printing())

    extensions.append(Plot(
        theano.config.device+"_result",
        channels=[['train_cost']],
        after_epoch=True
        ))


    main_loop = MainLoop(
            model=model,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions)
    main_loop.run()

if __name__ == "__main__":
    main()
