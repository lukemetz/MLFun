# Baseline / first draft heavily inspired by
# https://github.com/laurent-dinh/dl_tutorials/blob/master/part_4_rnn/imdb_main.py


# PassageGatedRecurrent: 53 seconds
# CustomBlocksGatedRecurrent: 411 seconds
# vanillaBlocksGatedRecurrent: 539 seconds
# MyPortOfPassageToBlocks: 363 -- c linker 361
# Direct passage with params: 445
# NoOtherthings, c linker 357


# New,
# My custom GRU, 128
# Blocks: 139

# using custom. Pretty sure this is the strict flag for scans.
# How slow is plotting?
# 128 no monitor plot, every 60 batches: 127, great! can monitor batches with training data almost free!, Monitoring 10 to ensure: 131, 129 So frree
# For

#theano.config.nvcc.flags='-arch=sm_52'

from sacred import Experiment
import ipdb

ex = Experiment("rnn")

train_p = None
test_p = None

import atexit
def exit_handler():
    if train_p:
        train_p.terminate()
    if test_p:
        test_p.terminate()

atexit.register(exit_handler)

@ex.config
def default_config():
    wstd = 0.02
    rnn_input_dim = 128
    rnn_dim = 128
    deeply_factor = 0.2

@ex.automain
def main_run(_config, _log):
    from collections import namedtuple
    c = namedtuple("Config", _config.keys())(*_config.values())

    _log.info("Running with" + str(_config))

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

    from transformers import DropSources
    global train_p
    global test_p

    x = T.tensor3('features')
    #m = T.matrix('features_mask')
    y = T.imatrix('targets')

    #x = x+m.mean()*0

    dropout_variables = []
    embedding_size = 300
    glove_version = "glove.6B.300d.txt"
    #embedding_size = 50
    #glove_version = "vectors.6B.50d.txt"

    gloveMapping = Linear(
            input_dim = embedding_size,
            output_dim = c.rnn_input_dim,
            weights_init = Orthogonal(),
            #weights_init = IsotropicGaussian(c.wstd),
            biases_init = Constant(0.0),
            name="gloveMapping"
            )
    gloveMapping.initialize()
    o = gloveMapping.apply(x)
    o = Rectifier(name="gloveRec").apply(o)
    dropout_variables.append(o)

    summed_mapped_glove = o.sum(axis=1) # take out the sequence
    glove_out = Linear(
            input_dim = c.rnn_input_dim,
            output_dim = 1.0,
            weights_init = IsotropicGaussian(c.wstd),
            biases_init = Constant(0.0),
            name="mapping_to_output"
            )
    glove_out.initialize()
    deeply_sup_0 = glove_out.apply(summed_mapped_glove)
    deeply_sup_probs = Sigmoid(name="deeply_sup_softmax").apply(deeply_sup_0)

    input_dim = c.rnn_input_dim
    hidden_dim = c.rnn_dim

    gru = GatedRecurrentFull(
            hidden_dim = hidden_dim,
            activation=Tanh(),
            #activation=bricks.Identity(),
            gate_activation=Sigmoid(),
            state_to_state_init=SumInitialization([Identity(1.0), IsotropicGaussian(c.wstd)]),
            state_to_reset_init=IsotropicGaussian(c.wstd),
            state_to_update_init=IsotropicGaussian(c.wstd),
            input_to_state_transform = Linear(
                input_dim=input_dim,
                output_dim=hidden_dim,
                weights_init=IsotropicGaussian(c.wstd),
                biases_init=Constant(0.0)),
            input_to_update_transform = Linear(
                input_dim=input_dim,
                output_dim=hidden_dim,
                weights_init=IsotropicGaussian(c.wstd),
                #biases_init=Constant(-2.0)),
                biases_init=Constant(-1.0)),
            input_to_reset_transform = Linear(
                input_dim=input_dim,
                output_dim=hidden_dim,
                weights_init=IsotropicGaussian(c.wstd),
                #biases_init=Constant(-3.0))
                biases_init=Constant(-2.0))
            )
    gru.initialize()
    rnn_in = o.dimshuffle(1, 0, 2)
    #rnn_in = o
    #rnn_out = gru.apply(rnn_in, mask=m.T)
    rnn_out = gru.apply(rnn_in)
    state_to_state = gru.rnn.state_to_state
    state_to_state.name = "state_to_state"
    #o = rnn_out[-1, :, :]
    o = rnn_out[-1]

    #o = rnn_out[:, -1, :]
    #o = rnn_out.mean(axis=1)

    #print rnn_last_out.eval({
        #x: np.ones((3, 101, 300), dtype=theano.config.floatX), 
        #m: np.ones((3, 101), dtype=theano.config.floatX)})
    #raw_input()
    #o = rnn_out.mean(axis=1)
    dropout_variables.append(o)

    score_layer = Linear(
            input_dim = hidden_dim,
            output_dim = 1,
            weights_init = IsotropicGaussian(std=c.wstd),
            biases_init = Constant(0.),
            name="linear2")
    score_layer.initialize()
    o = score_layer.apply(o)
    probs = Sigmoid().apply(o)


    #probs = deeply_sup_probs
    cost = - (y * T.log(probs) + (1-y) * T.log(1 - probs)).mean()
    #cost_deeply_sup0 = - (y * T.log(deeply_sup_probs) + (1-y) * T.log(1 - deeply_sup_probs)).mean()
    # cost += cost_deeply_sup0 * c.deeply_factor

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
    cg = apply_dropout(cg, variables=dropout_variables, drop_prob=0.5)
    params = cg.parameters

    algorithm = GradientDescent(
            cost = cg.outputs[0],
            params=params,
            step_rule = CompositeRule([
                StepClipping(threshold=4),
                Adam(learning_rate=0.002,
                    beta1=0.1,
                    beta2=0.001),
                #NAG(lr=0.1, momentum=0.9),
                #AdaDelta(),
                ])

            )

    # ========
    print "setting up data"
    ports = {
            'gpu0_train' : 5557,
            'gpu0_test' : 5558,
            'cuda0_train' : 5557,
            'cuda0_test' : 5558,
            'opencl0:0_train' : 5557,
            'opencl0:0_test' : 5558,
            'gpu1_train' : 5559,
            'gpu1_test' : 5560,
            }

    #batch_size = 16
    #batch_size = 32
    batch_size = 40
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

        padded = DropSources(padded, ['features_mask'])

        fuel.server.start_server(padded, port=port, hwm=20)

    train_port = ports[theano.config.device + '_train']
    train_p = Process(target=start_server, args=(train_port, 'train'))
    train_p.start()

    test_port = ports[theano.config.device + '_test']
    test_p = Process(target=start_server, args=(test_port, 'test'))
    test_p.start()

    #train_stream = ServerDataStream(('features', 'features_mask', 'targets'), port=train_port)
    #test_stream = ServerDataStream(('features', 'features_mask', 'targets'), port=test_port)

    train_stream = ServerDataStream(('features', 'targets'), port=train_port)
    test_stream = ServerDataStream(('features', 'targets'), port=test_port)

    print "setting up model"
    #ipdb.set_trace()

    n_examples = 25000
    print "Batches per epoch", n_examples // (batch_size + 1)
    batches_extensions = 100
    monitor_rate = 50
    #======
    model = Model(cg.outputs[0])
    extensions = []
    extensions.append(EpochProgress(batch_per_epoch=n_examples // batch_size + 1))
    extensions.append(TrainingDataMonitoring(
        [
            cost,
            misclassification
            ],
        prefix='train',
        every_n_batches=monitor_rate,
        ))

    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        data_stream=test_stream,
        prefix='test',
        after_epoch=True,
        before_first_epoch=False
        ))

    extensions.append(Timing())
    extensions.append(Printing())

    #extensions.append(Plot("norms", channels=[['train_lstm_norm', 'train_pre_norm']], after_epoch=True))
    #extensions.append(Plot(theano.config.device+"_result", channels=[['test_misclassification', 'train_misclassification']], after_epoch=True))

    #extensions.append(PlotHistogram(
        #channels=['train_state_to_state'],
        #bins=50,
        #every_n_batches=30))

    extensions.append(Plot(
        theano.config.device+"_result",
        channels=[['train_cost'], ['train_misclassification']],
        every_n_batches=monitor_rate))


    main_loop = MainLoop(
            model=model,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions)
    main_loop.run()
