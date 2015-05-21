# Baseline / first draft heavily inspired by
# https://github.com/laurent-dinh/dl_tutorials/blob/master/part_4_rnn/imdb_main.py

import theano
from theano import tensor as T
from dataset import IMDB

from blocks.bricks.lookup import LookupTable
from blocks.initialization import Uniform, Constant
from blocks.bricks.recurrent import LSTM, SimpleRecurrent
from blocks.bricks import Linear, Sigmoid, Tanh

from blocks.extensions import Printing, Timing
from blocks.extensions.monitoring import (DataStreamMonitoring,
        TrainingDataMonitoring)

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
    x = T.imatrix('features')
    m = T.matrix('features_mask')
    y = T.imatrix('targets')
    #x_int = x.astype(dtype='int32').T
    x_int = x.T

    train_dataset = IMDB('train')
    n_voc = len(train_dataset.dict.keys())
    n_h = 2
    lookup = LookupTable(
            length=n_voc+2,
            dim = n_h*4,
            weights_init = Uniform(std=0.01),
            biases_init = Constant(0.)
        )
    lookup.initialize()

    #rnn = SimpleRecurrent(
            #dim = n_h,
            #activation=Tanh(),
            #weights_init = Uniform(std=0.01),
            #biases_init = Constant(0.)
        #)
    rnn = LSTM(
            dim = n_h,
            activation=Tanh(),
            weights_init = Uniform(std=0.01),
            biases_init = Constant(0.)
        )

    rnn.initialize()
    score_layer = Linear(
            input_dim = n_h,
            output_dim = 1,
            weights_init = Uniform(std=0.01),
            biases_init = Constant(0.))
    score_layer.initialize()

    embedding = lookup.apply(x_int) * T.shape_padright(m.T)
    #embedding = lookup.apply(x_int) + m.T.mean()*0
    #embedding = lookup.apply(x_int) + m.T.mean()*0

    rnn_states = rnn.apply(embedding, mask=m.T)
    #rnn_states, rnn_cells = rnn.apply(embedding)
    rnn_out_mean_pooled = rnn_states[-1]
    #rnn_out_mean_pooled = rnn_states.mean()

    probs = Sigmoid().apply(
        score_layer.apply(rnn_out_mean_pooled))

    cost = - (y * T.log(probs) + (1-y) * T.log(1 - probs)).mean()
    cost.name = 'cost'
    misclassification = (y * (probs < 0.5) + (1-y) * (probs > 0.5)).mean()
    misclassification.name = 'misclassification'


    # =================

    cg = ComputationGraph([cost])
    params = cg.parameters
    algorithm = GradientDescent(
            cost = cost,
            params=params,
            step_rule = CompositeRule([
                StepClipping(threshold=10),
                Adam(),
                #AdaDelta(),
                ])

            )


    # ========

    test_dataset = IMDB('test')
    batch_size = 64
    n_train = train_dataset.num_examples
    train_stream = DataStream(
            dataset=train_dataset,
            iteration_scheme=ShuffledScheme(
                examples=n_train,
                batch_size=batch_size)
            )
    train_padded = Padding(
            data_stream=train_stream,
            mask_sources=('features',)
            #mask_sources=[]
            )


    test_stream = DataStream(
            dataset=test_dataset,
            iteration_scheme=ShuffledScheme(
                examples=n_train,
                batch_size=batch_size)
            )
    test_padded = Padding(
            data_stream=test_stream,
            mask_sources=('features',)
            #mask_sources=[]
            )
    #import ipdb
    #ipdb.set_trace()

    #======
    model = Model(cost)
    extensions = []
    extensions.append(EpochProgress(batch_per_epoch=train_dataset.num_examples // batch_size + 1))
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification],
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

    main_loop = MainLoop(
            model=model,
            data_stream=train_padded,
            algorithm=algorithm,
            extensions=extensions)
    main_loop.run()


if __name__ == "__main__":
    main()
