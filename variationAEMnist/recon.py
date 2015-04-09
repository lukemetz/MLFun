from model import VAModel

from blocks.dump import load_parameter_values
from blocks.model import Model
import sys
from fuel.datasets.mnist import MNIST
from fuel.streams import DataStream

from fuel.schemes import ShuffledScheme
import theano
import logging
import numpy as np


logging.basicConfig()

m = VAModel()

# load parameters
model = Model(m.variational_cost)
print "loading params"
params = load_parameter_values(sys.argv[1])
model.set_param_values(params)

test_dataset = MNIST('test', sources=['features'])
test_scheme = ShuffledScheme(test_dataset.num_examples, 128)
test_stream = DataStream(test_dataset, iteration_scheme=test_scheme)

_func_noisy = theano.function([m.X], m.noisy)
_func_produced = theano.function([m.X], m.produced)

batch = test_stream.get_epoch_iterator().next()[0]
out_noise = _func_noisy(batch)
out_produced = _func_produced(batch)
import cv2
for k in range(10):
    print out_noise.shape
    img = np.reshape(out_noise[k, :], (28, 28))
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)

    img = np.reshape(out_produced[k, :], (28, 28))
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('masdf', img)

    cv2.waitKey(0)

