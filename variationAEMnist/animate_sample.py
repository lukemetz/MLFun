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

_func_sample = theano.function([m.Z], m.sampled)
#_func_noisy = theano.function([m.X], m.noisy)
#_func_produced = theano.function([m.X], m.produced)

#batch = test_stream.get_epoch_iterator().next()[0]
#out_noise = _func_noisy(batch)
#out_produced = _func_produced(batch)
import cv2
#out_sampled = _func_sample(np.random.normal(size=(20, m.n_hidden)).astype(theano.config.floatX))
num_img = 20
img_size = 256

n = 500*3
t = np.linspace(0, 1, n)
t = 1-t
x = np.sin(t * 4.0 * np.pi * 2.0) * 2*(t+.2)
y = np.cos(t * 4.0 * np.pi * 2.0) * 2*(t+.2)

zs = np.vstack((x, y)).T.astype(theano.config.floatX)
out_sampled = _func_sample(zs)

canvas = np.zeros((img_size*num_img, img_size*num_img))
for j in range(n):
    img = np.reshape(out_sampled[j, :], (28, 28))
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('dump/img%05d.png'%j, (1.0 - img)*255)
