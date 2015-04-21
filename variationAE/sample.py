from model import VAModel

from blocks.dump import load_parameter_values
from blocks.model import Model
import sys
from fuel.datasets.cifar10 import CIFAR10
from fuel.streams import DataStream

from fuel.schemes import ShuffledScheme
import theano
import theano.tensor as T
import logging
import numpy as np


logging.basicConfig()

m = VAModel()

# load parameters
model = Model(m.variational_cost)
print "loading params"
params = load_parameter_values(sys.argv[1])
model.set_param_values(params)

test_dataset = CIFAR10('test', sources=['features'])
test_scheme = ShuffledScheme(test_dataset.num_examples, 128)
test_stream = DataStream(test_dataset, iteration_scheme=test_scheme)

o = T.reshape(m.sampled, (m.sampled.shape[0], 3, 32, 32))
out = o.dimshuffle((0, 2, 3, 1))
_func_sample = theano.function([m.Z], out)
#_func_noisy = theano.function([m.X], m.noisy)
#_func_produced = theano.function([m.X], m.produced)

#batch = test_stream.get_epoch_iterator().next()[0]
#out_noise = _func_noisy(batch)
#out_produced = _func_produced(batch)
import cv2
#out_sampled = _func_sample(np.random.normal(size=(20, m.n_hidden)).astype(theano.config.floatX))
num_img = 15
img_size = 64
zs = np.random.normal(size=(num_img*num_img, m.n_zs)).astype(theano.config.floatX)

out_sampled = _func_sample(zs)
canvas = np.zeros((img_size*num_img, img_size*num_img, 3))
for j in range(num_img):
    for k in range(num_img):
        img = np.reshape(out_sampled[k+j*num_img, :], (32, 32, 3))
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        canvas[(j*img_size):(j+1)*img_size, k*img_size:(k+1)*img_size, :] = img

cv2.imshow('img', 1.-canvas)
cv2.imwrite('img.png', (1.0 - canvas)*255)
cv2.waitKey(0)


