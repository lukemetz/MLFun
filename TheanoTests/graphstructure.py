import theano
from theano import tensor
import numpy

x = tensor.dscalar('x')

f = theano.function([x], 10*x, mode='DebugMode')

print f(5)
f(0)
f(7)
