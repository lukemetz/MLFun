import theano
from theano import tensor as T
import pydot

a = T.matrix("matrixA")

b = T.dot(a, (a + 1) * a)

c = (b + a)**2

f = theano.function([a], c)

#theano.printing.pydotprint(c, outfile='unopt.png', var_with_name_simple=True)
#theano.printing.pydotprint(f, outfile='opt.png', var_with_name_simple=True)

f = theano.function([a], c)
theano.printing.pydotprint(f, outfile='gpuopt.png', var_with_name_simple=True)
theano.printing.pydotprint(f, outfile='gpuopt.png', var_with_name_simple=False)
