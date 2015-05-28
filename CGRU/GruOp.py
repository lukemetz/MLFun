import theano
from theano import tensor
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from blocks.utils import shared_floatx_zeros
from theano.gof import COp


class GruOp(GpuOp, COp):
    __props__ = ()

    def __init__(self):
        COp.__init__(self, ['GruOp.c'],
                'APPLY_SPECIFIC(gated_unit_main)')

    def make_node(self, inp,
            inp_to_hidden):
    #def make_node(self, inp,
            #inp_to_hidden, inp_to_update, inp_to_reset):
        inp = as_cuda_ndarray_variable(inp)

        inp_to_hidden = as_cuda_ndarray_variable(inp_to_hidden)

        weights = [inp_to_hidden]

        for w in weights:
            assert w.dtype == "float32"
            assert w.ndim == 2

        out_type = CudaNdarrayType((False, False))
        return theano.Apply(self, [inp, inp_to_hidden], [out_type()])

    def c_code_cache_version(self):
        return tuple()

gru = GruOp()

if __name__ == "__main__":
    theano.config.optimizer='None'
    import numpy as np
    x = tensor.matrix("inp_variable")
    #x = tensor.tensor3("inp_variable")
    inp_to_hidden = shared_floatx_zeros((7, 3), name="inp_to_hidden_shared")
    inp_to_hidden += 1
    res = gru(x, inp_to_hidden)

    f = theano.function([x], res, mode='DebugMode')
    print f.maker.fgraph.toposort()
    print f(np.ones((6,7), dtype=theano.config.floatX))
    print np.asarray(f(np.ones((6,7), dtype=theano.config.floatX)))
