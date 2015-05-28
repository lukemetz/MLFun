from theano import gof
import ipdb

class Double(gof.Type):

    def filter(self, x, strict=False, allow_downcast=None):
        if strict:
            if isinstance(x, float):
                return x
            else:
                raise TypeError('expected a float')
        elif allow_downcast:
            return float(x)
        else:
            x_float = float(x)
            if x_float == x:
                return x_float
            else:
                raise TypeError('The double cannot accuratly rep')
    def values_eq_approx(self, x, y, tolerance=1e-4):
        return abs(x-y) / (abs(x) + abs(y)) < tolerance

    def __str__(self):
        return "double"

double = Double()
#ipdb.set_trace()

class BinaryDoubleOp(gof.Op):
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
    def __eq__(self, other):
        return type(self) == type(other) and (self.name == other.name) and (self.fn == other.fn)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.name) ^ hash(self.fn)

    def make_node(self, x, y):
        if isinstance(x, (int, float)):
            x = gof.Constant(double, x)
        if isinstance(y,  (int, float)):
            y = gof.Constant(double, y)
        if x.type != double or y.type != double:
            raise TypeError("Only works for doubles silly")

        return gof.Apply(self, [x, y], [double(), double()])

    def perform(self, node, inp, output):
        x,y = inp
        z,v = output

        z[0] = self.fn(x,y)
        v[0] = self.fn(x,y) + self.fn(x,y)

add = BinaryDoubleOp(name="add", fn =lambda x, y: x + y)

x = gof.Variable(double, name="x")
y = gof.Variable(double, name="y")

apply_ = add.make_node(x,y)
o = apply_.outputs[0]
o = apply_.outputs[1]

#o = add(x, y)
import theano
#f = theano.function([x, y], o, mode='DebugMode')
#print f(12, 23)

import numpy
import theano
from theano import gof
import theano.tensor as T

class VectorTimesScalar(gof.Op):
    __props__ = ()

    def make_node(self, x, y):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 0:
            raise TypeError('y must be a scalar')

        # Create an output variable of the same type as x
        output_var = x.type()

        return gof.Apply(self, [x, y], [output_var])

    def c_code_cache_version(self):
        return (1, 1)

    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        z, = out

        # Extract the dtypes of the inputs and outputs storage to
        # be able to declare pointers for those dtypes in the C
        # code.
        dtype_x = node.inputs[0].dtype
        dtype_y = node.inputs[1].dtype
        dtype_z = node.outputs[0].dtype

        itemsize_x = numpy.dtype(dtype_x).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize

        fail = sub['fail']
        print dtype_x, dtype_z
        print name
        print inp
        print sub

        c_code = """
        // Validate that the output storage exists and has the same
        // dimension as x.
        if (NULL == %(z)s ||
            PyArray_DIMS(%(x)s)[0] != PyArray_DIMS(%(z)s)[0])
        {
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_EMPTY(1,
                                                PyArray_DIMS(%(x)s),
                                                PyArray_TYPE(%(x)s),
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        // Perform the vector multiplication by a scalar
        {
            /* The declaration of the following variables is done in a new
            scope to prevent cross initialization errors */
            npy_%(dtype_x)s* x_data_ptr =
                            (npy_%(dtype_x)s*)PyArray_DATA(%(x)s);
            npy_%(dtype_z)s* z_data_ptr =
                            (npy_%(dtype_z)s*)PyArray_DATA(%(z)s);
            npy_%(dtype_y)s y_value =
                            ((npy_%(dtype_y)s*)PyArray_DATA(%(y)s))[0];
            int x_stride = PyArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
            int z_stride = PyArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
            int x_dim = PyArray_DIMS(%(x)s)[0];

            for(int i=0; i < x_dim; i++)
            {
                z_data_ptr[i * z_stride] = (x_data_ptr[i * x_stride] *
                                            y_value);
            }
        }
        """

        return c_code % locals()

from theano import tensor
vecxscalar = VectorTimesScalar()
print vecxscalar
x = tensor.vector("x")
y = tensor.scalar("scalar")
out = vecxscalar(x, y)
f = theano.function([x, y], out)
print f([1,2,3], 2.2)


import theano
from theano import gof

class VectorTimesVector(gof.COp):
    __props__ = ()

    func_file = "./vectorTimesVector.c"
    func_name = "APPLY_SPECIFIC(vector_times_vector)"

    def __init__(self):
        super(VectorTimesVector, self).__init__(self.func_file,
                                                self.func_name)

    def make_node(self, x, y):
        # Validate the inputs' type
        if x.type.ndim != 1:
            raise TypeError('x must be a 1-d vector')
        if y.type.ndim != 1:
            raise TypeError('y must be a 1-d vector')

        # Create an output variable of the same type as x
        output_var = theano.tensor.TensorType(
                        dtype=theano.scalar.upcast(x.dtype, y.dtype),
                        broadcastable=[False])()

        return gof.Apply(self, [x, y], [output_var])

vecxvec = VectorTimesVector()
x = tensor.vector()
y = tensor.vector()
z = vecxvec(x,y)

f = theano.function([x, y], z)
print f([1,2], [2,3])
