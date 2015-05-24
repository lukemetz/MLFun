from blocks.initialization import NdarrayInitialization
import numpy as np

class SumInitialization(NdarrayInitialization):
    """Initialize based on the sum of a list of initializations

    Parameters
    ---------
    initializations : `[NdarrayInitialization]`
        array of initializations with generate method.
    """
    def __init__(self, initializations):
        self.initializations = initializations

    def generate(self, rng, shape):
        inits = [init.generate(rng, shape) for init in self.initializations]
        return sum(inits)

class MultiplyInitialization(NdarrayInitialization):
    """Initialize based on the elementwise multiplication of the list of initializations

    Parameters
    ---------
    initializations : `[NdarrayInitialization]`
        array of initializations with generate method.
    """
    def __init__(self, initializations):
        self.initializations = initializations

    def generate(self, rng, shape):
        inits = [init.generate(rng, shape) for init in self.initializations]
        return reduce(operator.mul, inits, 1)
