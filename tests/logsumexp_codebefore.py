# Test that we do trick if there's code before

import peephole
import jax.numpy as np
from jax import make_jaxpr


@peephole.improve
def test(x):
    x = 2 * x
    return np.log(np.sum(np.exp(x)))


print(test(np.array(range(5))))