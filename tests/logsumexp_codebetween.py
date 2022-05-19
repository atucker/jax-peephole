# Test that we do logsumexp trick if there's code between lines

import peephole
import jax.numpy as np
from jax import make_jaxpr


@peephole.improve
def test(x):
    a = np.exp(x)
    d = x * 2 # has to be unrelated to a, b, c
    b = np.sum(a)
    c = np.log(b)
    return c


print(test(np.array(range(5))))