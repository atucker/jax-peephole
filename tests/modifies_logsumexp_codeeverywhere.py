# Test that we do logsumexp trick if there's code between lines

import peephole
import jax.numpy as np
from jax import make_jaxpr

@peephole.improve
def test(x):
    x = x * 2
    a = np.exp(x)
    d = x * 2 # has to be unrelated to a, b, c
    b = np.sum(a)
    c = np.log(b)
    return c + d


print(test(np.array(range(5))) is not None)