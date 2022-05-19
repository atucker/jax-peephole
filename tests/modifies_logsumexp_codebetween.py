# Test that we do logsumexp trick if there's code between lines

import peephole
import jax.numpy as np
from jax import make_jaxpr


def test(x):
    a = np.exp(x)
    d = x * 2 # has to be unrelated to a, b, c
    b = np.sum(a)
    c = np.log(b)
    return c


x = np.array(range(5))
ir = make_jaxpr(test)(x)
print(peephole.maybe_peephole_logsumexp_trick(ir))