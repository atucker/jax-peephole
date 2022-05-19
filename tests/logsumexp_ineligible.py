# Test that we don't logsumexp trick if the variables get used elsewhere

import peephole
import jax.numpy as np
from jax import make_jaxpr


def test(x):
    b = np.sum(np.exp(x))
    c = np.exp(b)
    return np.log(b)


x = np.array(range(5))
ir = make_jaxpr(test)(x)
print(peephole.maybe_peephole_logsumexp_trick(ir))