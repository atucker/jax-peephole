# Test that we do trick if there's code before

import peephole
import jax.numpy as np
from jax import make_jaxpr

def test(x):
    x = 2 * x
    return np.log(np.sum(np.exp(x)))


x = np.array(range(5))
ir = make_jaxpr(test)(x)
print(peephole.maybe_peephole_logsumexp_trick(ir) is not None)