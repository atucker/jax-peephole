# Test that we do logsumexp trick if there's code after

import peephole
import jax.numpy as np
from jax import make_jaxpr

def test(x):
    out = np.log(np.sum(np.exp(x)))
    return 2 * out


x = np.array(range(5))
ir = make_jaxpr(test)(x)
print(peephole.maybe_peephole_logsumexp_trick(ir) is not None)