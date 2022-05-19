# Test that we don't logsumexp trick if it's not a logsumexp

import peephole
import jax.numpy as np
from jax import make_jaxpr


def test(x):
    return np.log(x)


x = np.array(range(5))
ir = make_jaxpr(test)(x)
print(peephole.maybe_peephole_logsumexp_trick(ir))