# Test that we do logsumexp trick if there's code after

import peephole
import jax.numpy as np
from jax import make_jaxpr


@peephole.improve
def test(x):
    out = np.log(np.sum(np.exp(x)))
    return 2 * out


print(test(np.array(range(5))))