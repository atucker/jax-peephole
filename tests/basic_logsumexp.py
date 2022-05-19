import peephole
import jax.numpy as np

peephole.DEBUG = False
@peephole.improve
def test(x):
    return np.log(np.sum(np.exp(x)))

print(test(np.array(range(5))))