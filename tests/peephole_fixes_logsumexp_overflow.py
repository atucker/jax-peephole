import peephole
import jax.numpy as np

peephole.DEBUG = False


def test(x):
    return np.log(np.sum(np.exp(x)))


print(test(1000*np.array(range(5))))
print(peephole.improve(test)(1000*np.array(range(5))))