import math
import numpy as np

def inner_product(u, v):
    return (u * v).sum()

def gram_schmidt(u):
    """Applies the Gram-Schmidt process to a list of vectors
    and returns a new orthogonal set of vectors.

    Args:
        u (List): A list of 1D vectors

    """

    v = []
    for i in range(len(u)):
        n = u[i] - sum(inner_product(v[j], u[i]) / inner_product(v[j], v[j]) * v[j] for j in range(i))
        v.append(n)
        if np.all((n == 0)):
            break
    return v

def orthonormalize(u):
    """Orthonormalizes a set of vectors, i.e. produces a new set of orthogonal
    vectors whose lengths are all unit.

    Args:
        u (List[np.ndarray]): A list of 1D vectors
    """

    V = gram_schmidt(u)
    return [x / math.sqrt(inner_product(x, x)) for x in V]


if __name__ == '__main__':
    U = [np.array([0., 0., 3.]), np.array([0., 1., 0.]), np.array([1., 0., 0.])]
    V = gram_schmidt(U)
    O = orthonormalize(U)

    print(V)
    print(O)
