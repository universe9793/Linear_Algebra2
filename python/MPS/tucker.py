# tucker-decomposition hosvd;
import torch as tc


def hosvd(x):
    ndim = x.ndimension()
    U = list()
    lm = list()
    for n in range(ndim):
        index = list(range(ndim))
        index.pop(n)
        _mat = tc.tensordot(x, x, [index, index])
        _lm, _U = tc.linalg.eigh(_mat)
        U.append(_U)
        lm.append(_lm)
    G = tucker_product(x, U, dim=0)
    # core tensor
    return G, U, lm


def tucker_product(x, U, dim=1):
    # both calculate T and G
    ndim = x.ndimension()
    U1 = list()
    for n in range(len(U)):
        U1.append(U[n])
    for n in range(ndim):
        x = tc.tensordot(U1[n], x, [[dim], [0]])
        x = x.permute(list(range(1, ndim)) + [0])
    return x
