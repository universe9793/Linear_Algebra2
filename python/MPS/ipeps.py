import torch as tc
import numpy as np
from scipy.linalg import expm


def creat_Ham(S):
    s_up = tc.tensor([0.0, 1.0, 0.0, 0.0], dtype=tc.float64).reshape(2, 2)
    s_dn = tc.tensor([0.0, 0.0, 1.0, 0.0], dtype=tc.float64).reshape(2, 2)
    s_z = 0.5 * tc.tensor([1.0, 0.0, 0.0, -1.0], dtype=tc.float64).reshape(2, 2)
    if S == 1:
        s_up = tc.tensor([0.0, np.sqrt(2), 0.0, 0.0, 0.0, np.sqrt(2), 0.0, 0.0, 0.0],
                         dtype=tc.float64).reshape(3, 3)
        s_dn = tc.tensor([0.0, 0.0, 0.0, np.sqrt(2), 0.0, 0.0, 0.0, np.sqrt(2), 0],
                         dtype=tc.float64).reshape(3, 3)
        s_z = (tc.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                         dtype=tc.float64).reshape(3, 3))

    return 0.5 * (tc.kron(s_up, s_dn) + tc.kron(s_dn, s_up)) + tc.kron(s_z, s_z)


def creat_peps(physics_dim, chi_dim):
    # 1, 2, 3, 4, 5, 6, 7, 8
    lam = [tc.randn(chi_dim, chi_dim), tc.randn(chi_dim, chi_dim),
           tc.randn(chi_dim, chi_dim), tc.randn(chi_dim, chi_dim),
           tc.randn(chi_dim, chi_dim), tc.randn(chi_dim, chi_dim),
           tc.randn(chi_dim, chi_dim), tc.randn(chi_dim, chi_dim)]
    # A, B, C, D
    # physics, left, up, right, down
    gam = [tc.randn(physics_dim, chi_dim, chi_dim, chi_dim, chi_dim),
           tc.randn(physics_dim, chi_dim, chi_dim, chi_dim, chi_dim),
           tc.randn(physics_dim, chi_dim, chi_dim, chi_dim, chi_dim),
           tc.randn(physics_dim, chi_dim, chi_dim, chi_dim, chi_dim)]
    return gam, lam

def creat_tensor(index):


def update(Lambda_update, Gamma_update):


if __name__ == '__main__':
    s = 0.5
    phy_dim = int(s * 2 + 1)
    chi = 5
    Gamma, Lambda = creat_peps(phy_dim, chi)
    Ham = creat_Ham(s)
    tao = 0.01
    U = tc.tensor(expm(-1 * tao * Ham.numpy())).reshape(phy_dim, phy_dim,
                                                        phy_dim, phy_dim)

    check_index = 0
    while True:
        gamma_update = [Gamma[0], Gamma[1]]
        lambda_update = [Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4], Lambda[5], Lambda[6]]