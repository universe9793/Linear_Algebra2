import numpy as np
import torch as tc
import pandas as pd
from scipy.linalg import expm


def itebdForCalculateHessenbergModelGroundStateEnergy(S, chi, T):
    deltaT = 0.01
    loopTimes = int(T / deltaT)

    phy_dim = int(S * 2) + 1
    Gama = np.random.rand(2, chi, phy_dim, chi)
    Lambda = np.random.rand(2, chi)

    H = creat_Ham(S).numpy()
    U = expm(-deltaT * H).reshape(phy_dim, phy_dim, phy_dim, phy_dim)
    E = 0

    for i in range(loopTimes):
        A = np.mod(i, 2)
        B = np.mod(i + 1, 2)

        Theta = np.tensordot(np.diag(Lambda[B, :]), Gama[A, :, :, :], axes=(1, 0))
        Theta = np.tensordot(Theta, np.diag(Lambda[A, :]), axes=(2, 0))
        Theta = np.tensordot(Theta, Gama[B, :, :, :], axes=(2, 0))
        Theta = np.tensordot(Theta, np.diag(Lambda[B, :]), axes=(3, 0))
        Theta = np.tensordot(Theta, U, axes=((1, 2), (0, 1)))

        Theta = np.reshape(np.transpose(Theta, (2, 0, 3, 1)), (chi * phy_dim, phy_dim * chi))

        print("Loop Times:", i, "E =", -np.log(np.sum(Theta ** 2)) / (deltaT * 2))
        X, newLambda, Y = np.linalg.svd(Theta)
        Lambda[A, :] = newLambda[0:chi] / np.sqrt(np.sum(newLambda[0:chi] ** 2))

        X = X[0:phy_dim * chi, 0:chi]
        X = np.reshape(X, (phy_dim, chi, chi))
        X = np.transpose(X, (1, 0, 2))
        Gama[A, :, :, :] = np.tensordot(np.diag(Lambda[B, :] ** (-1)), X, axes=(1, 0))

        Y = Y[0:chi, 0:phy_dim * chi]
        Y = np.reshape(Y, (chi, phy_dim, chi))
        Gama[B, :, :, :] = np.tensordot(Y, np.diag(Lambda[B, :] ** (-1)), axes=(2, 0))

        if i >= loopTimes - 2:
            E += -np.log(np.sum(Theta ** 2)) / (deltaT * 2)

    return E / 2


def creat_Ham(S):
    if S == 0.5:
        s_up = tc.tensor([0, 1, 0, 0], dtype=tc.float64).reshape(2, 2)
        s_dn = tc.tensor([0, 0, 1, 0], dtype=tc.float64).reshape(2, 2)
        s_z = 0.5 * tc.tensor([1, 0, 0, -1], dtype=tc.float64).reshape(2, 2)
    elif S == 1:
        s_up = tc.tensor([0, np.sqrt(2), 0, 0, 0, np.sqrt(2), 0, 0, 0], dtype=tc.float64).reshape(3, 3)
        s_dn = tc.tensor([0, 0, 0, np.sqrt(2), 0, 0, 0, np.sqrt(2), 0], dtype=tc.float64).reshape(3, 3)
        s_z = tc.tensor([1, 0, 0, 0, 0, 0, 0, 0, -1], dtype=tc.float64).reshape(3, 3)

    return 0.5 * (tc.kron(s_up, s_dn) + tc.kron(s_dn, s_up)) + tc.kron(s_z, s_z)


e = itebdForCalculateHessenbergModelGroundStateEnergy(0.5, 30, 200)
print(e)
