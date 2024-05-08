import torch as tc
import numpy as np
import time
from scipy.linalg import expm

'''
这里可以将Theta的计算设置为一个类，但是只有两个不等价张量参与计算，所以就直接写了
'''


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


def creat_mps(chi_dim, phy_dim):
    G = tc.randn(2, chi_dim, phy_dim, chi_dim, dtype=tc.float64)
    L = tc.randn(2, chi_dim, dtype=tc.float64)
    return G, L


if __name__ == '__main__':
    start_time = time.time()
    tao = 0.01
    # 截断维数
    chi = 32
    # 自旋
    s = 0.5
    E = tc.tensor([0, 0], dtype=tc.float64)
    # 物理指标
    physics_dim = int(2 * s) + 1

    Gamma, Lambda = creat_mps(chi, physics_dim)

    Ham = creat_Ham(s)

    U = tc.tensor(expm(-1 * tao * Ham.numpy())).reshape(physics_dim, physics_dim,
                                                        physics_dim, physics_dim)
    check_ab = 0
    check_norm = tc.tensor([0, 0], dtype=tc.float64)
    while True:
        a = np.mod(check_ab, 2)
        b = np.mod(check_ab + 1, 2)

        Theta = tc.einsum('aa,aic->aic', tc.diag(Lambda[b]), Gamma[a])
        Theta = tc.einsum('aic,cc->aic', Theta, tc.diag(Lambda[a]))
        Theta = tc.einsum('aid,dje->aije', Theta, Gamma[b])
        Theta = tc.einsum('aije,ee->aije', Theta, tc.diag(Lambda[b]))
        Theta = tc.einsum('aijy,ijcd->cady', Theta, U)

        pp = Theta.norm()
        Psi = tc.einsum('iajy,ijcd->cady', Theta, Ham.reshape(physics_dim, physics_dim,
                                                              physics_dim, physics_dim))
        Psi = tc.einsum('cady, cady-> ', Psi, Theta)
        print("<Eg> = {:.10f}".format(((Psi / (pp.norm() ** 2)).numpy())))

        Theta = Theta.reshape(physics_dim * chi, physics_dim * chi)
        E[a] = -tc.log(Theta.norm() ** 2) / (2 * tao)
        check_norm[a] = Theta.norm()
        print('E = {:.14f}'.format(E[a].numpy()))

        U_svd, Lambda_svd, V = tc.linalg.svd(Theta)

        Lambda[a] = Lambda_svd[0:chi] / Lambda_svd[0:chi].norm()

        X = U_svd[:, 0:chi]
        X = tc.reshape(X, (physics_dim, chi, chi))
        X = tc.permute(X, (1, 0, 2))
        Gamma[a] = tc.einsum('aa,ajk->ajk', tc.diag(1.0 / Lambda[b]), X)

        Y = V[0:chi, :]
        Y = np.reshape(Y, (chi, physics_dim, chi))
        Gamma[b] = tc.einsum('ijk,kk->ijk', Y, tc.diag(1.0 / Lambda[b]))

        check_ab = check_ab + 1

        # 不收敛就改为判断能量,或者熔断
        if tc.abs(check_norm[a] - check_norm[b]) < 1e-12 or check_ab > 19998:
            break

    end_time = time.time()

    print('Eg = ', tc.mean(E).numpy())
    print('Total Steps = ', check_ab + 1)
    execution_time = end_time - start_time
    print("Runtime = {:.4f}s".format(execution_time))
    if s == 0.5:
        print("Error = {:.10f}".format(((tc.mean(E).numpy()) - 0.25 + np.log(2))))
