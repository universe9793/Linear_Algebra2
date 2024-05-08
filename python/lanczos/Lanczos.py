import torch as tc
import numpy as np
import time


def heisenberg(L):
    length = 2 ** L

    eye2 = tc.eye(2, dtype=tc.float64)
    H_xy = tc.zeros(length, length, dtype=tc.float64)
    H_z = tc.zeros(length, length, dtype=tc.float64)
    hs_up = tc.tensor([[0, 1], [0, 0]], dtype=tc.float64)
    hs_d = tc.tensor([[0, 0], [1, 0]], dtype=tc.float64)
    hsz = tc.tensor([[0.5, 0], [0, -0.5]], dtype=tc.float64)

    for s_index in range(1, L + 1):
        hs_u = tc.eye(2, dtype=tc.float64)
        hs_down = tc.eye(2, dtype=tc.float64)
        hs_z = tc.eye(2, dtype=tc.float64)

        if s_index == 1 or s_index == L or s_index == L - 1:
            for ss in range(L, 1, -1):
                if s_index == L and L > 2:
                    if ss == L:
                        hs_u = tc.kron(eye2, hs_up)
                        hs_down = tc.kron(eye2, hs_d)
                        hs_z = tc.kron(eye2, hsz)
                    elif ss == 2:
                        hs_u = tc.kron(hs_d, hs_u)
                        hs_down = tc.kron(hs_up, hs_down)
                        hs_z = tc.kron(hsz, hs_z)
                    else:
                        hs_u = tc.kron(eye2, hs_u)
                        hs_down = tc.kron(eye2, hs_down)
                        hs_z = tc.kron(eye2, hs_z)
                elif s_index == 2 and L == 2:
                    hs_u = tc.kron(hs_d, hs_up)
                    hs_down = tc.kron(hs_up, hs_d)
                    hs_z = tc.kron(hsz, hsz)
                elif s_index == L - 1:
                    if ss == L:
                        hs_u = tc.kron(hs_up, hs_d)
                        hs_down = tc.kron(hs_d, hs_up)
                        hs_z = tc.kron(hsz, hsz)
                    else:
                        hs_u = tc.kron(eye2, hs_u)
                        hs_down = tc.kron(eye2, hs_down)
                        hs_z = tc.kron(eye2, hs_z)
                elif s_index == 1 and L != 2:
                    if ss == 2:
                        hs_u = tc.kron(hs_up, hs_u)
                        hs_down = tc.kron(hs_d, hs_down)
                        hs_z = tc.kron(hsz, hs_z)
                    elif ss == 3:
                        hs_u = tc.kron(hs_d, hs_u)
                        hs_down = tc.kron(hs_up, hs_down)
                        hs_z = tc.kron(hsz, hs_z)
                    else:
                        hs_u = tc.kron(eye2, hs_u)
                        hs_down = tc.kron(eye2, hs_down)
                        hs_z = tc.kron(eye2, hs_z)

            H_xy += 0.5 * hs_u + 0.5 * hs_down
            H_z += hs_z
        else:
            for ss in range(L, 1, -1):
                if ss == s_index + 2:
                    hs_u = tc.kron(hs_d, hs_u)
                    hs_down = tc.kron(hs_up, hs_down)
                    hs_z = tc.kron(hsz, hs_z)
                elif ss == s_index + 1:
                    hs_u = tc.kron(hs_up, hs_u)
                    hs_down = tc.kron(hs_d, hs_down)
                    hs_z = tc.kron(hsz, hs_z)
                else:
                    hs_u = tc.kron(eye2, hs_u)
                    hs_down = tc.kron(eye2, hs_down)
                    hs_z = tc.kron(eye2, hs_z)

            H_xy += 0.5 * hs_u + 0.5 * hs_down
            H_z += hs_z
    return H_xy.to_sparse(), H_z.to_sparse()


def lanczos(ham, matrix_dim, S):
    f_n = tc.randn(matrix_dim, dtype=tc.float64)
    f_n = 1 / f_n.norm() * f_n
    f_nn = tc.matmul(ham, f_n)
    k = 0
    energy = 1
    diag = []
    diag_sub = []
    while True:  # 计算顺序 h11, h21, h22, h22, h32, h23, h33 ...
        # 这里计算对角元
        diag_h = tc.einsum('i, i', f_n, f_nn)
        diag.append(diag_h)
        if k > int(3 * S):
            n = len(diag)
            K = tc.zeros(n, n, dtype=tc.float64)
            K += tc.diag(tc.tensor(diag))
            K += tc.diag(tc.tensor(diag_sub), diagonal=1)
            eigenvalues = tc.linalg.eigvalsh(K, UPLO='U')

            if abs(energy - eigenvalues[0]) < 1e-10:
                break
            energy = eigenvalues[0]
        k += 1
        # 这里计算Fn+1在上一次循环中，已将-sub_diag_h * f_n-1减去,所以下一步直接减去hnn*f_n就行
        f_nn = f_nn - diag_h * f_n
        sub_diag_h = f_nn.norm()
        # 这里是计算下一行的非对角元
        diag_sub.append(sub_diag_h)
        f_nn = 1 / sub_diag_h * f_nn
        # 把Fn+1转为fn+1
        f_n, f_nn = f_nn, -sub_diag_h * f_n
        f_nn = tc.matmul(ham, f_n) + f_nn
        # 接下来要计算h_n+1 n+1，本应该是f_n+1和Hf_n+1做内积
        # 但是为了代码运算，这里加上了-sub_diag_h * f_n但是无所谓，在做内积时，f_n已经被赋值为f_n+1他们俩正交，这一项为0
    return eigenvalues[0]


if __name__ == '__main__':
    s = 12
    dim = 2 ** s
    Delta = tc.arange(-1.5, 1.55, 0.05)
    print("Creat Ham_Heisenberg")
    Ham_xy, Ham_z = heisenberg(s)
    E = np.zeros(Delta.size(0))
    t1 = time.time()
    for i in range(Delta.size(0)):
        groundstate_energy = lanczos(Ham_xy + Delta[i] * Ham_z, dim, s)
        # 将哈密顿量保存为稀疏矩阵格式，然后在循环内做乘法，速度能提升十几倍
        print("{:.2f}".format(Delta[i].numpy()), "\t", "{:.6f}".format(groundstate_energy.numpy()))
        E[i] = groundstate_energy
    t2 = time.time()
    print("Total Time = {:.6}s".format(t2 - t1))
    print("Each Lanczos Step Time = {:.4}ms".format(1000*(t2 - t1)/60))
