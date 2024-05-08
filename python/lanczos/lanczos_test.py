import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import eigh_tridiagonal as tqli
from scipy.linalg import norm
import time


def generate_map():
    matrix_basis = []
    basis_matrix = {}
    k1 = -1
    for j in range(2 << (n - 1)):
        S = sum((j >> k) & 1 for k in range(n))
        if S == n // 2:
            k1 += 1
            matrix_basis.append(j)
            basis_matrix[j] = k1
    return matrix_basis, basis_matrix


def generate_hm():
    hm = sparse.lil_matrix((matrix_dim, matrix_dim))
    print(matrix_dim)
    for j in range(matrix_dim):
        j1 = matrix_basis[j]
        hm[j, j] = sum((float((j1 >> k) & 1) - 1 / 2) * (float((j1 >> ((k + 1) % n)) & 1) - 1 / 2) for k in range(n))
        for k in range(n):
            if ((j1 >> k) & 1) + ((j1 >> ((k + 1) % n)) & 1) == 1:
                j2 = basis_matrix[j1 ^ (1 << k) ^ (1 << ((k + 1) % n))]
                hm[j, j2] = 1 / 2
    return hm


def lanczos(hm, matrix_dim):
    k_diag = []
    k_sub_diag = []
    v0 = np.random.rand(matrix_dim)
    v0 = 1 / norm(v0) * v0
    v1 = hm @ v0
    k = 0
    b1 = 1
    while True:
        print(">>>Lanczos Step", k)
        a0 = v0 @ v1
        k_diag.append(a0)
        if k > 29:
            eig_value, eig_vector = tqli(np.array(k_diag), np.array(k_sub_diag), select='i', select_range=(0, 0))
            if abs(b1 * eig_vector[k]) < 1e-10:
                print(">>>lanczos Finished ")
                break
        k += 1
        v1 = -a0 * v0 + v1
        b1 = norm(v1)
        k_sub_diag.append(b1)
        v1 = 1 / b1 * v1
        v0, v1 = v1, -b1 * v0
        v1 = hm @ v0 + v1 # 因为v1 = -b1*v0与被赋值为v1的v0正交，所以计算a0的时候可以这一项是0
    # eig_value = np.double(eig_value)
    return eig_value


if __name__ == '__main__':
    n = int(input())
    print("Heisenberg Chain,n=", n)
    t1 = time.time()
    matrix_basis, basis_matrix = generate_map()
    matrix_dim = np.size(matrix_basis)
    hm = generate_hm()
    eig_value = lanczos(hm, matrix_dim)
    t2 = time.time()
    print("Ground State Energy=", eig_value)
    print("Time Spent=", t2 - t1, "s")
