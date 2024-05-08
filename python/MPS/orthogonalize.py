import torch
import torch as tc
import numpy as np


def orthogonalize(A):
    AL = A / A.norm()
    AR = A / A.norm()
    dim = A.size()

    while True:
        while True:
            AL0 = AL.reshape(-1, AL.size(-1))
            AR0 = AR.reshape(-1, AR.size(0))

            UL, Lambda, VL = tc.linalg.svd(AL0, full_matrices=False)
            LambdaL = torch.diag(Lambda)

            AL = tc.einsum('ia,ab,bjk->ijk', LambdaL, VL, A)
            AL = AL / AL.norm()

            UR, Lambda, VR = tc.linalg.svd(AR0, full_matrices=False)
            LambdaR = torch.diag(Lambda)

            AR = tc.einsum('ia,ab,bjk->ijk', VR, LambdaR, A)
            AR = AR / AR.norm()

            AL0 = AL0.reshape(dim)
            AR0 = AR0.reshape(dim)

            error1 = tc.norm(tc.abs(AL) - tc.abs(AL0))
            error2 = tc.norm(tc.abs(AR) - tc.abs(AR0))
            if error1.item() < 1e-8 and error2.item() < 1e-8:
                break
        error = A
        A = tc.einsum('ia,ab,bjc,cd,dk->ijk', LambdaL, VL, A, VR, LambdaR)
        error = tc.norm(error-A)
        if error.item() < 1e-8:
            break
        print(error)
    return [A, AL, AR]


tensor = torch.randn(10, 9, 10, dtype=tc.float64)
tensor = orthogonalize(tensor)

