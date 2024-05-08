import torch as tc
import numpy as np

a = tc.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tc.float32)
a = a.reshape(3, 3)
u, s, vh = tc.linalg.svd(a)
print(u,s,vh)
aa = tc.einsum('ij, jk, kl -> il', u, tc.diag(s), vh)
print(aa)