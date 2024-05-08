import torch as tc

import tucker

test = tc.randn(3, 4, 2, dtype=tc.float64)
Core, V, Lm = tucker.hosvd(test)

tensor1 = tucker.tucker_product(Core, V, dim=0)
error = tc.norm(test - tensor1)
print('error = ' + str(error))

print(V[0].mm(V[0].t()))
print(V[1].mm(V[1].t()))
print(V[2].mm(V[2].t()))

