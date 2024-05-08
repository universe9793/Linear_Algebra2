import torch
import time


def heisenberg(L):
    length = 2 ** L

    eye2 = torch.eye(2, dtype=torch.float64)
    H_xy = torch.zeros(length, length, dtype=torch.float64)
    H_z = torch.zeros(length, length, dtype=torch.float64)
    hs_up = torch.tensor([[0, 1], [0, 0]], dtype=torch.float64)
    hs_d = torch.tensor([[0, 0], [1, 0]], dtype=torch.float64)
    hsz = torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.float64)

    for s in range(1, L + 1):
        hs_u = torch.eye(2, dtype=torch.float64)
        hs_down = torch.eye(2, dtype=torch.float64)
        hs_z = torch.eye(2, dtype=torch.float64)

        if s == 1 or s == L or s == L - 1:
            for ss in range(L, 1, -1):
                if s == L and L > 2:
                    if ss == L:
                        hs_u = torch.kron(eye2, hs_up)
                        hs_down = torch.kron(eye2, hs_d)
                        hs_z = torch.kron(eye2, hsz)
                    elif ss == 2:
                        hs_u = torch.kron(hs_d, hs_u)
                        hs_down = torch.kron(hs_up, hs_down)
                        hs_z = torch.kron(hsz, hs_z)
                    else:
                        hs_u = torch.kron(eye2, hs_u)
                        hs_down = torch.kron(eye2, hs_down)
                        hs_z = torch.kron(eye2, hs_z)
                elif s == 2 and L == 2:
                    hs_u = torch.kron(hs_d, hs_up)
                    hs_down = torch.kron(hs_up, hs_d)
                    hs_z = torch.kron(hsz, hsz)
                elif s == L - 1:
                    if ss == L:
                        hs_u = torch.kron(hs_up, hs_d)
                        hs_down = torch.kron(hs_d, hs_up)
                        hs_z = torch.kron(hsz, hsz)
                    else:
                        hs_u = torch.kron(eye2, hs_u)
                        hs_down = torch.kron(eye2, hs_down)
                        hs_z = torch.kron(eye2, hs_z)
                elif s == 1 and L != 2:
                    if ss == 2:
                        hs_u = torch.kron(hs_up, hs_u)
                        hs_down = torch.kron(hs_d, hs_down)
                        hs_z = torch.kron(hsz, hs_z)
                    elif ss == 3:
                        hs_u = torch.kron(hs_d, hs_u)
                        hs_down = torch.kron(hs_up, hs_down)
                        hs_z = torch.kron(hsz, hs_z)
                    else:
                        hs_u = torch.kron(eye2, hs_u)
                        hs_down = torch.kron(eye2, hs_down)
                        hs_z = torch.kron(eye2, hs_z)

            H_xy += 0.5 * hs_u + 0.5 * hs_down
            H_z += hs_z
        else:
            for ss in range(L, 1, -1):
                if ss == s + 2:
                    hs_u = torch.kron(hs_d, hs_u)
                    hs_down = torch.kron(hs_up, hs_down)
                    hs_z = torch.kron(hsz, hs_z)
                elif ss == s + 1:
                    hs_u = torch.kron(hs_up, hs_u)
                    hs_down = torch.kron(hs_d, hs_down)
                    hs_z = torch.kron(hsz, hs_z)
                else:
                    hs_u = torch.kron(eye2, hs_u)
                    hs_down = torch.kron(eye2, hs_down)
                    hs_z = torch.kron(eye2, hs_z)

            H_xy += 0.5 * hs_u + 0.5 * hs_down
            H_z += hs_z
    return H_xy, H_z


if __name__ == "__main__":
    # Example usage:
    t1 = time.time()
    S = 14
    Delta = 1
    Ham_xy, Ham_z = heisenberg(S, Delta)
    t2 = time.time()
    print("Time to compute = ", t2 - t1, "s")
