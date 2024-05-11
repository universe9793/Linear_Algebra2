import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import time
from video_ising import Ising_Video


class MarkovMCIsing:

    def __init__(self, length, J, T, IsPlot):
        self.IsPlot = IsPlot
        self.L = length
        self.J = J
        self.Beta = 1 / T
        self.MC_Steps = int(0.5 * 120000 * 0.1 * self.L)
        self.Lattice = tc.randint(0, 2, (self.L, self.L), dtype=tc.int)
        self.Lattice[self.Lattice == 0] = -1
        self.Neighbor_Points = tc.ones(4)
        self.E = []
        self.Mag = []

    def IsingVideo(self):
        Ising_Video(self.J)

    def RandomPoint(self):
        return tc.randint(0, self.L, (1,)).item()

    def Neighbor_Lattice(self, i, j):
        self.Neighbor_Points[0] = self.Lattice[i - 1, j]
        self.Neighbor_Points[1] = self.Lattice[i, j - 1]
        self.Neighbor_Points[2] = self.Lattice[(i + 1) % self.L, j]
        self.Neighbor_Points[3] = self.Lattice[i, (j + 1) % self.L]
        return self.Neighbor_Points

    def DeltaEnergy(self, i, j):
        return -2 * self.J * self.Lattice[i, j].item() * tc.sum(self.Neighbor_Lattice(i, j)).item()

    def MetropolisAccessPossibility(self, i, j):
        return np.exp(-1 * self.Beta * self.DeltaEnergy(i, j))

    def AvgMag(self):
        self.Mag.append(np.abs(tc.sum(self.Lattice).item()) / self.L ** 2)

    def AvgEnergy(self):
        self.E.append((self.J * (tc.sum(self.Lattice[:-1, :] * self.Lattice[1:, :])
                                 + tc.sum(self.Lattice[:, :-1] * self.Lattice[:, 1:]))
                       + self.J * (tc.sum(self.Lattice[0, :] * self.Lattice[-1, :])
                                   + tc.sum(self.Lattice[:, 0] * self.Lattice[:, -1]))) / self.L ** 2)

    def SaveImg(self, steps):
        # 保存图像到文件
        plt.figure(figsize=(12.4, 9.6))
        plt.imshow(self.Lattice, cmap='binary')
        plt.axis('off')
        plt.savefig(f"img/{steps}.png")
        plt.close()
        print(steps)

    def QuantumMC(self):
        self.AvgMag()
        self.AvgEnergy()
        img_steps = 1
        for n in range(self.MC_Steps):
            i, j = self.RandomPoint(), self.RandomPoint()
            if self.MetropolisAccessPossibility(i, j) > tc.rand(1).item():
                self.Lattice[i, j] = -1 * self.Lattice[i, j]
                self.AvgEnergy()
                self.AvgMag()
            if self.IsPlot and (n % 278 == 1 or n == 0):
                self.SaveImg(img_steps)
                img_steps += 1
        if self.IsPlot:
            Ising_Video(self.J)
        return self.Lattice


if __name__ == '__main__':
    Ising_4 = MarkovMCIsing(200, 1, 1.5, 1)

    plt.figure(figsize=(12.4, 9.6))
    plt.imshow(Ising_4.Lattice, cmap='binary')
    plt.axis('off')
    plt.show()

    t1 = time.time()
    Ising_4.QuantumMC()
    t2 = time.time()
    print('{:.3f}s'.format(t2-t1))

    mc = plt.figure(figsize=(12.4, 9.6))
    plt.imshow(Ising_4.Lattice, cmap='binary')
    plt.axis('off')
    mc.show()

    plt.figure(figsize=(12.4, 9.6))
    plt.plot(Ising_4.Mag)
    plt.plot(Ising_4.E)
    plt.xlabel("MC_Steps")
    plt.ylabel("Energy/Magnetization")
    plt.show()
