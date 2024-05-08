# Linear_Algebra2
Linear_Algebra works

前几章用了eigen库，后面统一使用pytorch和libtorch

第二章：Givens矩阵和Householder矩阵；

第三章：精确对角化入门，以一维反铁磁海森堡模型(Heisenberg Model)为例；

第四章：矩阵分解和张量网络入门：SVD分解和一维Infinite-Size Heisenberg Model虚时演化iTEBD，还有待解决的二维方格子的ipeps投影纠缠对态算法与三角晶格(还有kogome lattice)的ipess投影纠缠单态算法；

第五章：Lanczos算法，写Lanczos算法计算12/14格点的PBC海森堡模型基态，并通过基态能量寻找相变点(这个算法写的不如知乎一个大佬的，他在总自旋为0的子空间里面找基态，用二进制运算构造出子空间哈密顿量，直接一步到位就把稀疏矩阵构造出来，然后用lanczos算法去算基态，整个时间很短，需要的内存很小，甚至能算到20格点，我这个算16就爆内存了$\sqrt{3x-1}+(1+x)^2$)；

第六章：密度矩阵重整化群算法(马上写)。
