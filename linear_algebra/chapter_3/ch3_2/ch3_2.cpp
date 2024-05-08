#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <fstream>
#include<eigen3/unsupported/Eigen/KroneckerProduct>
#include <chrono>

using namespace Eigen;
using namespace std;

double calculateTrace(const MatrixXd& matrix) {
    double trace = 0.0;
    for (int i = 0; i < matrix.cols(); ++i) {
        trace += matrix(i, i);
    }
    return trace;
}

void setkets(int count, MatrixXd kets, MatrixXd& U_H, MatrixXd kets_h, MatrixXd& U_HH){
    int i,j;
    for(i=0; i<kets.rows(); ++i){
        for(j=0;j<kets.rows();++j){
            U_H(i+count,j+count) = kets(i,j);
            U_HH(i+count,j+count) = kets_h(i,j);
        }
    }
}

//map拼接，按照相同的Sz的对角元的key，把频率累加值作为key映射到对应的小矩阵
map<int, MatrixXd> combineMaps(const map<double, int>& map1, const map<double, MatrixXd>& map2) {
    map<int, MatrixXd> combinedMap;
    int count = 0;

    for(auto it1 = map1.begin(); it1 != map1.end(); ++it1) {
        combinedMap[count] = map2.at(it1->first);
        count += it1->second;
    }

    return combinedMap;
}

//提取对角元为一个向量
VectorXd extractDiagonal(const MatrixXd& matrix) {
    VectorXd diagonal_elements(matrix.rows());

    for (int i = 0; i < matrix.rows(); ++i) {
        diagonal_elements(i) = matrix(i, i);
    }

    return diagonal_elements;
}

//统计简并数
map<double, int> tabulate(const VectorXd& vector) {
    map<double, int> frequency_map;

    for (int i = 0; i < vector.size(); ++i) {
        frequency_map[vector(i)]++;
    }

    return frequency_map;
}

//构建第一列为对角元，第二列为对应简并维数方阵的map
map<double,MatrixXd> createMatrices(const map<double,int>& frequency_map) {
    map<double,MatrixXd> matrix_map;

    for (const auto& pair : frequency_map) {
        double element = pair.first;
        int frequency = pair.second;
        MatrixXd matrix = MatrixXd::Zero(frequency, frequency);
        matrix_map[element] = matrix;
    }

    return matrix_map;
}

//根据对称性拆解哈密顿矩阵
MatrixXd Symmetry(double dig, const MatrixXd& H, const MatrixXd& Sz) {
    // 标记需要删除的行和列
    //std::vector<bool> to_remove(H.rows(), false);
    VectorXi to_remove = VectorXi::Zero(H.rows());
    for (int i = H.rows() - 1; i >= 0; --i) {
        if (Sz(i,i) != dig) {
            to_remove[i] = true;
        }
    }

    // 统计需要删除的行和列的数量
    int num_removed = std::count(to_remove.begin(), to_remove.end(), true);

    // 创建新的矩阵
    MatrixXd Hs(H.rows() - num_removed, H.cols() - num_removed);

    // 复制保留的部分到新矩阵
    int row_index = 0;
    for (int i = 0; i < H.rows(); ++i) {
        if (!to_remove[i]) {
            int col_index = 0;
            for (int j = 0; j < H.cols(); ++j) {
                if (!to_remove[j]) {
                    Hs(row_index, col_index) = H(i, j);
                    col_index++;
                }
            }
            row_index++;
        }
    }

    return Hs;
}


//构造Sz矩阵
MatrixXd S_z(int L) {
    int length = pow(2, L);
    MatrixXd eye2 = MatrixXd::Identity(2,2);

    MatrixXd hsz(2, 2);
    hsz << 0.5, 0, 
           0, -0.5;

    MatrixXd S_z = MatrixXd::Zero(length, length);
    MatrixXd sz;
    for (int s = 1; s < L + 1; s++) {
        sz =MatrixXd::Identity(2,2);
        for (int ss = L; ss > 1; ss=ss-1){
            if (s == L && ss == L) {
                sz = kroneckerProduct(eye2, hsz).eval();
            } else if (ss == s + 1) {
                sz = kroneckerProduct(hsz, sz).eval();
            } else {
                sz = kroneckerProduct(eye2, sz).eval();
            }

        }
        // 存储sz
        S_z +=sz;
    }
    cout << "-----------S_z-------------" << endl;
    //cout << S_z << endl;
    return S_z;
}

//构造哈密顿量
MatrixXd heisenberg(int L) {
    int length = int(pow(2, L));

    MatrixXd eye2 = MatrixXd::Identity(2, 2);
    MatrixXd H = MatrixXd::Zero(length, length);
    MatrixXd hs_up(2, 2); hs_up << 0, 1,0, 0;
    MatrixXd hs_d(2, 2); hs_d << 0, 0, 1, 0;
    MatrixXd hsz(2, 2); hsz << 0.5, 0, 0, -0.5;

    MatrixXd hs_u, hs_down, hs_z;

    //OBC就改为s<L;
    for (int s = 1; s < L+1; s++) {
        hs_u = MatrixXd::Identity(2, 2);
        hs_down = MatrixXd::Identity(2, 2);
        hs_z = MatrixXd::Identity(2, 2);

        if (s == 1 || s == L || s == L - 1) {
            for (int ss = L; ss > 1; ss--) {
                if (s == L && L > 2) {
                    if (ss == L) {
                        hs_u = kroneckerProduct(eye2, hs_up).eval();
                        hs_down = kroneckerProduct(eye2, hs_d).eval();
                        hs_z = kroneckerProduct(eye2, hsz).eval();
                    }
                    else if (ss == 2) {
                        hs_u = kroneckerProduct(hs_d, hs_u).eval();
                        hs_down = kroneckerProduct(hs_up, hs_down).eval();
                        hs_z = kroneckerProduct(hsz, hs_z).eval();
                    }
                    else {
                        hs_u = kroneckerProduct(eye2, hs_u).eval();
                        hs_down = kroneckerProduct(eye2, hs_down).eval();
                        hs_z = kroneckerProduct(eye2, hs_z).eval();
                    }
                }
                else if (s == 2 && L == 2) {
                    hs_u = kroneckerProduct(hs_d, hs_up).eval();
                    hs_down = kroneckerProduct(hs_up, hs_d).eval();
                    hs_z = kroneckerProduct(hsz, hsz).eval();
                }
                else if (s == L - 1) {
                    if (ss == L) {
                        hs_u = kroneckerProduct(hs_up, hs_d).eval();
                        hs_down = kroneckerProduct(hs_d, hs_up).eval();
                        hs_z = kroneckerProduct(hsz, hsz).eval();
                    }
                    else {
                        hs_u = kroneckerProduct(eye2, hs_u).eval();
                        hs_down = kroneckerProduct(eye2, hs_down).eval();
                        hs_z = kroneckerProduct(eye2, hs_z).eval();
                    }
                }
                else if (s == 1 && L != 2) {
                    if (ss == 2) {
                        hs_u = kroneckerProduct(hs_up, hs_u).eval();
                        hs_down = kroneckerProduct(hs_d, hs_down).eval();
                        hs_z = kroneckerProduct(hsz, hs_z).eval();
                    }
                    else if (ss == 3) {
                        hs_u = kroneckerProduct(hs_d, hs_u).eval();
                        hs_down = kroneckerProduct(hs_up, hs_down).eval();
                        hs_z = kroneckerProduct(hsz, hs_z).eval();
                    }
                    else {
                        hs_u = kroneckerProduct(eye2, hs_u).eval();
                        hs_down = kroneckerProduct(eye2, hs_down).eval();
                        hs_z = kroneckerProduct(eye2, hs_z).eval();
                    }
                }
            }
            H += 0.5*hs_u+0.5*hs_down+hs_z;
        }
        else {
            for (int ss = L; ss > 1; ss--) {
                if (ss == s + 2) {
                    hs_u = kroneckerProduct(hs_d, hs_u).eval();
                    hs_down = kroneckerProduct(hs_up, hs_down).eval();
                    hs_z = kroneckerProduct(hsz, hs_z).eval();
                }
                else if (ss == s + 1) {
                    hs_u = kroneckerProduct(hs_up, hs_u).eval();
                    hs_down = kroneckerProduct(hs_d, hs_down).eval();
                    hs_z = kroneckerProduct(hsz, hs_z).eval();
                }
                else {
                    hs_u = kroneckerProduct(eye2, hs_u).eval();
                    hs_down = kroneckerProduct(eye2, hs_down).eval();
                    hs_z = kroneckerProduct(eye2, hs_z).eval();
                }
            }
            H += 0.5*hs_u+0.5*hs_down+hs_z;
        }
    }

    cout << "-----------PBC_H-----------" << endl;
    //cout << H << endl;

    return H;
}


int main() {
    int L;
    MatrixXd H; MatrixXd Sz;MatrixXd HH;

    cout << "Input L:";
    cin >> L;
    // 获取程序开始执行的时间点
    auto start = std::chrono::steady_clock::now();
    H = heisenberg(L);
    Sz = S_z(L);
    HH = H*H;

    cout<<"---------对称性构造--------"<<endl;
    //构建第一列为对角元，第二列为对应简并维数的map
    VectorXd dig_sz = extractDiagonal(Sz);
    map<double, int> frequency_map = tabulate(dig_sz);

    //构建第一列为Sz对角元，第二列为对应简并维数方阵的map
    map<double,MatrixXd> matrixmap = createMatrices(frequency_map);
    //为HH构建matrixmapHH
    map<double,MatrixXd> matrixmapHH = createMatrices(frequency_map);

    //拆分矩阵,同时构建H和HH的matrixmap
    cout<<"---------拆分矩阵----------"<<endl;
    for (auto& pair : matrixmap) {
        pair.second = Symmetry(pair.first, H, Sz);
        matrixmapHH[pair.first] = Symmetry(pair.first, HH, Sz);
    }

    //构建保存本征矢量的map
    map<int, MatrixXd> eigenkets= combineMaps(frequency_map,matrixmap);
    map<int, MatrixXd> eigenkets_h= combineMaps(frequency_map,matrixmapHH);

    //求解本征问题
    VectorXd eigenvalues;
    VectorXd eigenvalues_h;
    cout<<"---------解本征问题--------"<<endl;
    for (auto& pair : eigenkets) {
        // 计算当前矩阵的本征值和本征右矢
        SelfAdjointEigenSolver<MatrixXd> eigensolver(pair.second);
        SelfAdjointEigenSolver<MatrixXd> eigensolver_h(eigenkets_h[pair.first]);
        if (eigensolver.info() != Success||eigensolver_h.info() !=Success) {
            std::cerr << "Failed to compute eigenvalues for matrix " << pair.first << std::endl;
            continue; // 继续下一个矩阵的计算
        }
        
        // 将本征矢量添加到 eigenkets 中
        pair.second = eigensolver.eigenvectors();
        eigenkets_h[pair.first] = eigensolver_h.eigenvectors();

        // 将本征值添加到 VectorXd 中
        eigenvalues.conservativeResize(eigenvalues.size() + eigensolver.eigenvalues().size());
        eigenvalues.segment(eigenvalues.size() - eigensolver.eigenvalues().size(), eigensolver.eigenvalues().size()) = eigensolver.eigenvalues();
        //HH的本征值
        eigenvalues_h.conservativeResize(eigenvalues_h.size()+eigensolver_h.eigenvalues().size());
        eigenvalues_h.segment(eigenvalues_h.size() - eigensolver_h.eigenvalues().size(), eigensolver_h.eigenvalues().size()) = eigensolver_h.eigenvalues();
    }

    //构造U矩阵
    int length = int(pow(2,L));
    MatrixXd U_H = MatrixXd::Zero(length,length);
    MatrixXd U_HH = MatrixXd::Zero(length,length);

    cout<<"---------构造U矩阵---------"<<endl;
    for(auto& pair : eigenkets){
        setkets(pair.first, eigenkets[pair.first],U_H,eigenkets_h[pair.first], U_HH);
    }

    
    // 设置起始点和结束点
    double startt = 0.01;
    double endt = 2.5;
    // 设置步长
    double step = 0.01;

    cout<<"---------初始化Cv----------"<<endl;
    // 计算向量的大小
    int size = int((endt - startt) / step) + 1;
    
    VectorXd T = VectorXd::Zero(size);
    for(int i=0;i<size;i++)  T(i) = startt+i*0.01;
    VectorXd Cv = VectorXd::Zero(size);

    cout<<"---------计算Cv------------"<<endl;
        
    double Z;
    double trace2;
    double trace3;

    MatrixXd expH = MatrixXd::Zero(length,length);
    MatrixXd DH = MatrixXd::Zero(length,length);
    MatrixXd DH2 = DH;
    VectorXd eigenvalues_1 = eigenvalues;
    
    //预计算HH和H
    MatrixXd DHS = DH;
    DHS.diagonal() = eigenvalues;
    DHS = U_H*DHS*U_H.transpose();

    MatrixXd DHH = DH;
    DHH.diagonal() = eigenvalues_h;
    //这里用幺正对角化形式是因为
    DHH = U_HH*DHH*U_HH.transpose();

    for(int i=0;i<size;++i){
        eigenvalues_1 = eigenvalues/(-1*T(i));
        expH.diagonal() = eigenvalues_1.array().exp();
        expH = U_H*expH*U_H.transpose();
        Z = calculateTrace(expH);
        
        DH = DH*expH;
        trace2 = calculateTrace(DH);
        
        DH2 = DH2*expH;
        trace3 = calculateTrace(DH2);
        
        Cv(i) = (1/(L*T(i)*T(i)))*((trace3/Z)-pow((trace2/Z),2));

        DH = DHS;
        DH2 = DHH;
        expH = MatrixXd::Zero(length,length);
	//cout<<i<<endl;
    }

    //cout<<Cv<<endl;

    cout<<"---------计算完毕----------"<<endl;
    cout<<"---------保存文件----------"<<endl;
    std::ofstream outfile("data.txt"); // 打开一个文件用于写入
    outfile << Cv << endl; // 写入数据到文件
    outfile.close(); // 关闭文件

    auto end = chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "---------Total time--------" << endl << duration.count() << "ms" << endl;

    return 0;
}
