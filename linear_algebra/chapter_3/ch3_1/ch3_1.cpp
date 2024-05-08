#include <iostream>
#include <Eigen/Dense>
#include <map>
#include<eigen3/unsupported/Eigen/KroneckerProduct>
#include <chrono>

using namespace Eigen;
using namespace std;

//验证H和Sz的对易关系
bool check(const MatrixXd& H,const MatrixXd& Sz){
     MatrixXd HS = H*Sz;
     cout<<"-----------HSz-------------"<<endl;
     MatrixXd SH = Sz*H;
     cout<<"-----------SzH-------------"<<endl;

     return HS.isApprox(SH);
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
    int L;bool is_check;
    MatrixXd H; MatrixXd Sz;

    cout << "Input L:";
    cin >> L;
    // 获取程序开始执行的时间点
    auto start = std::chrono::steady_clock::now();
    H = heisenberg(L);
    Sz = S_z(L);

    cout<<"-----------Check-----------"<<endl;
    
    is_check = check(H,Sz);
    if (is_check)  cout << "[H,Sz] = 0" << std::endl;
    else cout << "[H,Sz] != 0" << std::endl;
    

    cout<<"---------对称性构造--------"<<endl;
    //构建第一列为对角元，第二列为对应简并维数的map
    VectorXd dig_sz = extractDiagonal(Sz);
    map<double, int> frequency_map = tabulate(dig_sz);

    /*std::cout << "Element : Frequency" << std::endl;
    for (const auto& pair : frequency_map) {
        cout << pair.first << " : " << pair.second << std::endl;
    }*/

    //构建第一列为对角元，第二列为对应简并维数方阵的map
    map<double,MatrixXd> matrixmap = createMatrices(frequency_map);

    /*std::cout << "Element : Matrix" << std::endl;
    for (const auto& pair : matrixmap) {
        cout << pair.first << " :\n\n" << pair.second <<"\n"<<endl;
    }*/

    //拆分矩阵
    cout<<"---------拆分矩阵----------"<<endl;
    for (auto& pair : matrixmap) {
        pair.second = Symmetry(pair.first, H, Sz);
        //cout<< pair.first<<endl;
        //cout << pair.second <<"\n"<< std::endl;
    }

    //整理本征值
    VectorXd eigenvalues;

    cout<<"---------整理本征值--------"<<endl;
    for (const auto& pair : matrixmap) {
        // 计算当前矩阵的本征值
        SelfAdjointEigenSolver<MatrixXd> eigensolver(pair.second);
        if (eigensolver.info() != Success) {
            std::cerr << "Failed to compute eigenvalues for matrix " << pair.first << std::endl;
            continue; // 继续下一个矩阵的计算
        }
        
        // 将本征值添加到 VectorXd 中
        eigenvalues.conservativeResize(eigenvalues.size() + eigensolver.eigenvalues().size());
        eigenvalues.segment(eigenvalues.size() - eigensolver.eigenvalues().size(), eigensolver.eigenvalues().size()) = eigensolver.eigenvalues();
    }

    // 打印所有本征值
    sort(eigenvalues.data(), eigenvalues.data() + eigenvalues.size(),greater<double>());
    //cout << "Eigenvalues: \n" << eigenvalues << std::endl;

    // 计算能隙
    cout<<"---------计算能隙----------"<<endl;
    double band_gap;
    if (eigenvalues[eigenvalues.size() - 1] == eigenvalues[eigenvalues.size() - 2]) {
        // 最小值存在简并，继续向下找到非简并的最小值
        int i = eigenvalues.size() - 2;
        while (i >= 0 && eigenvalues[i] == eigenvalues[eigenvalues.size() - 2]) {
            --i;
        }
        band_gap = eigenvalues[i-1] - eigenvalues[i];
    } else {
        // 最小值不存在简并，直接计算能隙
        band_gap = eigenvalues[eigenvalues.size() - 2] - eigenvalues[eigenvalues.size() - 1];
    }
    
    cout <<"---------能隙--------------" << endl<< band_gap << endl;
    // 获取程序结束执行的时间点
    auto end = std::chrono::steady_clock::now();

    // 计算程序的运行时间，单位为毫秒
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 打印程序运行时间
    std::cout << "---------Total time--------" << endl << duration.count() << "ms" << endl;

    return 0;
}
