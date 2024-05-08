#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

pair<vector<vector<double>>, vector<double>> diagonalizeMatrix(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    // 将输入矩阵转换为Eigen矩阵
    MatrixXd eigen_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigen_matrix(i, j) = matrix[i][j];
        }
    }

    // 求解本征值和本征向量
    SelfAdjointEigenSolver<MatrixXd> eigensolver(eigen_matrix);
    if (eigensolver.info() != Success) {
        cerr << "本征值求解失败！" << endl;
        exit(1);
    }

    // 获取本征值
    VectorXd eigenvalues = eigensolver.eigenvalues();

    // 将本征值从大到小排列
    vector<double> sorted_eigenvalues;
    for (int i = eigenvalues.size() - 1; i >= 0; --i) {
        sorted_eigenvalues.push_back(eigenvalues(i));
    }

    // 获取对角化后的矩阵
    MatrixXd diagonal_matrix = eigensolver.eigenvalues().asDiagonal();

    // 将对角化后的矩阵转换为vector<vector<double>>类型
    vector<vector<double>> result_matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result_matrix[i][j] = diagonal_matrix(i, j);
        }
    }

    // 返回对角化后的矩阵和本征值
    return make_pair(result_matrix, sorted_eigenvalues);
}

vector<vector<double>> kron(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2){
    int row1 = matrix1.size();
    int col1 = matrix1[0].size();
    int row2 = matrix2.size();
    int col2 = matrix2[0].size();

    vector<vector<double>> size(row1 * row2, vector<double>(col1 * col2));

    for(int i = 0; i < row1; i++)
        for(int j = 0; j < row1; j++)
            for(int k = 0; k < row2; k++)
                for(int l = 0; l < row2; l++)
                    size[i*row2+k][j*row2+l] = matrix1[i][j]*matrix2[k][l];

return size;
}

int check(const vector<vector<double>>& H, const vector<vector<double>>& Sz){
    int is_check = 1;
    int length = H.size();
    vector<vector<double>> hs(length,vector<double>(length,0.0));
    vector<vector<double>> sh(length,vector<double>(length,0.0));

    cout<<"Check begin..."<<endl;
    cout<<"==============="<<endl;
    for (int i=0;i<length; i++) {
        for (int j=0;j<length;j++) {
            for (int k=0;k<length; k++) {
                hs[i][j] += H[i][k] * Sz[k][j];
                sh[i][j] += Sz[i][k] * H[k][j];
            }
        }
    }

    for (int i=0;i<length; i++) {
        if(!is_check) break;
        for (int j=0;j<length;j++) {
            if(abs(hs[i][j]-sh[i][j])>1e-6){
                is_check = 0;
                break;
            }
        }
    }

return is_check;
}

vector<vector<double>> S_z(int L){
    int length = pow(2,L);
    vector<vector<double>> eye2 = {{1,0},{0,1}};
    vector<vector<double>> hsz = {{0.5,0},{0,-0.5}};
    vector<vector<double>>S_z(length,vector<double>(length,0.0));
    vector<vector<double>> sz;

    for(int s=1;s<L+1;s++){
        sz = {{1,0},{0,1}};
        for(int ss=L;ss>1;ss=ss-1){
            if(s==L&&ss==L){
                sz = kron(eye2,hsz);
            }
            else if(ss==s+1){
                sz = kron(hsz,sz);
            }
            else{
                sz = kron(eye2,sz);
            }
        }
        //存储sz
        for(int i=0;i<length;i++){
            for(int j=0;j<length;j++){
                S_z[i][j] += sz[i][j];
            }
        }
    }
    cout<<"S_z矩阵"<<endl;
    for(int i = 0; i < length+1; i++){
        for(int j = 0; j < length; j++){
            if(i<length)
            cout << S_z[i][j] << " ";
            else cout<<"==";
        }
        if(i<length)
        cout << ";"<<endl;
        else cout<<endl;
    }
return S_z;
}

vector<vector<double>> heisenberg(int L){
    int length = int(pow(2,L));

    vector<vector<double>> eye2 = {{1,0},{0,1}};
    vector<vector<double>> H(length, vector<double>(length,0.0));
    vector<vector<double>> hs_up = {{0,1},{0,0}};
    vector<vector<double>> hs_d = {{0,0},{1,0}};
    vector<vector<double>> hsz = {{0.5,0},{0,-0.5}};
    vector<vector<double>> hs_u;vector<vector<double>> hs_down;vector<vector<double>> hs_z;
    
    //PBC
    //for(int s=1;s<L+1;s++){
    //OBC
    for(int s=1;s<L;s++){
        hs_u = {{1,0},{0,1}};
        hs_down = {{1,0},{0,1}};
        hs_z = {{1,0},{0,1}};
        if(s==1||s==L||s==L-1){
            for(int ss=L;ss>1;ss=ss-1){

                if(s==L&&L>2){//尾接头
                    if(ss==L){
                        hs_u = kron(eye2,hs_up);
                        hs_down = kron(eye2,hs_d);
                        hs_z = kron(eye2,hsz);
                    }
                    else if(ss==2){
                        hs_u = kron(hs_d, hs_u);
                        hs_down = kron(hs_up,hs_down);
                        hs_z = kron(hsz,hs_z);
                    }
                    else{
                        hs_u = kron(eye2,hs_u);
                        hs_down = kron(eye2,hs_down);
                        hs_z = kron(eye2,hs_z);
                    }
                }


                else if(s==2&&L==2){
                    hs_u = kron(hs_d,hs_up);
                    hs_down = kron(hs_up,hs_d);
                    hs_z = kron(hsz,hsz);
                }


                else if(s==L-1){//倒数第二个和最后一个
                    if(ss==L){
                        hs_u = kron(hs_up,hs_d);
                        hs_down = kron(hs_d,hs_up);
                        hs_z = kron(hsz,hsz);
                    }
                    else{
                        hs_u = kron(eye2,hs_u);
                        hs_down = kron(eye2,hs_down);
                        hs_z = kron(eye2,hs_z);
                    }
                }


                else if(s==1&&L!=2){
                    if(ss==2){
                        hs_u = kron(hs_up,hs_u);
                        hs_down = kron(hs_d,hs_down);
                        hs_z = kron(hsz,hs_z);
                    }
                    else if(ss==3){
                        hs_u = kron(hs_d,hs_u);
                        hs_down = kron(hs_up,hs_down);
                        hs_z = kron(hsz,hs_z);
                    }
                    else{
                        hs_u = kron(eye2,hs_u);
                        hs_down = kron(eye2,hs_down);
                        hs_z = kron(eye2,hs_z);
                    }
                }
            }
            //存储以上三类情况
            for(int i=0;i<length;i++){
                for(int j=0;j<length;j++){
                    H[i][j] += 0.5*hs_u[i][j]+0.5*hs_down[i][j]+hs_z[i][j];
                }
            }
        }

        else{//非以上三类情况
            for(int ss=L;ss>1;ss=ss-1){
                if(ss==s+2){
                    hs_u = kron(hs_d,hs_u);
                    hs_down = kron(hs_up,hs_down);
                    hs_z = kron(hsz,hs_z);
                }
                else if(ss==s+1){
                    hs_u = kron(hs_up,hs_u);
                    hs_down = kron(hs_d,hs_down);
                    hs_z = kron(hsz,hs_z);
                }
                else{
                    hs_u = kron(eye2,hs_u);
                    hs_down = kron(eye2,hs_down);
                    hs_z = kron(eye2,hs_z);
                }
            }
            for(int i=0;i<length;i++){
                for(int j=0;j<length;j++){
                    H[i][j] += 0.5*hs_u[i][j]+0.5*hs_down[i][j]+hs_z[i][j];
                }
            }
        }
    }
    cout<<"PBC条件下哈密顿矩阵"<<endl;
    for(int i = 0; i < length+1; i++){
        for(int j = 0; j < length; j++){
            if(i<length)
            cout << H[i][j] << " ";
            else cout<<"==";
        }
        if(i<length)
        cout << ";"<<endl;
        else cout<<endl;
    }

return H;
}


int main(){
    int L;int is_check;
    vector<vector<double>> H;
    vector<vector<double>> Sz;

    cout<<"Input L:"<<" ";
    cin >> L;
    H = heisenberg(L);
    Sz = S_z(L);
    is_check = check(H,Sz);
    if(is_check) cout<<"H*S_z = S_z*H 检验通过"<<endl;
    else cout<<"H*S_z = S_z*H 检验未通过"<<endl;

    
    // 对角化矩阵
    auto result = diagonalizeMatrix(H);
    cout<<"==============="<<endl;
    // 输出对角化后的矩阵
    cout << "对角化后的矩阵：" << endl;
    for (const auto& row : result.first) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout <<";"<< endl;
    }

    // 输出本征值
    cout<<"==================="<<endl;
    cout << "本征值（从大到小）：" << endl;
    for (const auto& eigenvalue : result.second) {
        cout << eigenvalue << endl;
    }

    // 计算能隙
    cout<<"============"<<endl;
    double band_gap;
    if (result.second[result.second.size() - 1] == result.second[result.second.size() - 2]) {
        // 最小值存在简并，继续向下找到非简并的最小值
        int i = result.second.size() - 2;
        while (i >= 0 && result.second[i] == result.second[result.second.size() - 2]) {
            --i;
        }
        band_gap = result.second[i-1] - result.second[i];
    } else {
        // 最小值不存在简并，直接计算能隙
        band_gap = result.second[result.second.size() - 2] - result.second[result.second.size() - 1];
    }
    
    cout << "能隙：" << band_gap << endl;

return 0;
}


