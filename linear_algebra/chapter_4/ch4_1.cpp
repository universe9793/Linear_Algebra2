#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <chrono>
#include <random>
using namespace std;
using namespace Eigen;

const int M = 300;
const int N = 200;

bool isZeroMatrix(const MatrixXcd& matrix, double tolerance) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (abs(matrix(i, j).real())>tolerance||abs(matrix(i, j).imag())>tolerance){
                return false;
            }
        }
    }
    return true;
}
MatrixXcd generateRandomComplexMatrix(int m, int n) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);

    MatrixXcd m_Random(m, n);
    // 使用随机数引擎生成随机复数矩阵
    for (int i = 0; i < m_Random.rows(); ++i) {
        for (int j = 0; j < m_Random.cols(); ++j) {
            uniform_real_distribution<double> distribution(-1.0, 1.0);
            double real_part = distribution(generator);
            double imag_part = distribution(generator);
            m_Random(i, j) = std::complex<double>(real_part, imag_part);
        }
    }
    return m_Random;
}
int main() {
    cout << std::setprecision(4);
    auto start = std::chrono::high_resolution_clock::now();

    MatrixXcd A = generateRandomComplexMatrix(M,N);
    MatrixXcd A_dagger = A.adjoint();
    MatrixXcd H = A_dagger*A;
    ComplexEigenSolver<Eigen::MatrixXcd> solver(H);

    //calculate the singular_values
    VectorXd singular_values = solver.eigenvalues().array().sqrt().real();

    //build matrix omega
    //calculate the count of singular_values (r)
    singular_values = singular_values.array().real().cwiseMax(0);
    Index count_sv = (singular_values.array() > 1e-2).count();

    MatrixXd Omega = MatrixXd::Zero(M, N);
    for(int i=0;i<min(M,N);++i){
        Omega(i,i) = singular_values(N-1-i);
    }
    //build V
    MatrixXcd V = solver.eigenvectors().rowwise().reverse();

    //build U
    MatrixXcd U = generateRandomComplexMatrix(M,M);
    for (int i = 0; i < int(count_sv); ++i) {
        U.col(i) = A*V.col(i)/Omega(i,i);
    }

    //extended the U matrix using schmidt process
    if(count_sv < M ){
        VectorXcd vec_schmidt= VectorXcd::Zero(M, 1);
        for(int i=int(count_sv);i<M;++i){
            vec_schmidt= VectorXcd::Zero(M, 1);
            for(int j=0;j<i;++j){
                vec_schmidt = vec_schmidt + (U.col(j).dot(U.col(i)))/
                        (U.col(j).dot(U.col(j)))*U.col(j);
            }
            U.col(i) -= vec_schmidt;
            U.col(i) = U.col(i)/U.col(i).norm();
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout<< "Total time: "<<duration.count()<<"ms"<<endl;

    //verification
    cout<<"=== Verification ==="<<endl;
    MatrixXcd error = A-U*Omega*V.adjoint();
    if (isZeroMatrix(error, 1e-5)) {
        cout << "=== A=U*Omega*V' ===" << endl;
    } else {
        cout << "=== A!=U*Omega*V' " << endl;
    }

    //U
    error = U.adjoint()-U.inverse();
    complex<double> det = U.determinant();
    if (isZeroMatrix(error, 1e-5) && abs(1-abs(det))<1e-5) {
        cout << "=== U^(-1)=U' & |det(U)|=1 ===" << endl;
    } else {
        cout << "=== U^(-1)!=U' " << endl;
    }

    //V
    error = V.inverse()-V.adjoint();
    det = V.determinant();
    if (isZeroMatrix(error, 1e-5) && abs(1-abs(det))<1e-5) {
        cout << "=== V^(-1)=V' & |det(V)|=1 ===" << endl;
    } else {
        cout << "=== V^(-1)!=V' " << endl;
    }

    //singular_value>0;
    for(int i=0;i<min(M,N);++i){
        if(Omega(i,i)<0){
            cout<<"=== singular_values<0 ==="<<endl;
            return 0;
        }
    }

    cout<<"=== singular_values>0 ==="<<endl;
    return 0;
}
