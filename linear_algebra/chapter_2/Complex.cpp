//#ifndef COMPLEX_H
//#define COMPLEX_H
//#define DRAND48_H
//#define mmm 0x100000000LL
//#define ccc 0xB16
//#define aaa 0x5DEECE66DLL
//#define pi 3.14159265358

#include"Complex.h"
double drand48(void)
{
	seed = (aaa * seed + ccc) & 0xFFFFFFFFFFFFLL;
	unsigned int x = seed >> 16;
    return 	((double)x / (double)mmm);
	
}

void srand48(unsigned int i)
{
    seed  = (((long long int)i) << 16) | rand();
}


void printMatrix(Complex *matrix, int n) {
    std::cout.precision(3);
    //精度为3
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

//复矩阵求逆函数
void complexMatrixInverse(Complex *U_xi, int n) {
    // Augment the matrix with an identity matrix
    Complex *augmentedMatrix = new Complex[n * 2 * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmentedMatrix[i * 2 * n + j] = U_xi[i * n + j];
            augmentedMatrix[i * 2 * n + n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; ++i) {
        // Normalize the pivot row
        Complex pivot = augmentedMatrix[i * 2 * n + i];
        for (int j = 0; j < 2 * n; ++j) {
            augmentedMatrix[i * 2 * n + j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                Complex factor = augmentedMatrix[k * 2 * n + i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmentedMatrix[k * 2 * n + j] -= factor * augmentedMatrix[i * 2 * n + j];
                }
            }
        }
    }

    // Copy the inverse matrix back to the original array
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U_xi[i * n + j] = augmentedMatrix[i * 2 * n + n + j];
        }
    }

    delete[] augmentedMatrix;
}

//复向量模长函数
double nrm(const int dimn, const Complex*array){
    double nrm2 = 0;
    for(int dimi=0; dimi<dimn; dimi++){
        nrm2+=real(array[dimi])*real(array[dimi])+imag(array[dimi])*imag(array[dimi]);
    }
    return sqrt(nrm2);
}
//复数加法函数
Complex c_sum(Complex x1,Complex x2){
    return Complex(real(x1)+real(x2),imag(x1)+imag(x2));
}
//复数减法函数
Complex c_sub(Complex x1,Complex x2){
    return Complex(real(x1)-real(x2),imag(x1)-imag(x2));
}

//复数乘法函数
Complex c_times(Complex x1,Complex x2){
    return Complex(real(x1)*real(x2)-imag(x1)*imag(x2),real(x1)*imag(x2)+real(x2)*imag(x1)); 
}
//复矩阵行列式函数
Complex calculateDeterminant(Complex *matrix, int n) {
    // Base case: 1x1 matrix
    if (n == 1) {
        return matrix[0];
    }

    Complex determinant = 0.0;
    Complex *submatrix = new Complex[(n - 1) * (n - 1)];

    // Laplace expansion
    for (int k = 0; k < n; ++k) {
        int sub_i = 0;
        for (int i = 1; i < n; ++i) {
            int sub_j = 0;
            for (int j = 0; j < n; ++j) {
                if (j != k) {
                    submatrix[sub_i * (n - 1) + sub_j] = matrix[i * n + j];
                    ++sub_j;
                }
            }
            ++sub_i;
        }

        // Recursive call to calculate determinant of submatrix
        Complex submatrixDeterminant = calculateDeterminant(submatrix, n - 1);

        // Add or subtract the contribution of the current element
        Complex sign = (k % 2 == 0) ? Complex(1.0, 0.0) : Complex(-1.0, 0.0);
        Complex elementContribution = c_times(matrix[k], submatrixDeterminant);

        determinant = c_sum(c_times(sign, elementContribution),determinant);

    }

    delete[] submatrix;
    return determinant;
}
//复数取共轭
Complex tran(Complex s){
    return Complex(real(s),-1*imag(s));
}
