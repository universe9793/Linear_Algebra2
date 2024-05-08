#define COMPLEX_H
#define DRAND48_H
#define mmm 0x100000000LL
#define ccc 0xB16
#define aaa 0x5DEECE66DLL
#define pi 3.14159265358
#include<iostream>
#include <stdlib.h>
#include<complex.h>

using namespace std;
typedef complex<double> Complex;
static unsigned long long seed = 1;

double drand48(void);
void srand48(unsigned int i);
void printMatrix(Complex *matrix, int n);
void complexMatrixInverse(Complex *U_xi, int n);
//复向量模长函数
double nrm(const int dimn, const Complex*array);
//复数加法函数
Complex c_sum(Complex x1,Complex x2);
//复数减法函数
Complex c_sub(Complex x1,Complex x2);

//复数乘法函数
Complex c_times(Complex x1,Complex x2);
//复矩阵行列式函数
Complex calculateDeterminant(Complex *matrix, int n);

Complex tran(Complex s);
