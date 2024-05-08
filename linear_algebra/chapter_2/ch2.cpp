#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "Complex.h"
#include <chrono>

//household变换
void Householder(int n, Complex*yita, const Complex*xi, Complex*U){
    Complex E[n][n];
    Complex w[n];
    Complex yita_h[n];
    Complex xi_h[n];
    int i,j;
    Complex phase(0,0);

    for(i=0;i<n;i++){
        yita_h[i]=yita[i];xi_h[i]=xi[i];
        //计算出两个向量内积
        phase += c_times(tran(yita[i]),xi[i]); 
        for(j=0;j<n;j++){
            E[i][j] = Complex(0,0);
            if(i==j)E[i][j] = Complex(1,0);
        }
    }

    //计算内积的模，得到e^{i\theta}
    double length = sqrt(real(phase)*real(phase)+imag(phase)*imag(phase));
    phase = Complex(real(phase)/length,imag(phase)/length);
    for(i=0;i<n;i++) w[i] = c_times(phase, yita_h[i])-xi_h[i];
    length = nrm(n, w);
    for(i=0;i<n;i++)w[i] = Complex(real(w[i])/length, imag(w[i])/length);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            U[i*n+j] = c_times(phase,E[i][j]-c_times(Complex(2,0),c_times(w[i],tran(w[j]))));
        }
    }

}
//givens变换
void Givens(int n, const Complex*yita, const Complex*xi, Complex*U_yita, Complex*U_xi){
    int gi,gj;
    Complex c_theta;
    Complex s_theta;
    Complex yita_s[2];
    Complex xi_s[2];
    Complex g[n][n];
    Complex g_xi[n][n];
    Complex u_g[n][n];
    Complex u_gxi[n][n];
    
    for(int s = 0;s<n;s++){
        for(int ss = 0;ss<n;ss++){
            g[s][ss] = Complex(0,0);
            g_xi[s][ss] = Complex(0,0);
            if (s==ss){ 
                g[s][ss] = Complex(1,0);
                g_xi[s][ss] = Complex(1,0);
            }
        }
    }
    yita_s[0] = yita[0];
    xi_s[0] = xi[0];
    gi=0;
    for(gj=1; gj<n; gj++){
        yita_s[1] = yita[gj];
        xi_s[1] = xi[gj];
        
        //计算c和s
        c_theta = conj(yita_s[0])/(nrm(2,yita_s));
        s_theta = conj(yita_s[1])/(nrm(2,yita_s));
            
        //构造g矩阵为当前分量对应的givens矩阵
        g[gi][gi] = c_theta;g[gi][gj] = s_theta;
        g[gj][gi] = Complex(-1*(real(conj(s_theta))),-1*(imag(conj(s_theta))));g[gj][gj] = conj(c_theta);

        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                //u_g存储Givens矩阵连乘后的结果
                u_g[i][j] = U_yita[i*n+j];
                //U矩阵作为中间变量，待会存储g和u_g相乘的结果
                U_yita[i*n+j] = Complex(0,0);
            }
        }

        c_theta = conj(xi_s[0])/(nrm(2,xi_s));
        s_theta = conj(xi_s[1])/(nrm(2,xi_s));
            
        //构造g矩阵为当前分量对应的givens矩阵
        g_xi[gi][gi] = c_theta;g_xi[gi][gj] = s_theta;
        g_xi[gj][gi] = Complex(-1*(real(conj(s_theta))),-1*(imag(conj(s_theta))));g_xi[gj][gj] = conj(c_theta);

        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                //u_g存储连乘的结果
                u_gxi[i][j] = U_xi[i*n+j];
                //U矩阵作为中间变量，待会存储g和u_g相乘的结果
                U_xi[i*n+j] = Complex(0,0);
            }
        }

        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                for (int k = 0; k < n; k++) {
                    U_yita[i*n+j] = c_sum(U_yita[i*n+j],c_times(g[i][k],u_g[k][j]));
                }
            }
        }
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                for (int k = 0; k < n; k++) {
                    U_xi[i*n+j] = c_sum(U_xi[i*n+j],c_times(g_xi[i][k],u_gxi[k][j]));
                }
            }
        }

        g[gi][gi] = Complex(1,0);g[gi][gj] = Complex(0,0);
        g[gj][gi] = Complex(0,0);g[gj][gj] = Complex(1,0);
        g_xi[gi][gi] = Complex(1,0);g_xi[gi][gj] = Complex(0,0);
        g_xi[gj][gi] = Complex(0,0);g_xi[gj][gj] = Complex(1,0);
        yita_s[0] = nrm(2,yita_s);
        xi_s[0] = nrm(2,xi_s);
    }
}
int main(){
    int n,s,ss; //复空间维数
    double r;//复数模长
    double theta;//复数辐角
    srand48((unsigned)time(NULL));
    printf("Enter n:\n");
    // 获取程序开始执行的时间点
    auto start = std::chrono::steady_clock::now();
    scanf("%d",&n);
    printf("DIM C = %d \n", n);

    Complex *yita = new Complex[n];
    Complex *xi = new Complex[n];
    Complex *U = new Complex[n*n];
    Complex *U_yita = new Complex[n*n];
    Complex *U_xi = new Complex[n*n];
    Complex *yita_g = new Complex[n];
    //用于检验线性变换
    Complex *err = new Complex[n];

    //初始化
    for(s = 0;s<n;s++){
        for(ss = 0;ss<n;ss++){
            U_yita[s*n+ss] = Complex(0,0);
            U_xi[s*n+ss] = Complex(0,0);
            U[s*n+ss] = Complex(0,0);
            if (s==ss){
                U_yita[s*n+ss] = Complex(1,0);
                U_xi[s*n+ss] = Complex(1,0);
            }
        }
        r = drand48();
        yita[s]=Complex(r,0);
        r = drand48();
        xi[s]=Complex(r,0);
    }

    //归一化
    double nrm_yita = nrm(n, yita);
    double nrm_xi = nrm(n, xi);

    for(s = 0; s<n; s++){
        r = real(yita[s])/nrm_yita;
        theta = 2*pi*drand48();
        yita[s] = Complex(r*cos(theta),r*sin(theta));

        r = real(xi[s])/nrm_xi;
        theta = 2*pi*drand48();
        xi[s] = Complex(r*cos(theta),r*sin(theta));
    }
    printf("两向量模长分别为：%.2f, %.2f\n\n", nrm(n,yita), nrm(n,xi));

    //构造U_yita和U_xi矩阵
    Givens(n, yita, xi, U_yita, U_xi);

    //计算U_xi的逆矩阵
    complexMatrixInverse(U_xi, n);
    //计算U = (U_xi)^{-1}*U_yita
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            for (int k = 0; k < n; k++) {
                U[i*n+j] = c_sum(U[i*n+j],c_times(U_xi[i*n+k],U_yita[k*n+j]));
            }
        }
    }

    for(int i=0;i<n;i++){
        yita_g[i] = Complex(0,0);
        for (int k=0;k<n;k++) {
            yita_g[i] =c_sum(yita_g[i], c_times(U[i*n+k],yita[k]));
        }
    }
    printf("Givens变换构造的矩阵U\n");
    printMatrix(U,n);
    printf("\n");
    cout<<"验证得到的矩阵是幺正矩阵"<<endl;
    Complex det = calculateDeterminant(U,n);
    printf("det(U) = (%.3f,%.3fi)\t|det(U)| = %.3f\n",real(det),imag(det),real(c_times(tran(det),det)));
    Complex U_dagger[n*n];
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
        U_dagger[i*n+j] = tran(U[j*n+i]);
        }
    }
    cout<<"继续计算转置复共轭与其逆相等"<<endl;
    complexMatrixInverse(U, n);//逆
    //printMatrix(U,n);
    //printMatrix(U_dagger,n);
    //设置一个误差范围；
    double error = 1e-5;
    int is_check = 1;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            if(fabs(real(U[i*n+j])-real(U_dagger[i*n+j]))>error||fabs(real(U[i*n+j])-real(U_dagger[i*n+j]))>error)
            is_check = 0;
        }
        if(is_check == 0)
        break;
    }
    if(is_check)cout<<"已验证矩阵的转置复共轭与其逆相等"<<endl;
    else cout<<"幺正性检验未通过"<<endl;
    //打印U*yita和xi进行对比
    printf("yita\t\txi\t\tU*yita\t\terror\n");
    for(int i=0;i<n;i++){
        err[i] = yita_g[i]-xi[i];
        if(imag(yita[i])>0)printf("(%.2f+%.2fi)\t",real(yita[i]), imag(yita[i]));
        else printf("(%.2f%.2fi)\t",real(yita[i]), imag(yita[i]));
        if(imag(xi[i])>0)printf("(%.2f+%.2fi)\t",real(xi[i]), imag(xi[i]));
        else printf("(%.2f%.2fi)\t",real(xi[i]), imag(xi[i]));
        if(imag(yita_g[i])>0)printf("(%.2f+%.2fi)\t",real(yita_g[i]), imag(yita_g[i]));
        else printf("(%.2f%.2fi)\t",real(yita_g[i]), imag(yita_g[i]));
        if(imag(err[i])>0||imag(err[i])==0)printf("(%.5f+%.5fi)\n",real(err[i]), imag(err[i]));
        else printf("(%.5f%.5fi)\n",real(err[i]), imag(err[i]));
    }
    free(U_yita);free(U_xi);


    Householder(n,yita,xi,U);
    for(int i=0;i<n;i++){
        yita_g[i] = Complex(0,0);
        for (int k=0;k<n;k++) {
            yita_g[i] =c_sum(yita_g[i], c_times(U[i*n+k],yita[k]));
        }
    }
    printf("\n\n");
    printf("Householder变换构造的矩阵U\n");
    printMatrix(U,n);
    printf("\n");
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            U_dagger[i*n+j] = tran(U[j*n+i]);
        }
    }
    cout<<"验证得到的矩阵是幺正矩阵"<<endl;
    det = calculateDeterminant(U,n);
    printf("det(U) = (%.3f,%.3fi)\t|det(U)| = %.3f\n",real(det),imag(det),real(c_times(tran(det),det)));
    printf("继续计算矩阵的转置复共轭与其逆相等\n");
    complexMatrixInverse(U, n);
    //printMatrix(U,n);
    //设置一个误差范围；
    is_check = 1;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            if(fabs(real(U[i*n+j])-real(U_dagger[i*n+j]))>error||fabs(real(U[i*n+j])-real(U_dagger[i*n+j]))>error)
            is_check = 0;
        }
        if(is_check == 0)
        break;
    }
    if(is_check)cout<<"已验证矩阵的转置复共轭与其逆相等"<<endl;
    else cout<<"幺正性检验未通过"<<endl;
    printf("yita\t\txi\t\tU*yita\t\terror\n");
    
    //打印U*yita和xi进行对比
    for(int i=0;i<n;i++){
        err[i] = yita_g[i]-xi[i];
        if(imag(yita[i])>0)printf("(%.2f+%.2fi)\t",real(yita[i]), imag(yita[i]));
        else printf("(%.2f%.2fi)\t",real(yita[i]), imag(yita[i]));
        if(imag(xi[i])>0)printf("(%.2f+%.2fi)\t",real(xi[i]), imag(xi[i]));
        else printf("(%.2f%.2fi)\t",real(xi[i]), imag(xi[i]));
        if(imag(yita_g[i])>0)printf("(%.2f+%.2fi)\t",real(yita_g[i]), imag(yita_g[i]));
        else printf("(%.2f%.2fi)\t",real(yita_g[i]), imag(yita_g[i]));
        if(imag(err[i])>0||imag(err[i])==0)printf("(%.5f+%.5fi)\n",real(err[i]), imag(err[i]));
        else printf("(%.5f%.5fi)\n",real(err[i]), imag(err[i]));
    }

    printf("\n");
    free(yita);free(xi);free(yita_g);free(U);free(err);
    auto end = chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "---------Total time--------" << endl << duration.count() << "ms" << endl;
    return 0;
}