#include <iostream>
#include <torch/torch.h>
#include <iomanip>


torch::Tensor Creat_Ham(double S) {
    if (S == 0.5) {
        torch::Tensor s_up = torch::tensor({0.0, 1.0, 0.0, 0.0}, torch::kDouble).view({2, 2});
        torch::Tensor s_dn = torch::tensor({0.0, 0.0, 1.0, 0.0}, torch::kDouble).view({2, 2});
        torch::Tensor s_z = 0.5 * torch::tensor({1.0, 0.0, 0.0, -1.0}, torch::kDouble).view({2, 2});
        return 0.5 * (torch::kron(s_up, s_dn) + torch::kron(s_dn, s_up)) + torch::kron(s_z, s_z);
    } else if (S == 1.0) {
        torch::Tensor s_up = torch::tensor({0.0, sqrt(2.0), 0.0, 0.0, 0.0, sqrt(2.0), 0.0, 0.0, 0.0},
                                           torch::kDouble).view({3, 3});
        torch::Tensor s_dn = torch::tensor({0.0, 0.0, 0.0, sqrt(2.0), 0.0, 0.0, 0.0, sqrt(2.0), 0.0},
                                           torch::kDouble).view({3, 3});
        torch::Tensor s_z = torch::tensor({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0},
                                          torch::kDouble).view({3, 3});
        return 0.5 * (torch::kron(s_up, s_dn) + torch::kron(s_dn, s_up)) + torch::kron(s_z, s_z);
    }
    return torch::eye(2);
}

std::tuple<torch::Tensor, torch::Tensor> Creat_Mps(int chi_dim, int phy_dim) {
    torch::Tensor G = torch::rand({2, chi_dim, phy_dim, chi_dim}, torch::kDouble);
    torch::Tensor L = torch::rand({2, chi_dim}, torch::kDouble);
    return std::make_tuple(G, L);
}

int main() {
    double S = 0.5;
    int a, b;
    int physics_dim = int(2 * S) + 1;
    int chi = 30;

    torch::Tensor U_svd;
    torch::Tensor Lambda_svd;
    torch::Tensor V;
    torch::Tensor X;
    torch::Tensor Y;

    std::cout<<"---Creat Tensor MPS"<<std::endl;

    torch::Tensor Ham = Creat_Ham(S);
    std::tuple Mps = Creat_Mps(chi, physics_dim);
    torch::Tensor Theta;
    torch::Tensor Gamma = std::get<0>(Mps);
    torch::Tensor Lambda = std::get<1>(Mps);

    std::cout<<"---Creat Tensor U"<<std::endl;
    double tao = 0.01;
    torch::Tensor U = torch::matrix_exp(-1*tao*Ham).reshape({physics_dim, physics_dim,
                                                             physics_dim, physics_dim});

    torch::Tensor E = torch::tensor({0.0, 0.0}, torch::kDouble);

    int check_ab = 0;
    double Eg = 0;
    double err = 0;
    std::cout<<"---Begin Calculate"<<std::endl;

    while(true){
        a = check_ab % 2;
        b = (check_ab + 1) % 2;

        Theta = torch::einsum("aa, aic -> aic", {torch::diag(Lambda[b]), Gamma[a]});
        Theta = torch::einsum("aic, cc -> aic", {Theta, torch::diag(Lambda[a])});
        Theta = torch::einsum("aid, dje -> aije", {Theta, Gamma[b]});
        Theta = torch::einsum("aije, ee -> aije", {Theta, torch::diag(Lambda[b])});
        Theta = torch::einsum("aijy, ijcd -> aycd", {Theta, U});
        Theta = Theta.permute({2, 0, 3, 1}).reshape({chi * physics_dim, physics_dim * chi});


        E[a] = -torch::log(Theta.norm()*Theta.norm()) / (2 * tao);
        err = std::abs(Eg-E.mean().item<double>());
        Eg = E.mean().item<double>();
        std::cout << "E = " <<std::setprecision(10)<< Eg << std::endl;

        auto SVD = torch::linalg_svd(Theta);
        U_svd = std::get<0>(SVD);
        Lambda_svd = std::get<1>(SVD);
        V = std::get<2>(SVD);

        Lambda[a] = torch::div(Lambda_svd.index({torch::indexing::Slice(torch::indexing::None, chi)}),
                               Lambda_svd.index({torch::indexing::Slice(torch::indexing::None, chi)})
                               .norm());

        X = U_svd.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, chi)});
        X = torch::reshape(X, {physics_dim, chi, chi});
        X = X.permute({1, 0, 2});
        Gamma[a] = torch::einsum("ii, ijk -> ijk", {torch::diag(torch::reciprocal(Lambda[b])), X});

        Y = V.index({torch::indexing::Slice(torch::indexing::None, chi), torch::indexing::Slice()});
        Y = torch::reshape(Y, {chi, physics_dim, chi});
        Gamma[b] = torch::einsum("ijk, kk -> ijk", {Y, torch::diag(torch::reciprocal(Lambda[b]))});

        if(err<1e-8||check_ab>19999)break;
        check_ab++;
    }
    //FILE.close();
    std::cout<<"Total Steps = "<< check_ab <<std::endl;
    std::cout<<"Eg = "<<std::setprecision(10)<< E.mean().item<double>() <<std::endl;
    return 0;
}
