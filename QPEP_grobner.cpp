// LibQPEP: A Library for Globally Optimal Solving Quadratic Pose Estimation Problems,
//          It also gives highly accurate uncertainty description of the solutions.
//
// Authors: Jin Wu and Ming Liu
// Affiliation: Hong Kong University of Science and Technology (HKUST)
// Emails: jin_wu_uestc@hotmail.com; eelium@ust.hk
//
//
// QPEP_grobner.cpp: Solving QPEPs using different Groebner-basis methods


#include "QPEP_grobner.h"
#include "utils.h"
#include <cstdlib>
#include <numeric>

template <class __T>
static double J_func_wrapper(const Eigen::Quaternion<__T>& q,
                    const Eigen::Vector3<__T>& t,
                    const Eigen::MatrixX<__T>& coef_J_pure,
                    mon_J_pure_func_handle<__T> mon_J_pure_func)
{
    double J = 0;
    int num = coef_J_pure.cols();
    std::vector<__T> mon = mon_J_pure_func(q, t);
    for(int i = 0; i < num; ++i)
    {
        J = J + coef_J_pure(0, i) * mon[i];
    }
    return J;
}

template <class __T>
struct QPEP_runtime QPEP_WQ_grobner(Eigen::Matrix3<__T>& R,
                     Eigen::Vector3<__T>& t,
                     Eigen::Matrix4<__T>& X,
                     __T* min,
                     const Eigen::MatrixX<__T>& W,
                     const Eigen::MatrixX<__T>& Q,
                     const solver_func_handle<__T>& solver_func,
                     const mon_J_pure_func_handle<__T>& mon_J_pure_func,
                     const t_func_handle<__T>& t_func,
                     const Eigen::MatrixX<__T>& coef_J_pure,
                     const Eigen::MatrixX<__T>& coefs_tq,
                     const Eigen::MatrixX<__T>& pinvG,
                     const int* perm,
                     const struct QPEP_options& opt)
{
    struct QPEP_runtime stat;
    int solution_size = 0;
    Eigen::VectorX<__T> data;

    if(opt.ModuleName == "solver_WQ")
    {
        if(opt.ModuleName == "solver_WQ")
        {
            data.resize(272);
            data << vec<__T>(W), vec<__T>(Q);
            solution_size = 40;
        }
    }
    else if(opt.ModuleName == "solver_WQ_1_2_3_4_5_9_13_17_33_49_approx" ||
       opt.ModuleName == "solver_WQ_approx")
    {
        int permutation[3] = {0, 1, 2};
        if(perm)
        {
            permutation[0] = perm[0];
            permutation[1] = perm[1];
            permutation[2] = perm[2];
        }

        Eigen::Matrix<__T, 3, 64> W_;
        Eigen::Matrix<__T, 3, 4> Q_;
        for(int i = 0; i < 3; ++i)
        {
            W_.row(i) = W.row(permutation[i]);
            Q_.row(i) = Q.row(permutation[i]);
        }

        Eigen::MatrixX<__T> unique;

        if(opt.ModuleName == "solver_WQ_1_2_3_4_5_9_13_17_33_49_approx") {
            int unique_idx[16] = {6, 7, 8, 11, 12, 16, 22, 23, 24, 27, 28, 32, 43, 44, 48, 64};
            unique.resize(3, 16);
            for (int i = 0; i < 16; ++i) {
                unique.col(i) = W_.col(unique_idx[i] - 1).topRows(3);
            }
            data.resize(60);
            data << vec<__T>(unique), vec<__T>(Q_.topRows(3));
            solution_size = 27;
        }
        else if(opt.ModuleName == "solver_WQ_approx")
        {
            data.resize(204);
            data << vec<__T>(W_.topRows(3)), vec<__T>(Q_.topRows(3));
            solution_size = 27;
        }
    }

    assert(solution_size > 0);

    Eigen::MatrixX< std::complex<__T> > sols;
    sols.resize(4, solution_size);
    stat = solver_func(sols, data, opt);

    clock_t time1 = clock();
    Eigen::VectorX<__T> q0_, q1_, q2_, q3_;
    q0_.resize(solution_size);
    q1_.resize(solution_size);
    q2_.resize(solution_size);
    q3_.resize(solution_size);
    q0_ = sols.row(0).transpose().real();
    q1_ = sols.row(1).transpose().real();
    q2_ = sols.row(2).transpose().real();
    q3_ = sols.row(3).transpose().real();

    std::vector< Eigen::Quaternion<__T> > qs;
    Eigen::MatrixX<__T> ts;
    ts.resize(3, solution_size);
    std::vector<__T> Js;
    for(int i = 0; i < solution_size; ++i)
    {
        Eigen::Quaternion<__T> qq(q3_(i), q0_(i), q1_(i), q2_(i));
        qq.normalize();
        Eigen::Vector3<__T> tt = t_func(pinvG, coefs_tq, qq);
        qs.push_back(qq);
        ts.col(i) = tt;
        Js.push_back(J_func_wrapper(qq, tt, coef_J_pure, mon_J_pure_func));
    }

    std::vector<size_t> idx;
    sort_indices<__T>(idx, Js);
    R = q2R(qs[idx[0]]);
    t = ts.col(idx[0]);
    X << R, t, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
    clock_t time2 = clock();
    stat.timeEigen = (time2 - time1) / double(CLOCKS_PER_SEC);
    stat.statusEigen = 0;
    stat.lossGrobner = Js[idx[0]];

    return stat;
}


template struct QPEP_runtime QPEP_WQ_grobner(Eigen::Matrix3<float>& R,
                                             Eigen::Vector3<float>& t,
                                             Eigen::Matrix4<float>& X,
                                             float* min,
                                             const Eigen::MatrixX<float>& W,
                                             const Eigen::MatrixX<float>& Q,
                                             const solver_func_handle<float>& solver_func,
                                             const mon_J_pure_func_handle<float>& mon_J_pure_func,
                                             const t_func_handle<float>& t_func,
                                             const Eigen::MatrixX<float>& coef_J_pure,
                                             const Eigen::MatrixX<float>& coefs_tq,
                                             const Eigen::MatrixX<float>& pinvG,
                                             const int* perm,
                                             const struct QPEP_options& opt);

template struct QPEP_runtime QPEP_WQ_grobner(Eigen::Matrix3<double>& R,
                                             Eigen::Vector3<double>& t,
                                             Eigen::Matrix4<double>& X,
                                             double* min,
                                             const Eigen::MatrixX<double>& W,
                                             const Eigen::MatrixX<double>& Q,
                                             const solver_func_handle<double>& solver_func,
                                             const mon_J_pure_func_handle<double>& mon_J_pure_func,
                                             const t_func_handle<double>& t_func,
                                             const Eigen::MatrixX<double>& coef_J_pure,
                                             const Eigen::MatrixX<double>& coefs_tq,
                                             const Eigen::MatrixX<double>& pinvG,
                                             const int* perm,
                                             const struct QPEP_options& opt);

template struct QPEP_runtime QPEP_WQ_grobner(Eigen::Matrix3<long double>& R,
                                             Eigen::Vector3<long double>& t,
                                             Eigen::Matrix4<long double>& X,
                                             long double* min,
                                             const Eigen::MatrixX<long double>& W,
                                             const Eigen::MatrixX<long double>& Q,
                                             const solver_func_handle<long double>& solver_func,
                                             const mon_J_pure_func_handle<long double>& mon_J_pure_func,
                                             const t_func_handle<long double>& t_func,
                                             const Eigen::MatrixX<long double>& coef_J_pure,
                                             const Eigen::MatrixX<long double>& coefs_tq,
                                             const Eigen::MatrixX<long double>& pinvG,
                                             const int* perm,
                                             const struct QPEP_options& opt);


