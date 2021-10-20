// LibQPEP: A Library for Globally Optimal Solving Quadratic Pose Estimation Problems,
//          It also gives highly accurate uncertainty description of the solutions.
//
// Authors: Jin Wu and Ming Liu
// Affiliation: Hong Kong University of Science and Technology (HKUST)
// Emails: jin_wu_uestc@hotmail.com; eelium@ust.hk
//
//
// solver_WQ.cpp


#include "solver_WQ.h"
#include "solver_WQ_helper.h"
#include "utils.h"

template <class __T>
struct QPEP_runtime solver_WQ(Eigen::MatrixX< std::complex<__T> >& sol_,
        const Eigen::VectorX<__T>& data,
        const struct QPEP_options& opt) {
    assert(opt.ModuleName == "solver_WQ");

    struct QPEP_runtime stat;
    Eigen::MatrixX<__T> C1_;
    stat = GaussJordanElimination<__T>(C1_, data,
                                  data_func_solver_WQ_func,
                                  596, 40,
                                  opt, stat);
    stat.statusDecomposition = 0;

    clock_t time1 = clock();
    Eigen::Matrix<__T, 72, 40> RR;
    RR << - C1_.bottomRows(32), Eigen::Matrix<__T, 40, 40>::Identity();
    const int AM_ind[40] = {34, 1, 2, 3, 4, 5, 6, 7, 8, 35,
                      9, 10, 46, 11, 12, 13, 14, 15, 16,
                      17, 18, 47, 19, 20, 50, 21, 22, 23,
                      24, 25, 26, 51, 40, 28, 52, 29, 30, 53, 31, 32};
    Eigen::Matrix<__T, 40, 40> AM;
    for(int i = 0; i < 40; ++i)
    {
        AM.row(i) = RR.row(AM_ind[i] - 1);
    }

    Eigen::EigenSolver<Eigen::Matrix<__T, 40, 40> > eigs(AM);
    Eigen::Matrix<std::complex<__T>, 40, 40> V = eigs.eigenvectors();
    Eigen::MatrixX< std::complex<__T> > D(eigs.eigenvalues());

    Eigen::VectorX< std::complex<__T> > scale(40);
    for(int i = 0; i < 40; ++i)
    {
        scale(i) = sqrt(D(i) / (V(12, i) * V(34, i)));
        V.col(i) = V.col(i) * scale(i);
    }

    sol_.row(0) = V.row(0);
    sol_.row(1) = V.row(12);
    sol_.row(2) = V.row(24);
    sol_.row(3) = V.row(34);

    clock_t time2 = clock();
    stat.timeGrobner = (time2 - time1) / double(CLOCKS_PER_SEC);
    stat.statusGrobner = 0;
    return stat;
}


template struct QPEP_runtime solver_WQ(Eigen::MatrixX< std::complex<float> >& sol_,
                                       const Eigen::VectorX<float>& data,
                                       const struct QPEP_options& opt);

template struct QPEP_runtime solver_WQ(Eigen::MatrixX< std::complex<double> >& sol_,
                                       const Eigen::VectorX<double>& data,
                                       const struct QPEP_options& opt);

template struct QPEP_runtime solver_WQ(Eigen::MatrixX< std::complex<long double> >& sol_,
                                       const Eigen::VectorX<long double>& data,
                                       const struct QPEP_options& opt);