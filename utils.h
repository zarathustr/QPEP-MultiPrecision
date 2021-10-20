#ifndef LIBQPEP_UTILS_H
#define LIBQPEP_UTILS_H

#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>
#include <fstream>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <Eigen/Sparse>
#include "QPEP.h"

#ifdef USE_SUPERLU
#include <Eigen/SuperLUSupport>
#endif

template <class __T>
inline Eigen::VectorX<__T> vec(const Eigen::MatrixX<__T> X)
{
    Eigen::VectorX<__T> res;
    int m = X.cols();
    int n = X.rows();
    res.resize(m * n);
    int counter = 0;
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j) {
            res(counter) = X(j, i);
            counter = counter + 1;
        }
    }
    return res;
}

template <class __T>
inline Eigen::Matrix3<__T> q2R(const Eigen::Quaternion<__T>& q)
{
    __T q0 = q.x();
    __T q1 = q.y();
    __T q2 = q.z();
    __T q3 = q.w();
    __T q02 = q0 * q0;
    __T q12 = q1 * q1;
    __T q22 = q2 * q2;
    __T q32 = q3 * q3;

    Eigen::Matrix3<__T> R;
    R <<     q02 + q12 - q22 - q32,     2.0*q0*q3 + 2.0*q1*q2,     2.0*q1*q3 - 2.0*q0*q2,
             2.0*q1*q2 - 2.0*q0*q3,     q02 - q12 + q22 - q32,     2.0*q0*q1 + 2.0*q2*q3,
             2.0*q0*q2 + 2.0*q1*q3,     2.0*q2*q3 - 2.0*q0*q1,     q02 - q12 - q22 + q32;
    return R;
}

template <class __T>
inline Eigen::Vector4<__T> R2q(const Eigen::Matrix3<__T>& R)
{
//    __T det = R.determinant();
//    __T orthogonality = (R.transpose() * R).trace();
//    assert(std::fabs(det - 1.0) < 1e-8 && std::fabs(orthogonality - 3.0) < 1e-8);

    __T G11 = R(0, 0) + R(1, 1) + R(2, 2) - 3.0, G12 = R(1, 2) - R(2, 1), G13 = R(2, 0) - R(0, 2), G14 = R(0, 1) - R(1, 0);
    __T G22 = R(0, 0) - R(1, 1) - R(2, 2) - 3.0, G23 = R(0, 1) + R(1, 0), G24 = R(0, 2) + R(2, 0);
    __T G33 = R(1, 1) - R(0, 0) - R(2, 2) - 3.0, G34 = R(1, 2) + R(2, 1);

    Eigen::Vector4<__T> qRes = Eigen::Vector4<__T> (
            G14 * G23 * G23 - G13 * G23 * G24 - G14 * G22 * G33 + G12 * G24 * G33 + G13 * G22 * G34 - G12 * G23 * G34,
            G13 * G13 * G24 + G12 * G14 * G33 - G11 * G24 * G33 + G11 * G23 * G34 - G13 * G14 * G23 - G13 * G12 * G34,
            G13 * G14 * G22 - G12 * G14 * G23 - G12 * G13 * G24 + G11 * G23 * G24 + G12 * G12 * G34 - G11 * G22 * G34,
            - ( G13 * G13 * G22 - 2 * G12 * G13 * G23 + G11 * G23 * G23 + G12 * G12 * G33 - G11 * G22 * G33 ));
    qRes.normalize();
    if(qRes(0) < 0)
        qRes = - qRes;
    return qRes;
}

#define MAX2(A, B) (A > B ? A : B)
#define MAX4(A, B, C, D) (MAX2(MAX2(A, B), MAX2(C, D)))

template <class __T>
struct __node{
    __T value;
    int index;
};

template <class __T>
inline bool cmp(struct __node<__T> a, struct __node<__T> b)
{
    if (a.value < b.value)
    {
        return true;
    }
    return false;
}

template <typename T>
T sort_indices(std::vector<size_t> &idx, const std::vector<T> &v)
{
    __node<T>* a = new __node<T>[v.size()];
    for (int i = 0; i < v.size(); i++)
    {
        a[i].value = v[i];
        a[i].index = i;
    }

    std::sort(a, a + v.size(), cmp<T>);
    for (int i = 0; i < v.size(); i++)
    {
        idx.push_back(a[i].index);
    }
    delete[] a;

    return 0;
}


template <class __T>
inline __T powers(__T x, __T order)
{
    if(order == 2.0){
        return x * x;
    }
    else if(order == 3.0){
        return x * x * x;
    }
    else if(order == 4.0) {
        __T x2 = x * x;
        return x2 * x2;
    }
    else{
        return std::pow(x, order);
    }
}

template <class __T>
using data_func_handle = void (*)(Eigen::SparseMatrix<__T>& C1,
                                  Eigen::MatrixX<__T>& C2, const Eigen::VectorX<__T>& data);

template <class __T>
inline QPEP_runtime GaussJordanElimination(
        Eigen::MatrixX<__T>& C1_,
        const Eigen::VectorX<__T>& data,
        const data_func_handle<__T> data_func,
        const int& size_GJ,
        const int& size_AM,
        const struct QPEP_options& opt,
        const struct QPEP_runtime& stat_)
{
    assert(opt.DecompositionMethod == "SparseLU" ||
           opt.DecompositionMethod == "SparseQR" ||
           opt.DecompositionMethod == "SuperLU" ||
           opt.DecompositionMethod == "HouseholderQR" ||
           opt.DecompositionMethod == "PartialPivLU" ||
           opt.DecompositionMethod == "SVD" ||
           opt.DecompositionMethod == "BDCSVD" ||
           opt.DecompositionMethod == "Inv" ||
           opt.DecompositionMethod == "Cholesky" ||
           opt.DecompositionMethod == "LinSolve");

    struct QPEP_runtime stat = stat_;
    C1_.resize(size_GJ, size_AM);
    C1_.setZero();
    Eigen::MatrixX<__T> C2;
    C2.resize(size_GJ, size_AM);
    C2.setZero();

    if (opt.DecompositionMethod == "SparseLU")
    {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::SparseLU<Eigen::SparseMatrix<__T>, Eigen::COLAMDOrdering<int> > solver;
        solver.compute(C1);
        if (solver.info() != Eigen::Success) {
            std::cout << "Sparse LU Decomposition Failed!" << std::endl;
            stat.statusDecomposition = -3;
            return stat;
        }

        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorX<__T> c1 = solver.solve(C2.col(i));
            if (solver.info() != Eigen::Success) {
                std::cout << "Least Squares Failed!" << std::endl;
                stat.statusDecomposition = -6;
                return stat;
            }
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
    else if (opt.DecompositionMethod == "SparseQR")
    {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        C1.makeCompressed();
        Eigen::SparseQR<Eigen::SparseMatrix<__T>, Eigen::COLAMDOrdering<int> > solver;
        solver.compute(C1);
        if (solver.info() != Eigen::Success) {
            std::cout << "Sparse QR Decomposition Failed!" << std::endl;
            stat.statusDecomposition = -3;
            return stat;
        }

        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorX<__T> c1 = solver.solve(C2.col(i));
            if (solver.info() != Eigen::Success) {
                std::cout << "Least Squares Failed!" << std::endl;
                stat.statusDecomposition = -6;
                return stat;
            }
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
    else if(opt.DecompositionMethod == "HouseholderQR")
    {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixX<__T> CC1(C1.toDense());
        Eigen::HouseholderQR<Eigen::MatrixX<__T> > solver;
        solver.compute(CC1);
        for(int i = 0; i < C2.cols(); ++i)
        {
            Eigen::VectorX<__T> c1 = solver.solve(C2.col(i));
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
    else if(opt.DecompositionMethod == "PartialPivLU")
    {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixX<__T> CC1(C1.toDense());
        Eigen::PartialPivLU<Eigen::MatrixX<__T> > solver;
        solver.compute(CC1);
        for(int i = 0; i < C2.cols(); ++i)
        {
            Eigen::VectorX<__T> c1 = solver.solve(C2.col(i));
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
    else if(opt.DecompositionMethod == "SVD") {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixX<__T> CC1(C1.toDense());
        Eigen::JacobiSVD< Eigen::MatrixX<__T> > solver(CC1, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixX<__T> singularValues = solver.singularValues();
        Eigen::MatrixX<__T> singularValuesInv;
        singularValuesInv.resize(size_GJ, size_GJ);
        singularValuesInv.setZero();
        __T pinvtoler = 1.e-8; // choose your tolerance wisely
        for (int i = 0; i < size_GJ; ++i) {
            if (singularValues(i) > pinvtoler)
                singularValuesInv(i, i) = 1.0 / singularValues(i);
            else
                singularValuesInv(i, i) = 0.0;
        }
        Eigen::MatrixX<__T> pinv = solver.matrixV() * singularValuesInv * solver.matrixU().transpose();

        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorX<__T> c1 = pinv * C2.col(i);
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
#if EIGEN_VERSION_AT_LEAST(3,3,0)
    else if(opt.DecompositionMethod == "BDCSVD") {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixX<__T> CC1(C1.toDense());
        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorX<__T> c1 = CC1.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(C2.col(i));
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
#endif
    else if(opt.DecompositionMethod == "Inv") {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixX<__T> CC1(C1.toDense());
        Eigen::MatrixX<__T> inv = CC1.inverse();
        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorX<__T> c1 = inv * C2.col(i);
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
    else if(opt.DecompositionMethod == "Cholesky") {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixX<__T> CC1(C1.toDense());
        Eigen::LLT< Eigen::MatrixX<__T> > solver(CC1);
        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorX<__T> c1 = solver.solve(C2.col(i));
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
    else if(opt.DecompositionMethod == "LinSolve") {
        clock_t time1 = clock();
        Eigen::SparseMatrix<__T> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        clock_t time2 = clock();
        stat.timeDecompositionDataPrepare = (time2 - time1) / double(CLOCKS_PER_SEC);
        Eigen::SparseMatrix<__T> C1T = C1.transpose();
        Eigen::SparseMatrix<__T> AA = C1T * C1;
        Eigen::MatrixX<__T> BB = C1T.toDense() * C2;
        C1_ = AA.toDense().inverse() * BB;
        clock_t time3 = clock();
        stat.timeDecomposition = (time3 - time2) / double(CLOCKS_PER_SEC);
    }
#ifdef USE_SUPERLU
    else if(opt.DecompositionMethod == "Cholesky") {
        clock_t time1 = clock();
        Eigen::SparseMatrix<double> C1(size_GJ, size_GJ);
        C1.setZero();
        data_func(C1, C2, data);
        Eigen::MatrixXd CC1(C1.toDense());
        Eigen::LLT<Eigen::MatrixXd> solver(CC1);
        for (int i = 0; i < C2.cols(); ++i) {
            Eigen::VectorXd c1 = solver.solve(C2.col(i));
            C1_.col(i) = c1;
        }
        clock_t time2 = clock();
        stat.timeDecomposition = (time2 - time1) / double(CLOCKS_PER_SEC);
    }
#endif


    return stat;
}

template <typename T>
T mean(std::vector<T> data)
{
    int num = data.size();
    assert(num > 0);

    T s = data[0];
    double factor = 1.0 / ((double) num);
    if(num > 1)
        std::for_each(data.begin() + 1, data.end(), [&s](T x){s += x;});
    return s * factor;
}

template <class __T>
inline void readPnPdata(std::string filename,
                 Eigen::Matrix3<__T>& R0,
                 Eigen::Vector3<__T>& t0,
                 Eigen::Matrix3<__T>& K,
                 std::vector< Eigen::Vector3<__T> >& world_pt0,
                 std::vector< Eigen::Vector2<__T> >& image_pt0)
{
    std::ifstream input(filename);
    __T fx, fy, cx, cy;
    input >> R0(0, 0) >> R0(0, 1) >> R0(0, 2) >>
          R0(1, 0) >> R0(1, 1) >> R0(1, 2) >>
          R0(2, 0) >> R0(2, 1) >> R0(2, 2);
    input >> t0(0) >> t0(1) >> t0(2);
    input >> fx >> fy >> cx >> cy;
    K(0, 0) = fx;
    K(1, 1) = fy;
    K(0, 2) = cx;
    K(1, 2) = cy;
    K(2, 2) = 1.0;
    int num = 0;
    input >> num;
    world_pt0.resize(num);
    image_pt0.resize(num);
    for(int i = 0; i < num; ++i)
    {
        input >> world_pt0[i](0) >> world_pt0[i](1) >> world_pt0[i](2);
    }
    for(int i = 0; i < num; ++i)
    {
        input >> image_pt0[i](0) >> image_pt0[i](1);
    }
    input.close();
}

template <class __T>
inline void readpTopdata(std::string filename,
                  Eigen::Matrix3<__T>& R0,
                  Eigen::Vector3<__T>& t0,
                  std::vector<Eigen::Vector3<__T> >& r0,
                  std::vector<Eigen::Vector3<__T> >& b0,
                  std::vector<Eigen::Vector3<__T> >& nv)
{
    std::ifstream input(filename);
    input >> R0(0, 0) >> R0(0, 1) >> R0(0, 2) >>
          R0(1, 0) >> R0(1, 1) >> R0(1, 2) >>
          R0(2, 0) >> R0(2, 1) >> R0(2, 2);
    input >> t0(0) >> t0(1) >> t0(2);
    int num = 0;
    input >> num;
    r0.resize(num);
    b0.resize(num);
    nv.resize(num);
    for(int i = 0; i < num; ++i)
    {
        input >> r0[i](0) >> r0[i](1) >> r0[i](2);
    }
    for(int i = 0; i < num; ++i)
    {
        input >> b0[i](0) >> b0[i](1) >> b0[i](2);
    }
    for(int i = 0; i < num; ++i)
    {
        input >> nv[i](0) >> nv[i](1) >> nv[i](2);
    }
    input.close();
}

#ifdef USE_OPENCV
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

template <class __T>
inline std::vector< cv::Point3_<__T> > Vector3ToPoint3(std::vector< Eigen::Vector3<__T> > pt)
{
    std::vector< cv::Point3_<__T> > tmp;
    for(int i = 0; i < pt.size(); ++i)
    {
        Eigen::Vector3<__T> point = pt[i];
        cv::Point3_<__T> vec;
        vec.x = point(0);
        vec.y = point(1);
        vec.z = point(2);
        tmp.push_back(vec);
    }
    return tmp;
}

template <class __T>
inline std::vector< cv::Point_<__T> > Vector2ToPoint2(std::vector< Eigen::Vector2<__T> > pt)
{
    std::vector< cv::Point_<__T> > tmp;
    for(int i = 0; i < pt.size(); ++i)
    {
        Eigen::Vector2<__T> point = pt[i];
        cv::Point_<__T> vec;
        vec.x = point(0);
        vec.y = point(1);
        tmp.push_back(vec);
    }
    return tmp;
}
#endif

#endif //LIBQPEP_UTILS_H
