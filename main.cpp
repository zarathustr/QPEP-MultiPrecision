// LibQPEP: A Library for Globally Optimal Solving Quadratic Pose Estimation Problems,
//          It also gives highly accurate uncertainty description of the solutions.
//
// Authors: Jin Wu and Ming Liu
// Affiliation: Hong Kong University of Science and Technology (HKUST)
// Emails: jin_wu_uestc@hotmail.com; eelium@ust.hk
//
//
// main.cpp: Demos and visualization



#include <iostream>
#include "generateProjectedPoints.h"
#include "solver_WQ_1_2_3_4_5_9_13_17_33_49_approx.h"
#include "solver_WQ_approx.h"
#include "solver_WQ.h"
#include "utils.h"
#include "QPEP_grobner.h"
#include "pnp_WQD.h"
#include "pTop_WQD.h"
#include "misc_pnp_funcs.h"
#include "misc_pTop_funcs.h"
#include "QPEP_lm_single.h"
#include "QPEP.h"

#ifdef USE_OPENCV
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CVLib.h"
#endif

#ifndef NO_OMP
#include "omp.h"
int num_threads_ = 0;
#endif

#include "CovEstimation.h"
#include <Eigen/../unsupported/Eigen/KroneckerProduct>


template <class __T>
void test_generateProjectedPoints()
{
    Eigen::Matrix3<__T> R0;
    Eigen::Vector3<__T> t0;
    Eigen::Matrix3<__T> K;
    K.setZero();
    std::vector< Eigen::Vector3<__T> > world_pt0;
    std::vector< Eigen::Vector2<__T> > image_pt0;
    const std::string filename = "../data/pnp_data1.txt";
    readPnPdata(filename, R0, t0, K, world_pt0, image_pt0);

    std::vector< Eigen::Vector2<__T> > image_pt;
    std::vector<__T> s;
    generateProjectedPoints<__T>(image_pt, s, world_pt0, K, R0, t0);

    for(int i = 0; i < image_pt0.size(); ++i) {
        std::cout.precision(17);
        std::cout << "image_pt0: " << image_pt0[i].transpose() << std::endl;
        std::cout << "image_pt: " << image_pt[i].transpose() << std::endl << std::endl << std::endl;
    }
}

template <class __T>
struct dataBundle {
    Eigen::Matrix3<__T> __R0;
    Eigen::Vector3<__T> __t0;
    Eigen::Matrix3<__T> __K;
    std::vector<Eigen::Vector3<__T>> __world_pt0;
    std::vector<Eigen::Vector2<__T>> __image_pt0;
    std::vector<Eigen::Vector3<__T>> __rr0;
    std::vector<Eigen::Vector3<__T>> __bb0;
    std::vector<Eigen::Vector3<__T>> __nv0;
};

void *bundle_ex = nullptr;

template <class __T>
void test_pnp_WQD_init(const std::string& filename)
{
    assert(bundle_ex != nullptr);
    dataBundle<__T> *bundle = reinterpret_cast< dataBundle<__T>* > (bundle_ex);
    if(bundle->__image_pt0.size() < 3)
    {
        bundle->__K.setZero();
        readPnPdata(filename, bundle->__R0, bundle->__t0, bundle->__K, bundle->__world_pt0, bundle->__image_pt0);
    }
}

template <class __T>
void test_pnp_WQD(const bool& verbose,
                  const bool& use_opencv,
                  const __T problem_scale)
{
    assert(bundle_ex != nullptr);
    dataBundle<__T> *bundle = reinterpret_cast< dataBundle<__T>* > (bundle_ex);
    Eigen::Matrix3<__T> R0 = bundle->__R0;
    Eigen::Vector3<__T> t0 = bundle->__t0;
    Eigen::Matrix3<__T> K = bundle->__K;
    std::vector< Eigen::Vector3<__T> > world_pt0 = bundle->__world_pt0;
    std::vector< Eigen::Vector2<__T> > image_pt0 = bundle->__image_pt0;

    clock_t time1 = clock();
    Eigen::Matrix4<__T> XX;
    XX << R0, t0, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;

    Eigen::Matrix<__T, 4, 64> W;
    Eigen::Matrix<__T, 4, 4> Q;
    Eigen::Matrix<__T, 3, 37> D;
    Eigen::Matrix<__T, 4, 24> coef_f_q_sym;
    Eigen::Matrix<__T, 1, 70> coef_J_pure;
    Eigen::Matrix<__T, 3, 10> coefs_tq;
    Eigen::Matrix<__T, 3, 3> pinvG;
    pnp_WQD(W, Q, D, coef_f_q_sym, coef_J_pure, coefs_tq, pinvG, image_pt0, world_pt0, K, problem_scale);

    Eigen::Matrix<__T, 3, 64> W_ = W.topRows(3);
    Eigen::Matrix<__T, 3, 4> Q_ = Q.topRows(3);
    W_.row(0) = W.row(0) + W.row(1) + W.row(2);
    W_.row(1) = W.row(1) + W.row(2) + W.row(3);
    W_.row(2) = W.row(2) + W.row(3) + W.row(0);

    Q_.row(0) = Q.row(0) + Q.row(1) + Q.row(2);
    Q_.row(1) = Q.row(1) + Q.row(2) + Q.row(3);
    Q_.row(2) = Q.row(2) + Q.row(3) + Q.row(0);
    clock_t time2 = clock();
    double timeDataPrepare = ((double)(time2 - time1)) / double(CLOCKS_PER_SEC);

    Eigen::Matrix3<__T> R;
    Eigen::Vector3<__T> t;
    Eigen::Matrix4<__T> X;
    __T min[27];
    struct QPEP_options opt;
    opt.ModuleName = "solver_WQ_1_2_3_4_5_9_13_17_33_49_approx";
    opt.DecompositionMethod = "PartialPivLU";

    struct QPEP_runtime stat = QPEP_WQ_grobner<__T>(R, t, X, min, W_, Q_,
                    solver_WQ_1_2_3_4_5_9_13_17_33_49_approx,
                    mon_J_pure_pnp_func,
                    t_pnp_func,
                    coef_J_pure, coefs_tq, pinvG, nullptr, opt);

    Eigen::Vector4<__T> q0 = R2q(R);
    stat = QPEP_lm_single<__T>(R, t, X, q0, 100, 5e-2,
                   eq_Jacob_pnp_func,
                   t_pnp_func,
                   coef_f_q_sym, coefs_tq, pinvG, stat);

    if(verbose) {
        std::cout << "Time DataPrepare: " << timeDataPrepare << " s " << std::endl;
        std::cout << "Time DecompositionDataPrepare: " << stat.timeDecompositionDataPrepare << " s " << std::endl;
        std::cout << "Time Decomposition: " << stat.timeDecomposition << " s " << std::endl;
        std::cout << "Time Grobner: " << stat.timeGrobner << " s " << std::endl;
        std::cout << "Time Eigen: " << stat.timeEigen << " s " << std::endl;
        std::cout << "Time LM: " << stat.timeLM << " s " << std::endl;
        std::cout << "True X: " << std::endl << XX << std::endl;
        std::cout << "QPEP X: " << std::endl << X << std::endl << std::endl;
    }

    if(!use_opencv)
        return;


#ifdef USE_OPENCV
    std::vector< cv::Point_<__T> > image_pt0_cv = Vector2ToPoint2<__T>(image_pt0);
    std::vector< cv::Point3_<__T> > world_pt0_cv = Vector3ToPoint3<__T>(world_pt0);
    cv::Mat intrinsics, distCoeffs;
    int cv_type;
    if(!strcmp(typeid(__T).name(), "double"))
        cv_type = CV_64FC1;
    else if(!strcmp(typeid(__T).name(), "float"))
        cv_type = CV_32FC1;

    distCoeffs = cv::Mat::zeros(4, 1, cv_type);
    for (int i = 0; i < 3; i++)
        distCoeffs.at<__T>(i, 0) = 0.0;

    intrinsics.create(3, 3, cv_type);
    intrinsics.at<__T>(0, 0) = K(0, 0);
    intrinsics.at<__T>(1, 1) = K(1, 1);
    intrinsics.at<__T>(0, 2) = K(0, 2);
    intrinsics.at<__T>(1, 2) = K(1, 2);
    intrinsics.at<__T>(2, 2) = 1;

    cv::Mat rvec, tvec;
    cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    cv::Mat rotation_matrix;
    rotation_matrix = cv::Mat(3, 3, cv_type, cv::Scalar::all(0));
    cv::Rodrigues(rvec, rotation_matrix);
    Eigen::Matrix3<__T> Rot;
    Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
            rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
            rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
    Eigen::Vector3<__T> Trans;
    Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
    XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
    if(verbose) {
        std::cout << "Opencv EPnP X: " << std::endl << XX << std::endl;
    }


    if(world_pt0.size() == 4)
    {
        cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_P3P);
        cv::Rodrigues(rvec, rotation_matrix);
        Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
                rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
                rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
        Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
        XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
        if(verbose) {
            std::cout << "Opencv P3P X: " << std::endl << XX << std::endl;
        }
    }


    cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_DLS);
    cv::Rodrigues(rvec, rotation_matrix);
    Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
            rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
            rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
    Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
    XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
    if(verbose) {
        std::cout << "Opencv DLS X: " << std::endl << XX << std::endl;
    }


    cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_UPNP);
    cv::Rodrigues(rvec, rotation_matrix);
    Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
            rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
            rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
    Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
    XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
    if(verbose) {
        std::cout << "Opencv UPnP X: " << std::endl << XX << std::endl;
    }

#ifdef USE_OPENCV4
    if(world_pt0.size() == 4)
    {
        cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_AP3P);
        cv::Rodrigues(rvec, rotation_matrix);
        Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
                rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
                rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
        Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
        XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
        if(verbose) {
            std::cout << "Opencv AP3P X: " << std::endl << XX << std::endl;
        }

        cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
        cv::Rodrigues(rvec, rotation_matrix);
        Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
                rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
                rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
        Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
        XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
        if(verbose) {
            std::cout << "Opencv IPPE X: " << std::endl << XX << std::endl;
        }


        cv::solvePnP(cv::Mat(world_pt0_cv), cv::Mat(image_pt0_cv), intrinsics, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
        cv::Rodrigues(rvec, rotation_matrix);
        Rot << rotation_matrix.at<__T>(0, 0), rotation_matrix.at<__T>(0, 1), rotation_matrix.at<__T>(0, 2),
                rotation_matrix.at<__T>(1, 0), rotation_matrix.at<__T>(1, 1), rotation_matrix.at<__T>(1, 2),
                rotation_matrix.at<__T>(2, 0), rotation_matrix.at<__T>(2, 1), rotation_matrix.at<__T>(2, 2);
        Trans << tvec.at<__T>(0), tvec.at<__T>(1), tvec.at<__T>(2);
        XX << Rot, Trans, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;
        if(verbose) {
            std::cout << "Opencv IPPE Square X: " << std::endl << XX << std::endl;
        }
    }
#endif
#endif
}


template <class __T>
void test_pTop_WQD_init(const std::string& filename)
{
    assert(bundle_ex != nullptr);
    dataBundle<__T> *bundle = reinterpret_cast< dataBundle<__T>* > (bundle_ex);
    if(bundle->__rr0.size() < 3) {
        readpTopdata(filename, bundle->__R0, bundle->__t0, bundle->__rr0, bundle->__bb0, bundle->__nv0);
    }
}

template <class __T>
void test_pTop_WQD(const bool& verbose) {
    assert(bundle_ex != nullptr);
    dataBundle<__T> *bundle = reinterpret_cast< dataBundle<__T>* > (bundle_ex);
    std::vector<Eigen::Vector3<__T>> rr0 = bundle->__rr0;
    std::vector<Eigen::Vector3<__T>> bb0 = bundle->__bb0;
    std::vector<Eigen::Vector3<__T>> nv0 = bundle->__nv0;
    Eigen::Matrix4<__T> XX;
    XX << bundle->__R0, bundle->__t0, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;

    Eigen::Matrix<__T, 4, 64> W;
    Eigen::Matrix<__T, 4, 4> Q;
    Eigen::Matrix<__T, 3, 28> D;
    Eigen::Matrix<__T, 3, 9> G;
    Eigen::Vector3<__T> c;
    Eigen::Matrix<__T, 4, 24> coef_f_q_sym;
    Eigen::Matrix<__T, 1, 85> coef_J_pure;
    Eigen::Matrix<__T, 3, 11> coefs_tq;
    Eigen::Matrix<__T, 3, 3> pinvG;
    pTop_WQD(W, Q, D, G, c, coef_f_q_sym, coef_J_pure, coefs_tq, pinvG, rr0, bb0, nv0);

    Eigen::Matrix<__T, 3, 64> W_ = W.topRows(3);
    Eigen::Matrix<__T, 3, 4> Q_ = Q.topRows(3);
    W_.row(0) = W.row(0) + W.row(1) + W.row(2);
    W_.row(1) = W.row(1) + W.row(2) + W.row(3);
    W_.row(2) = W.row(2) + W.row(3) + W.row(0);

    Q_.row(0) = Q.row(0) + Q.row(1) + Q.row(2);
    Q_.row(1) = Q.row(1) + Q.row(2) + Q.row(3);
    Q_.row(2) = Q.row(2) + Q.row(3) + Q.row(0);

    Eigen::Matrix3<__T> R;
    Eigen::Vector3<__T> t;
    Eigen::Matrix4<__T> X;
    __T min[27];
    struct QPEP_options opt;
    opt.ModuleName = "solver_WQ_approx";
    opt.DecompositionMethod = "PartialPivLU";

    struct QPEP_runtime stat =
            QPEP_WQ_grobner<__T>(R, t, X, min, W_, Q_,
                    solver_WQ_approx,
                    mon_J_pure_pTop_func,
                    t_pTop_func,
                    coef_J_pure, coefs_tq, pinvG, nullptr, opt);

    Eigen::Vector4<__T> q0 = R2q(R);

    stat = QPEP_lm_fsolve<__T>(R, t, X, q0, 100, 5e-2,
                   eq_Jacob_pTop_func,
                   t_pTop_func,
                   coef_f_q_sym, coefs_tq, pinvG, W, Q, stat);

    if(verbose)
    {
        std::cout << "True X: " << std::endl << XX << std::endl;
        std::cout << "QPEP X: " << std::endl << X << std::endl << std::endl;
    }
}


#ifdef USE_OPENCV

template <class __T>
Eigen::MatrixX<__T> covx(const std::vector<Eigen::MatrixX<__T>>& x,
                     const std::vector<Eigen::MatrixX<__T>>& y)
{
    int len = x.size();
    std::vector<Eigen::MatrixX<__T>> res(len);
    Eigen::MatrixX<__T> mean_x = mean(x);
    Eigen::MatrixX<__T> mean_y = mean(y);
    for(int i = 0; i < len; ++i)
    {
        res[i].resize(x[0].rows(), y[0].rows());
        res[i] = (x[i] - mean_x) * (y[i] - mean_y).transpose();
    }
    return mean(res);
}

template <class __T>
void test_pTop_noise_init(const std::string& name)
{
    assert(bundle_ex != nullptr);
    dataBundle<__T> *bundle = reinterpret_cast< dataBundle<__T>* > (bundle_ex);
    if(bundle->__rr0.size() < 3) {
        readpTopdata(name, bundle->__R0, bundle->__t0, bundle->__rr0, bundle->__bb0, bundle->__nv0);
    }
}

template <class __T>
void test_pTop_noise(cv::Mat& img,
                     const int& num,
                     const __T& noise,
                     const __T& fontsize,
                     const bool& verbose) {
    assert(bundle_ex != nullptr);
    dataBundle<__T> *bundle = reinterpret_cast< dataBundle<__T>* > (bundle_ex);
    std::vector<Eigen::Vector3<__T>> rr0 = bundle->__rr0;
    std::vector<Eigen::Vector3<__T>> bb0 = bundle->__bb0;
    std::vector<Eigen::Vector3<__T>> nv0 = bundle->__nv0;
    Eigen::Matrix4<__T> XX;
    XX << bundle->__R0, bundle->__t0, Eigen::Vector3<__T>::Zero(3).transpose(), 1.0;

    std::vector<Eigen::MatrixX<__T>> qs(num);
    std::vector<Eigen::MatrixX<__T>> ys(num);
    std::vector<Eigen::MatrixX<__T> > vs(num);
    std::vector<Eigen::MatrixX<__T> > ds(num);
    std::vector<Eigen::MatrixX<__T> > gs(num);
    std::vector<Eigen::MatrixX<__T> > cs(num);
    std::vector<Eigen::MatrixX<__T> > ts(num);

    for(int j = 0; j < num; ++j) {
        std::vector<Eigen::Vector3<__T>> rr(rr0);
        std::vector<Eigen::Vector3<__T>> bb(bb0);
        std::vector<Eigen::Vector3<__T>> nv(nv0);
        for (int i = 0; i < rr.size(); ++i) {
            rr[i] += noise * Eigen::Vector3<__T>::Random();
            bb[i] += noise * Eigen::Vector3<__T>::Random();
            nv[i] += noise * Eigen::Vector3<__T>::Random();
        }

        Eigen::Matrix<__T, 4, 64> W;
        Eigen::Matrix<__T, 4, 4> Q;
        Eigen::Matrix<__T, 3, 28> D;
        Eigen::Matrix<__T, 3, 9> G;
        Eigen::Vector3<__T> c;
        Eigen::Matrix<__T, 4, 24> coef_f_q_sym;
        Eigen::Matrix<__T, 1, 85> coef_J_pure;
        Eigen::Matrix<__T, 3, 11> coefs_tq;
        Eigen::Matrix<__T, 3, 3> pinvG;
        pTop_WQD(W, Q, D, G, c, coef_f_q_sym, coef_J_pure, coefs_tq, pinvG, rr, bb, nv);

        Eigen::Matrix<__T, 3, 64> W_ = W.topRows(3);
        Eigen::Matrix<__T, 3, 4> Q_ = Q.topRows(3);
        W_.row(0) = W.row(0) + W.row(1) + W.row(2);
        W_.row(1) = W.row(1) + W.row(2) + W.row(3);
        W_.row(2) = W.row(2) + W.row(3) + W.row(0);

        Q_.row(0) = Q.row(0) + Q.row(1) + Q.row(2);
        Q_.row(1) = Q.row(1) + Q.row(2) + Q.row(3);
        Q_.row(2) = Q.row(2) + Q.row(3) + Q.row(0);

        Eigen::Matrix3<__T> R;
        Eigen::Vector3<__T> t;
        Eigen::Matrix4<__T> X;
        __T min[27];
        struct QPEP_options opt;
        opt.ModuleName = "solver_WQ_approx";
        opt.DecompositionMethod = "PartialPivLU";

        struct QPEP_runtime stat =
                QPEP_WQ_grobner<__T>(R, t, X, min, W_, Q_,
                                solver_WQ_approx,
                                mon_J_pure_pTop_func,
                                t_pTop_func,
                                coef_J_pure, coefs_tq, pinvG, nullptr, opt);

        Eigen::Vector4<__T> q0 = R2q(R);

        stat = QPEP_lm_fsolve<__T>(R, t, X, q0, 100, 5e-2,
                              eq_Jacob_pTop_func,
                              t_pTop_func,
                              coef_f_q_sym, coefs_tq, pinvG, W, Q, stat);

        q0 = R2q(R);
        qs[j].resize(4, 1);
        qs[j] = q0;
        ts[j].resize(3, 1);
        ts[j] = t;
        Eigen::Matrix<__T, 28, 1> v;
        Eigen::Matrix<__T, 9, 1> y;
        v.setZero();
        y.setZero();
        v_func_pTop(v, q0);
        y_func_pTop(y, q0);
        vs[j].resize(28, 1);
        vs[j] = v;
        ys[j].resize(9, 1);
        ys[j] = y;
        ds[j].resize(3 * 28, 1);
        ds[j] = vec<__T>(D);
        gs[j].resize(3 * 9, 1);
        gs[j] = vec<__T>(G);
        cs[j].resize(3, 1);
        cs[j] = c;

        if (verbose) {
            std::cout << "True X: " << std::endl << XX << std::endl;
            std::cout << "QPEP X: " << std::endl << X << std::endl << std::endl;
        }
    }

    Eigen::Vector4<__T> mean_q = mean< Eigen::MatrixX<__T> >(qs);

//    std::vector<Eigen::MatrixX<__T>> qs_(qs);
//    std::for_each(qs_.begin(), qs_.end(), [=](Eigen::Vector4<__T>& x){x = x - mean_q;});
//    Eigen::Matrix4<__T> res(Eigen::Matrix4<__T>::Zero());
//    std::for_each(qs_.begin(), qs_.end(), [&res](const Eigen::Vector4<__T>& x){res += x * x.transpose();});
//    Eigen::Matrix4<__T> Sigma_data_stat_ = res * 1.0 / ((double) qs_.size());
    Eigen::Matrix4<__T> Sigma_q_stat = covx(qs, qs);
    Eigen::MatrixX<__T> Sigma_t_stat = covx(ts, ts);
    Eigen::MatrixX<__T> Sigma_v_stat = covx(vs, vs);
    Eigen::MatrixX<__T> Sigma_y_stat = covx(ys, ys);
    Eigen::MatrixX<__T> Sigma_d_stat = covx(ds, ds);
    Eigen::MatrixX<__T> Sigma_g_stat = covx(gs, gs);
    Eigen::MatrixX<__T> Sigma_c_stat = covx(cs, cs);
    Eigen::MatrixX<__T> Sigma_gc_stat = covx(gs, cs);
    Eigen::MatrixX<__T> Sigma_dg_stat = covx(ds, gs);
    Eigen::MatrixX<__T> Sigma_dc_stat = covx(ds, cs);
    Eigen::MatrixX<__T> Sigma_cg_stat = Sigma_gc_stat.transpose();
    Eigen::MatrixX<__T> Sigma_gd_stat = Sigma_dg_stat.transpose();
    Eigen::MatrixX<__T> Sigma_cd_stat = Sigma_dc_stat.transpose();

    std::vector<Eigen::Vector3<__T>> rr(rr0);
    std::vector<Eigen::Vector3<__T>> bb(bb0);
    std::vector<Eigen::Vector3<__T>> nv(nv0);
    for (int i = 0; i < rr.size(); ++i) {
        rr[i] += noise * Eigen::Vector3<__T>::Random();
        bb[i] += noise * Eigen::Vector3<__T>::Random();
        nv[i] += noise * Eigen::Vector3<__T>::Random();
    }

    Eigen::Matrix<__T, 4, 64> W;
    Eigen::Matrix<__T, 4, 4> Q;
    Eigen::Matrix<__T, 3, 28> D;
    Eigen::Matrix<__T, 3, 9> G;
    Eigen::Vector3<__T> c;
    Eigen::Matrix<__T, 4, 24> coef_f_q_sym;
    Eigen::Matrix<__T, 1, 85> coef_J_pure;
    Eigen::Matrix<__T, 3, 11> coefs_tq;
    Eigen::Matrix<__T, 3, 3> pinvG;
    pTop_WQD<__T>(W, Q, D, G, c, coef_f_q_sym, coef_J_pure, coefs_tq, pinvG, rr, bb, nv);

    Eigen::Matrix<__T, 3, 64> W_ = W.topRows(3);
    Eigen::Matrix<__T, 3, 4> Q_ = Q.topRows(3);
    W_.row(0) = W.row(0) + W.row(1) + W.row(2);
    W_.row(1) = W.row(1) + W.row(2) + W.row(3);
    W_.row(2) = W.row(2) + W.row(3) + W.row(0);

    Q_.row(0) = Q.row(0) + Q.row(1) + Q.row(2);
    Q_.row(1) = Q.row(1) + Q.row(2) + Q.row(3);
    Q_.row(2) = Q.row(2) + Q.row(3) + Q.row(0);

    Eigen::Matrix3<__T> R;
    Eigen::Vector3<__T> t;
    Eigen::Matrix4<__T> X;
    __T min[27];
    struct QPEP_options opt;
    opt.ModuleName = "solver_WQ_approx";
    opt.DecompositionMethod = "PartialPivLU";

    struct QPEP_runtime stat =
            QPEP_WQ_grobner<__T>(R, t, X, min, W_, Q_,
                            solver_WQ_approx,
                            mon_J_pure_pTop_func,
                            t_pTop_func,
                            coef_J_pure, coefs_tq, pinvG, nullptr, opt);

    Eigen::Vector4<__T> q0 = R2q(R);

    stat = QPEP_lm_fsolve<__T>(R, t, X, q0, 100, 5e-2,
                          eq_Jacob_pTop_func,
                          t_pTop_func,
                          coef_f_q_sym, coefs_tq, pinvG, W, Q, stat);
    Eigen::Vector4<__T> q = R2q(R);
    Eigen::Matrix<__T, 28, 1> v;
    Eigen::Matrix<__T, 9, 1> y;
    v.setZero();
    y.setZero();
    v_func_pTop(v, q);
    y_func_pTop(y, q);
    Eigen::MatrixX<__T> V_cal = Eigen::kroneckerProduct(v.transpose(), Eigen::Matrix3<__T>::Identity());
    Eigen::MatrixX<__T> Y_cal = Eigen::kroneckerProduct(y.transpose(), Eigen::Matrix3<__T>::Identity());
    Eigen::MatrixX<__T> partial_v_q_sym_val, partial_y_q_sym_val;
    partial_v_q_sym_val.resize(28, 4);
    partial_y_q_sym_val.resize(9, 4);
    partial_v_q_sym_val.setZero();
    partial_y_q_sym_val.setZero();
    jv_func_pTop(partial_v_q_sym_val, q);
    jy_func_pTop(partial_y_q_sym_val, q);
    Eigen::MatrixX<__T> F = D * partial_v_q_sym_val + G * partial_y_q_sym_val;
    Eigen::MatrixX<__T> cov_left = V_cal * Sigma_d_stat * V_cal.transpose() + Y_cal * Sigma_g_stat * Y_cal.transpose() + Sigma_c_stat +
                               V_cal * Sigma_dg_stat * Y_cal.transpose() + V_cal * Sigma_dc_stat + Y_cal * Sigma_gc_stat +
                               Sigma_cd_stat * V_cal.transpose() + Sigma_cg_stat * Y_cal.transpose() + Y_cal * Sigma_gd_stat * V_cal.transpose();

    clock_t time1 = clock();
    Eigen::Matrix4<__T> cov;
    Eigen::MatrixX<__T> cov_left_ = cov_left;
    __T cov_tr = cov_left_.trace();
    __T scaling = 1.0 / (cov_tr);
    cov_left_ *= scaling;
    csdp_cov<__T>(cov, F, cov_left_, q);
    cov = cov / scaling;
    clock_t time2 = clock();
    std::cout << "Estimated Covariance:" << std::endl << cov << std::endl;

    std::vector<Eigen::Vector4<__T>> qs__(num);
    for(int i = 0; i < num; ++i)
    {
        qs__[i] = qs[i];
    }
    __T scale = fabs(cov_left.trace() / (F * cov * F.transpose()).trace());
    plotQuatCov<__T>(img, Sigma_q_stat, scale * cov, qs__, mean_q, fontsize);

    std::cout << "Stat Covariance:" << std::endl << Sigma_q_stat << std::endl;
    std::cout << "Time CSDP Covariance Estimation: " << (time2 - time1) / double(CLOCKS_PER_SEC) << std::endl;
}
#endif


enum TestMethods {
    METHOD_PNP = 1,
    METHOD_PTOP,
} method;




int test_double(int argc,char ** argv) {

    double time = 0.0;
    clock_t time1 = clock(), time2;
    double loops = 100.0;
    
    //TODO: Change this to METHOD_PTOP
    //      if you need to test Point-to-Plane Registration
    method = METHOD_PNP;

#ifndef NO_OMP
    num_threads_ = omp_get_max_threads();
    omp_set_num_threads(num_threads_);
    Eigen::initParallel();
    Eigen::setNbThreads(num_threads_);
#endif

    dataBundle<double> bundle;
    bundle_ex = &bundle;

#ifdef USE_OPENCV
    int row, col;
    getScreenResolution(col, row);
    col = MIN(col, row) * 0.9;
    row = col;
#ifdef OSX_BIG_SUR
    col *= 2;
    row *= 2;
#endif
    double fontsize = row / 1920.0;
    cv::Mat imageDraw = cv::Mat::zeros(row, col, CV_8UC3);
    cv::Mat ColorMask(row, col, CV_8UC3, cv::Scalar(1, 1, 1) * 255);
    cv::addWeighted(imageDraw, 0.0, ColorMask, 1.0, 0, imageDraw);

    test_pTop_noise_init<double>("../data/pTop_data-100pt-1.txt");
    test_pTop_noise<double>(imageDraw, 1500, 1e-5, fontsize, false);

    imshow("imageDraw", imageDraw);
    cv::waitKey(0);
#endif

    std::cout.precision(16);

    time1 = clock();
    loops = 1000.0;
    
    if(method == METHOD_PNP)
        test_pnp_WQD_init<double>("../data/pnp_data-500pt-1.txt");
    else
        test_pTop_WQD_init<double>("../data/pTop_data-4096pt-1.txt");

    {
#ifndef NO_OMP
#pragma omp parallel for num_threads(num_threads_)
#endif
        for(int i = 0; i < (int) loops; ++i)
        {
            if(method == METHOD_PNP)
                test_pnp_WQD<double>(true, false, 1e-8);
            else
                test_pTop_WQD<double>(true);
        }
    }

    time2 = clock();
    time = time2 - time1;
    std::cout << "Time: " << time / loops / double(CLOCKS_PER_SEC) << std::endl;

    return 0;
}



int test_float(int argc,char ** argv) {

    double time = 0.0;
    clock_t time1 = clock(), time2;
    double loops = 100.0;

    //TODO: Change this to METHOD_PTOP
    //      if you need to test Point-to-Plane Registration
    method = METHOD_PNP;

#ifndef NO_OMP
    num_threads_ = omp_get_max_threads();
    omp_set_num_threads(num_threads_);
    Eigen::initParallel();
    Eigen::setNbThreads(num_threads_);
#endif

    dataBundle<float> bundle;
    bundle_ex = &bundle;

#ifdef USE_OPENCV
    int row, col;
    getScreenResolution(col, row);
    col = MIN(col, row) * 0.9;
    row = col;
    float fontsize = row / 1920.0;
    cv::Mat imageDraw = cv::Mat::zeros(row, col, CV_8UC3);
    cv::Mat ColorMask(row, col, CV_8UC3, cv::Scalar(1, 1, 1) * 255);
    cv::addWeighted(imageDraw, 0.0, ColorMask, 1.0, 0, imageDraw);

    test_pTop_noise_init<float>("../data/pTop_data-100pt-1.txt");
    test_pTop_noise<float>(imageDraw, 1500, 3e-2, fontsize, false);

    imshow("imageDraw", imageDraw);
    cv::waitKey(0);
#endif

    std::cout.precision(16);

    time1 = clock();
    loops = 1000.0;

    if(method == METHOD_PNP)
        test_pnp_WQD_init<float>("../data/pnp_data-500pt-1.txt");
    else
        test_pTop_WQD_init<float>("../data/pTop_data-4096pt-1.txt");

    {
#ifndef NO_OMP
#pragma omp parallel for num_threads(num_threads_)
#endif
        for(int i = 0; i < (int) loops; ++i)
        {
            if(method == METHOD_PNP)
                test_pnp_WQD<float>(true, false, 1e-7);
            else
                test_pTop_WQD<float>(true);
        }
    }

    time2 = clock();
    time = time2 - time1;
    std::cout << "Time: " << time / loops / double(CLOCKS_PER_SEC) << std::endl;

    return 0;
}

int main(int argc, char**argv)
{
    test_float(argc, argv);
    test_double(argc, argv);
    return 0;
}
