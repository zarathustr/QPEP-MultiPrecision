//
// Created by Jin Wu on 1/11/2020.
//

#ifndef LIBQPEP_QPEP_LM_SINGLE_H
#define LIBQPEP_QPEP_LM_SINGLE_H

#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include "QPEP_grobner.h"

template <class __T>
struct QPEP_runtime QPEP_lm_single(Eigen::Matrix3<__T>& R,
                                   Eigen::Vector3<__T>& t,
                                   Eigen::Matrix4<__T>& X,
                                   const Eigen::Vector4<__T>& q0,
                                   const int& max_iter,
                                   const __T& mu,
                                   const eq_Jacob_func_handle<__T>& eq_Jacob_func,
                                   const t_func_handle<__T>& t_func,
                                   const Eigen::MatrixX<__T>& coef_f_q_sym,
                                   const Eigen::MatrixX<__T>& coefs_tq,
                                   const Eigen::MatrixX<__T>& pinvG,
                                   const struct QPEP_runtime& stat_);

template <class __T>
struct QPEP_runtime QPEP_lm_fsolve(Eigen::Matrix3<__T>& R,
                                   Eigen::Vector3<__T>& t,
                                   Eigen::Matrix4<__T>& X,
                                   const Eigen::Vector4<__T>& q0,
                                   const int& max_iter,
                                   const __T& mu,
                                   const eq_Jacob_func_handle<__T>& eq_Jacob_func,
                                   const t_func_handle<__T>& t_func,
                                   const Eigen::MatrixX<__T>& coef_f_q_sym,
                                   const Eigen::MatrixX<__T>& coefs_tq,
                                   const Eigen::MatrixX<__T>& pinvG,
                                   const Eigen::MatrixX<__T>& W,
                                   const Eigen::MatrixX<__T>& Q,
                                   const struct QPEP_runtime& stat_);

#endif //LIBQPEP_QPEP_LM_SINGLE_H
