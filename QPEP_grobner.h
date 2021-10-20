//
// Created by Jin Wu on 1/11/2020.
//

#ifndef LIBQPEP_QPEP_GROBNER_H
#define LIBQPEP_QPEP_GROBNER_H

#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include "QPEP.h"

template <class __T>
using solver_func_handle = struct QPEP_runtime (*)(Eigen::MatrixX< std::complex<__T> >& sol,
        const Eigen::VectorX<__T>& data,
        const struct QPEP_options& opt);

template <class __T>
using mon_J_pure_func_handle =  std::vector<__T> (*)(const Eigen::Quaternion<__T>& q,
        const Eigen::Vector3<__T>& t);

template <class __T>
using t_func_handle = Eigen::Vector3<__T> (*)(const Eigen::MatrixX<__T>& pinvG,
        const Eigen::MatrixX<__T>& coefs_tq,
        const Eigen::Quaternion<__T>& q);

template <class __T>
using eq_Jacob_func_handle = void (*)(Eigen::Matrix<__T, 4, 1>& eq,
        Eigen::Matrix<__T, 4, 4>& Jacob,
        const Eigen::MatrixX<__T>& coef_f_q_sym,
        const Eigen::Vector4<__T>& q);

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
                                    const struct QPEP_options& opt);

#endif //LIBQPEP_QPEP_GROBNER_H
