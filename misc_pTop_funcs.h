//
// Created by Jin Wu on 1/11/2020.
//

#ifndef LIBQPEP_MISC_PTOP_FUNCS_H
#define LIBQPEP_MISC_PTOP_FUNCS_H

#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion


template <class __T>
Eigen::Vector3<__T> t_pTop_func(const Eigen::MatrixX<__T>& pinvG,
                                const Eigen::MatrixX<__T>& coefs_tq,
                                const Eigen::Quaternion<__T>& q);

template <class __T>
std::vector<__T> mon_J_pure_pTop_func(const Eigen::Quaternion<__T>& q,
                                      const Eigen::Vector3<__T>& t);

template <class __T>
void eq_Jacob_pTop_func(Eigen::Matrix<__T, 4, 1>& eq,
                        Eigen::Matrix<__T, 4, 4>& Jacob,
                        const Eigen::MatrixX<__T>& coef_f_q_sym,
                        const Eigen::Vector4<__T>& q);

template <class __T>
void v_func_pTop(Eigen::Matrix<__T, 28, 1>& v,
                 const Eigen::Vector4<__T>& q);

template <class __T>
void y_func_pTop(Eigen::Matrix<__T, 9, 1>& y,
                 const Eigen::Vector4<__T>& q);

template <class __T>
void jv_func_pTop(Eigen::MatrixX<__T>& jv,
                  const Eigen::Vector4<__T>& q);

template <class __T>
void jy_func_pTop(Eigen::MatrixX<__T>& jy,
                  const Eigen::Vector4<__T>& q);



#endif //LIBQPEP_MISC_PTOP_FUNCS_H
