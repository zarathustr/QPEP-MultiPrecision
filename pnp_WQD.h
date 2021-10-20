//
// Created by Jin Wu on 31/10/2020.
//

#ifndef LIBQPEP_PNP_WQD_H
#define LIBQPEP_PNP_WQD_H

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
void pnp_WQD(Eigen::Matrix<__T, 4, 64>& W,
             Eigen::Matrix<__T, 4, 4>& Q,
             Eigen::Matrix<__T, 3, 37>& D,
             Eigen::Matrix<__T, 4, 24>& coef_f_q_sym,
             Eigen::Matrix<__T, 1, 70>& coef_J_pure,
             Eigen::Matrix<__T, 3, 10>& coefs_tq,
             Eigen::Matrix<__T, 3, 3>& pinvG,
             const std::vector<Eigen::Vector2<__T> >& image_pt,
             const std::vector<Eigen::Vector3<__T> >& world_pt,
             const Eigen::Matrix3<__T>& K,
             const __T& scale);

#endif //LIBQPEP_PNP_WQD_H
