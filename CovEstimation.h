//
// Created by Jin Wu on 6/11/2020.
//

#ifndef LIBQPEP_COVESTIMATION_H
#define LIBQPEP_COVESTIMATION_H


#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <Eigen/Sparse>


template <class __T>
void csdp_cov(Eigen::Matrix4<__T>& cov,
              const Eigen::MatrixX<__T>& F,
              const Eigen::Matrix3<__T>& cov_left,
              const Eigen::Vector4<__T>& q);

#endif //LIBQPEP_COVESTIMATION_H
