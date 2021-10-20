#ifndef LIBQPEP_PTOP_WQD_H
#define LIBQPEP_PTOP_WQD_H

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
void pTop_WQD(Eigen::Matrix<__T, 4, 64>& W,
              Eigen::Matrix<__T, 4, 4>& Q,
              Eigen::Matrix<__T, 3, 28>& D,
              Eigen::Matrix<__T, 3, 9>& GG,
              Eigen::Vector3<__T>& c,
              Eigen::Matrix<__T, 4, 24>& coef_f_q_sym,
              Eigen::Matrix<__T, 1, 85>& coef_J_pure,
              Eigen::Matrix<__T, 3, 11>& coefs_tq,
              Eigen::Matrix<__T, 3, 3>& pinvG,
              const std::vector<Eigen::Vector3<__T>>& rr,
              const std::vector<Eigen::Vector3<__T>>& bb,
              const std::vector<Eigen::Vector3<__T>>& nv);

#endif //LIBQPEP_PTOP_WQD_H
