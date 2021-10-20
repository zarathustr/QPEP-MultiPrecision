//
// Created by Jin Wu on 3/11/2020.
//

#ifndef LIBQPEP_SOLVER_HELPER_H
#define LIBQPEP_SOLVER_HELPER_H


#include <time.h>
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <Eigen/Sparse>
#include "QPEP.h"

template <class __T>
void data_func_solver_WQ_func(Eigen::SparseMatrix<__T>& C1,
                              Eigen::MatrixX<__T>& C2,
                              const Eigen::VectorX<__T>& data);

#endif //LIBQPEP_SOLVER_HELPER_H
