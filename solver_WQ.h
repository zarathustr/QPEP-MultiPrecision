//
// Created by Jin Wu on 3/11/2020.
//

#ifndef LIBQPEP_SOLVER_WQ_H
#define LIBQPEP_SOLVER_WQ_H


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
struct QPEP_runtime solver_WQ(Eigen::MatrixX< std::complex<__T> >& sol_,
                              const Eigen::VectorX<__T>& data,
                              const struct QPEP_options& opt);

#endif //LIBQPEP_SOLVER_WQ_H
