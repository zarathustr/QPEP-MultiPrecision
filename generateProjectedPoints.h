#ifndef LIBQPEP_GENERATEPROJECTEDPOINTS_H
#define LIBQPEP_GENERATEPROJECTEDPOINTS_H

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
void generateProjectedPoints(std::vector<Eigen::Vector2<__T> >& image_pt,
                             std::vector<__T>& s,
                             const std::vector<Eigen::Vector3<__T> >& world_pt,
                             const Eigen::Matrix3<__T>& K,
                             const Eigen::Matrix3<__T>& R,
                             const Eigen::Vector3<__T>& t);

#endif //LIBQPEP_GENERATEPROJECTEDPOINTS_H
