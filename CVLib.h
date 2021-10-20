#ifndef LIBQPEP_CVLIB_H
#define LIBQPEP_CVLIB_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <Eigen/Sparse>

#ifdef USE_OPENCV
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#if WIN32
#include <windows.h>
#else
#if !defined(OSX_10_9) && !defined(OSX_BIG_SUR)
#include <X11/Xlib.h>
#endif
#endif

void getScreenResolution(int &width, int &height);

template <class __T>
void plotCov4d(cv::Mat& img,
               __T& x_min,
               __T& x_max,
               __T& y_min,
               __T& y_max,
               const Eigen::Matrix4<__T>& cov,
               const std::vector< Eigen::Vector4<__T> >& data,
               const Eigen::Vector4<__T>& mean,
               const int& a,
               const int& b,
               const int& ellipse_size,
               const cv::Scalar& ellipse_color,
               const int& point_size,
               const cv::Scalar& point_color,
               const __T& size,
               const int& linestyle,
               const Eigen::Vector2<__T>& bias);

template <class __T>
void plotQuatCov(cv::Mat& img,
                 const Eigen::Matrix4<__T>& cov1,
                 const Eigen::Matrix4<__T>& cov2,
                 const std::vector<Eigen::Vector4<__T>>& qs,
                 const Eigen::Vector4<__T>& mean_q,
                 const __T& fontsize);

#endif

#endif //LIBQPEP_CVLIB_H
