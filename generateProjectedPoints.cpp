// LibQPEP: A Library for Globally Optimal Solving Quadratic Pose Estimation Problems,
//          It also gives highly accurate uncertainty description of the solutions.
//
// Authors: Jin Wu and Ming Liu
// Affiliation: Hong Kong University of Science and Technology (HKUST)
// Emails: jin_wu_uestc@hotmail.com; eelium@ust.hk
//
//
// generateProjectedPoints.cpp: Generate image points for PnP problems


#include "generateProjectedPoints.h"

template <class __T>
void generateProjectedPoints(std::vector<Eigen::Vector2<__T> >& image_pt,
                             std::vector<__T>& s,
                             const std::vector<Eigen::Vector3<__T> >& world_pt,
                             const Eigen::Matrix3<__T>& K,
                             const Eigen::Matrix3<__T>& R,
                             const Eigen::Vector3<__T>& t)
{
    int numPoints = world_pt.size();

    for(int i = 0; i < numPoints; ++i)
    {
        s.push_back(0);
    }

    for(int i = 0; i < numPoints; ++i)
    {
        Eigen::Vector4<__T> world_point(world_pt[i](0), world_pt[i](1), world_pt[i](2), 1);
        Eigen::Matrix<__T, 3, 4> cameraMatrix;
        cameraMatrix << R.transpose(), t;
        cameraMatrix = K * cameraMatrix;
        Eigen::Vector3<__T> projectedPoint = cameraMatrix * world_point;
        s[i] = projectedPoint(2);
        Eigen::Vector2<__T> projectedPoint_(projectedPoint(0), projectedPoint(1));
        image_pt.push_back(projectedPoint_ / s[i]);
    }
}
