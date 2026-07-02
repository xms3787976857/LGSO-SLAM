#pragma once
#ifndef LGSO_CAMERA_H_
#define LGSO_CAMERA_H_

#include <memory>

#include "util/NumType.h"

namespace dso {
    struct CalibHessian;

    /**
     * @brief Pinhole camera model (LDSO compatibility wrapper)
     */
    struct LDSOCamera {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LDSOCamera(double fx, double fy, double cx, double cy) : fx(fx), fy(fy), cx(cx), cy(cy) {}

        double fx = 0, fy = 0, cx = 0, cy = 0;
        std::shared_ptr<CalibHessian> mpCH = nullptr;
    };
}
#endif
