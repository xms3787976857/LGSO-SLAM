#pragma once

#include <memory>
#include <set>

#include "util/NumType.h"


namespace dso {

    struct Frame;
    struct Feature;
	struct PointHessian;


    /**
     * @brief Point represents a 3D map point.
     * New map points are generated from immature features in the activiate function of FullSystem
     * NOTE we current don't clean the outlier map points, do this if you really want
     */
    struct Point {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
         * point can only be created with a host feature
         * it will try to create a point hessian from the immature point stored in the feature
         * but if it doesn't have, will create an empty point hessian
         * @param hostFeature
         */
        Point(std::shared_ptr<Feature> hostFeature);

        Point();

        enum PointStatus {
            ACTIVE = 0,     // point is still active in optimization
            OUTLIER,        // considered as outlier in optimization
            OUT,            // out side the boundary
            MARGINALIZED    // marginalized, usually also out of boundary, but also can be set because the host frame is marged
        };  // the status of this point

        /**
         * release the point hessian
         */
        void ReleasePH();

        /**
         * compute 3D position in world
         */
        void ComputeWorldPos();

        // save and load
        void save(std::ofstream &fout);

        void load(std::ifstream &fin, std::vector<std::shared_ptr<Frame>> &allKFs);

        // =========================================================================================================
        unsigned long id = 0;              // id
        static unsigned long mNextId;
        PointStatus status = PointStatus::ACTIVE;  // status of this point
        Vec3 mWorldPos = Vec3::Zero();        // pos in world
        std::weak_ptr<Feature> mHostFeature;     // the hosting feature creating this point

        // internal structures
        std::shared_ptr<PointHessian> mpPH;  // point with hessians

    };

}