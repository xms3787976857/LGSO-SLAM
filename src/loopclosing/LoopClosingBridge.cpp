#include "LoopClosingBridge.h"

// ===== LDSO internals (hidden from main DSO) =====
#include "ldso/NumTypes.h"
#include "ldso/Frame.h"
#include "ldso/Map.h"
#include "ldso/Feature.h"
#include "ldso/Point.h"
#include "ldso/Camera.h"
#include "ldso/LoopClosing.h"
#include "ldso/FeatureDetector.h"
#include "ldso/FeatureMatcher.h"
#include "ldso/GlobalCalib.h"
#include "ldso/CalibHessian.h"

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace dso {

struct LoopClosingBridge::Impl {
    // LDSO vocabulary
    std::shared_ptr<DBoW3::Vocabulary> voc;

    // LDSO global map
    std::shared_ptr<ldso::Map> globalMap;

    // LDSO loop closing
    std::shared_ptr<ldso::LoopClosing> loopClosing;

    // All keyframes in order
    std::vector<std::shared_ptr<ldso::Frame>> allKFs;

    // Current active window IDs
    std::vector<int> activeIds;

    // Camera intrinsics
    std::shared_ptr<ldso::Camera> camera;
    std::shared_ptr<ldso::internal::CalibHessian> Hcalib;

    // The fake fullsystem needed by LDSO LoopClosing
    // We only need vocab, globalMap, and Hcalib from it
    struct FakeFullSystem {
        std::shared_ptr<DBoW3::Vocabulary> vocab;
        std::shared_ptr<ldso::Map> globalMap;
        std::shared_ptr<ldso::internal::CalibHessian> Hcalib;
        std::shared_ptr<ldso::CoarseDistanceMap> coarseDistanceMap;

        std::shared_ptr<ldso::CoarseDistanceMap> GetDistanceMap() {
            return coarseDistanceMap;
        }

        std::vector<std::shared_ptr<ldso::Frame>> GetActiveFrames() {
            return {};
        }
    };
    std::shared_ptr<FakeFullSystem> fakeFS;

    bool lastLoopDetected = false;
    bool initialized = false;
};

LoopClosingBridge::LoopClosingBridge()
    : impl(std::make_unique<Impl>())
{
    impl->fakeFS = std::make_shared<Impl::FakeFullSystem>();
    impl->fakeFS->coarseDistanceMap = std::make_shared<ldso::CoarseDistanceMap>(
        ldso::internal::wG[0], ldso::internal::hG[0]);
}

LoopClosingBridge::~LoopClosingBridge() {
    if (impl->loopClosing) {
        impl->loopClosing->SetFinish(true);
    }
}

bool LoopClosingBridge::loadVocabulary(const std::string &vocPath) {
    impl->voc = std::make_shared<DBoW3::Vocabulary>();
    try {
        impl->voc->load(vocPath);
    } catch (...) {
        return false;
    }

    // Create global map
    impl->globalMap = std::make_shared<ldso::Map>(nullptr);

    // Create fake fullsystem shim
    impl->fakeFS->vocab = impl->voc;
    impl->fakeFS->globalMap = impl->globalMap;

    if (ldso::setting_enableLoopClosing) {
        impl->loopClosing = std::make_shared<ldso::LoopClosing>(
            reinterpret_cast<ldso::FullSystem*>(impl->fakeFS.get()));
    }

    impl->initialized = true;
    return true;
}

void LoopClosingBridge::addKeyFrame(int kfId, const double* camToWorld,
                                     const unsigned char* image, int width, int height,
                                     double fx, double fy, double cx, double cy) {
    if (!impl->initialized) return;

    // Create camera if not yet created
    if (!impl->camera) {
        impl->camera = std::make_shared<ldso::Camera>(fx, fy, cx, cy);
        impl->Hcalib = std::make_shared<ldso::internal::CalibHessian>(impl->camera);
        impl->fakeFS->Hcalib = impl->Hcalib;
    }

    // Create Frame
    auto frame = std::make_shared<ldso::Frame>(kfId);
    frame->kfId = kfId;

    // Set pose
    Eigen::Matrix<double, 3, 3> R;
    R << camToWorld[0], camToWorld[1], camToWorld[2],
         camToWorld[4], camToWorld[5], camToWorld[6],
         camToWorld[8], camToWorld[9], camToWorld[10];
    Eigen::Matrix<double, 3, 1> t(camToWorld[3], camToWorld[7], camToWorld[11]);
    Sophus::SE3d Tcw(R.transpose(), -(R.transpose() * t));
    frame->setPose(Tcw);
    frame->setPoseOpti(Sophus::Sim3d(Tcw.matrix()));

    // Process image if available
    if (image && width > 0 && height > 0) {
        frame->CreateFH(frame);
        auto& fh = frame->frameHessian;
        cv::Mat bgr(height, width, CV_8UC3, (void*)image);
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32FC1, 1.0/255.0);
        fh->makeImages((float*)gray.data, impl->Hcalib);
        ldso::FeatureDetector detector;
        detector.DetectCorners(2000, frame);
        if (impl->voc) frame->ComputeBoW(impl->voc);
    }

    impl->allKFs.push_back(frame);
    impl->globalMap->AddKeyFrame(frame);
    if (impl->loopClosing) impl->loopClosing->InsertKeyFrame(frame);
}


void LoopClosingBridge::setActiveWindow(const std::vector<int> &activeIds) {
    impl->activeIds = activeIds;
}

void LoopClosingBridge::finalize() {
    if (impl->loopClosing) {
        impl->loopClosing->SetFinish(true);
    }
    if (impl->globalMap) {
        impl->globalMap->lastOptimizeAllKFs();
        impl->globalMap->UpdateAllWorldPoints();
    }
}

bool LoopClosingBridge::getOptimizedPose(int kfId, double* camToWorld4x4) const {
    for (auto& f : impl->allKFs) {
        if (f->kfId == kfId) {
            Sophus::Sim3d Scw = f->getPoseOpti();
            Eigen::Matrix4d M = Scw.matrix();
            for (int i = 0; i < 16; i++)
                camToWorld4x4[i] = M(i / 4, i % 4);
            return true;
        }
    }
    return false;
}

bool LoopClosingBridge::loopDetected() const {
    return impl->lastLoopDetected;
}

int LoopClosingBridge::numKeyFrames() const {
    return impl->allKFs.size();
}

} // namespace dso
