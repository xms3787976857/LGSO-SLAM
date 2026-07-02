#pragma once
#ifndef LOOPCLOSING_BRIDGE_H_
#define LOOPCLOSING_BRIDGE_H_

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace dso {

/**
 * @brief Bridge to LDSO loop closing module.
 *
 * All LDSO internals are hidden via PIMPL pattern.
 * The main DSO headers never include any LDSO headers.
 */
class LoopClosingBridge {
public:
    LoopClosingBridge();
    ~LoopClosingBridge();

    /**
     * Load ORB vocabulary. Must be called before addKeyFrame().
     * @return true on success
     */
    bool loadVocabulary(const std::string &vocPath);

    /**
     * Add a keyframe to the loop closing system.
     * Called from FullSystem::makeKeyFrame().
     * @param kfId         unique keyframe ID
     * @param camToWorld   4x4 pose matrix (column-major like Sophus)
     * @param image        input image (for ORB feature extraction)
     * @param fx, fy, cx, cy  camera intrinsics
     */
    void addKeyFrame(int kfId, const double* camToWorld,
                     const unsigned char* image, int width, int height,
                     double fx, double fy, double cx, double cy);

    /**
     * Set current slide-window frame IDs so loop closing can exclude them.
     * @param activeIds  list of currently active KF ids in the DSO window
     */
    void setActiveWindow(const std::vector<int> &activeIds);

    /**
     * Run final pose graph optimization after SLAM ends.
     */
    void finalize();

    /**
     * Get optimized (loop-corrected) pose for a keyframe.
     * @return true if optimized pose is available, false if not
     */
    bool getOptimizedPose(int kfId, double* camToWorld4x4) const;

    /**
     * Check if a loop was detected in the last addKeyFrame call.
     */
    bool loopDetected() const;

    /**
     * Get the number of keyframes in the global map.
     */
    int numKeyFrames() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace dso
#endif
