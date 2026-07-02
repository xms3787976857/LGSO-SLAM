/**
 * Lightweight color-based image segmentation for SLAM smoothness prior.
 *
 * Uses a simplified SLIC-like approach: clusters initialized on a regular
 * grid, each pixel searches only its local neighborhood (2S×2S) for the
 * nearest cluster center in (R,G,B,X,Y) space. Designed to run in <1ms
 * on GPU for 1200×680 images.
 */

#pragma once

#include <torch/torch.h>

/**
 * Segment an RGB image into superpixel-like regions.
 *
 * @param image      [3, H, W] float32, values in [0, 1] (RGB order)
 * @param num_seeds  desired number of superpixels (actual may differ slightly)
 * @param compactness  weight for spatial term (higher = more compact regions)
 * @param num_iters  number of k-means iterations (3-5 is sufficient)
 * @return  [H, W] int64 label map, labels are 0..N-1
 */
torch::Tensor segmentImage(
    torch::Tensor image,
    int num_seeds = 200,
    float compactness = 20.0f,
    int num_iters = 4);
