/**
 * Edge-based image segmentation for GS-SLAM smoothness prior.
 *
 * Simple two-step process:
 *   1. Detect edges via Sobel gradient magnitude
 *   2. Connected-components labeling on non-edge pixels
 *
 * No clustering, no k-means, no SLIC. Just edges → regions.
 */

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cfloat>

// ──── Sobel 3×3 gradient ─────────────────────────────────────

__global__ void sobel_edges(
    const float* __restrict__ img,     // [3, H, W], continuous memory
    float* __restrict__ grad_mag,      // [H, W]
    int H, int W)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u < 1 || u >= W - 1 || v < 1 || v >= H - 1) {
        if (u < W && v < H) grad_mag[v * W + u] = 0.0f;
        return;
    }

    // Use only green channel as luminance proxy (fastest)
    float g00 = img[1 * H * W + (v-1) * W + (u-1)];
    float g01 = img[1 * H * W + (v-1) * W +  u   ];
    float g02 = img[1 * H * W + (v-1) * W + (u+1)];
    float g10 = img[1 * H * W +  v    * W + (u-1)];
    float g12 = img[1 * H * W +  v    * W + (u+1)];
    float g20 = img[1 * H * W + (v+1) * W + (u-1)];
    float g21 = img[1 * H * W + (v+1) * W +  u   ];
    float g22 = img[1 * H * W + (v+1) * W + (u+1)];

    float gx = -g00 + g02 - 2.0f*g10 + 2.0f*g12 - g20 + g22;
    float gy = -g00 - 2.0f*g01 - g02 + g20 + 2.0f*g21 + g22;

    grad_mag[v * W + u] = sqrtf(gx*gx + gy*gy);
}

// ──── Threshold edges → binary edge map ───────────────────────

__global__ void threshold_edges(
    const float* __restrict__ grad_mag, // [H, W]
    int* __restrict__ edge_map,         // [H, W]  1=edge, 0=flat
    int H, int W, float thresh)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    float mag = grad_mag[v * W + u];
    edge_map[v * W + u] = (mag > thresh) ? 1 : 0;
}

// ──── Connected components: initialize labels ─────────────────

__global__ void cc_init(
    const int* __restrict__ edge_map,   // [H, W]
    int* __restrict__ labels,           // [H, W]
    int H, int W)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    int idx = v * W + u;
    if (edge_map[idx] == 1) {
        labels[idx] = -1;              // edge pixel → no region
    } else {
        labels[idx] = idx;             // flat pixel → its own root
    }
}

// ──── Connected components: union neighbors (1 iteration) ─────

__global__ void cc_union(
    int* __restrict__ labels,           // [H, W]
    int H, int W)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    int idx = v * W + u;
    if (labels[idx] < 0) return;  // edge pixel

    // Check right and down neighbors; union if both are non-edge
    if (u + 1 < W && labels[v * W + (u+1)] >= 0) {
        // Point both to the smaller root
        int a = labels[idx];
        int b = labels[v * W + (u+1)];
        if (a != b) {
            int root = min(a, b);
            labels[idx] = root;
            labels[v * W + (u+1)] = root;
        }
    }
    if (v + 1 < H && labels[(v+1) * W + u] >= 0) {
        int a = labels[idx];
        int b = labels[(v+1) * W + u];
        if (a != b) {
            int root = min(a, b);
            labels[idx] = root;
            labels[(v+1) * W + u] = root;
        }
    }
}

// ──── Connected components: path compression ──────────────────

__global__ void cc_flatten(
    int* __restrict__ labels,           // [H, W]
    int H, int W)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    int idx = v * W + u;
    if (labels[idx] < 0) return;

    // Find root by following parent pointers
    int root = labels[idx];
    while (root >= 0 && labels[root] >= 0 && labels[root] != root) {
        root = labels[root];
    }
    labels[idx] = root;  // path compression
}

// ──── Connected components: sequentialize labels ──────────────

__global__ void cc_compact(
    int* __restrict__ labels,           // [H, W]
    int* __restrict__ remap,            // [H*W]  temporary
    int* __restrict__ compact_labels,   // [H, W] output
    int H, int W, int* counter)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    int idx = v * W + u;
    if (labels[idx] < 0) {
        compact_labels[idx] = -1;
        return;
    }

    int root = labels[idx];
    if (root < 0) {
        compact_labels[idx] = -1;
        return;
    }

    // Assign new sequential ID via atomic
    if (remap[root] < 0) {
        int new_id = atomicAdd(counter, 1);
        remap[root] = new_id;
    }
    compact_labels[idx] = remap[root];
}

// ──── Public entry ────────────────────────────────────────────

torch::Tensor segmentImage(
    torch::Tensor image,
    int /*num_seeds unused*/,
    float /*compactness unused*/,
    int /*num_iters unused*/)
{
    TORCH_CHECK(image.dim() == 3 && image.size(0) == 3,
                "image must be [3, H, W]");
    TORCH_CHECK(image.is_cuda(), "image must be on CUDA");

    int C = image.size(0), H = image.size(1), W = image.size(2);

    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(image.device());
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(image.device());

    // Step 1: Sobel gradient magnitude
    torch::Tensor grad = torch::empty({H, W}, opt_f32);
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    sobel_edges<<<grid, block>>>(image.data_ptr<float>(), grad.data_ptr<float>(), H, W);

    // Step 2: Threshold → binary edge map
    torch::Tensor edge_map = torch::empty({H, W}, opt_i32);
    float thresh = 0.05f;  // gradient threshold (tune this)
    threshold_edges<<<grid, block>>>(grad.data_ptr<float>(), edge_map.data_ptr<int>(),
                                     H, W, thresh);

    // Step 3: Connected components on non-edge pixels
    torch::Tensor labels = torch::empty({H, W}, opt_i32);
    cc_init<<<grid, block>>>(edge_map.data_ptr<int>(), labels.data_ptr<int>(), H, W);

    // Union neighbours repeatedly (log(max_dim) passes ≈ 10-12 for 1200px)
    int num_passes = (int)ceilf(log2f((float)max(H, W))) + 2;
    for (int p = 0; p < num_passes; ++p) {
        cc_union<<<grid, block>>>(labels.data_ptr<int>(), H, W);
        cc_flatten<<<grid, block>>>(labels.data_ptr<int>(), H, W);
    }

    // Step 4: Compact labels into sequential 0..N-1
    torch::Tensor compact_labels = torch::full({H, W}, -1, opt_i32);
    torch::Tensor remap = torch::full({H * W}, -1, opt_i32);
    torch::Tensor counter = torch::zeros({1}, opt_i32);

    cc_compact<<<grid, block>>>(labels.data_ptr<int>(), remap.data_ptr<int>(),
                                compact_labels.data_ptr<int>(), H, W,
                                counter.data_ptr<int>());
    cudaDeviceSynchronize();

    return compact_labels.to(torch::kInt64);
}
