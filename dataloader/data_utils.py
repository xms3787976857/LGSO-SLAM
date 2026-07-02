import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from random import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .demo_superpoint import SuperPointFrontend 
import torch

'''Minor modifications from CAPS'''

# --- 新增：全局单例模型容器 ---
_GLOBAL_SP = None
_GLOBAL_AKAZE = None
_GLOBAL_XFEAT = None

local_path = 'verlab/accelerated_features'
local_path_1 = '/mnt/accelerated_features'

def get_xfeat_instance():
    global _GLOBAL_XFEAT
    if _GLOBAL_XFEAT is None:
        print("🚀 正在通过 Torch Hub 加载 XFeat (CPU 模式)...")

        # --- 【核心修改】开始 ---
        # 临时“致盲” PyTorch，让它报告没有 GPU 可用
        # 这样 XFeat 内部就会强制使用 'cpu' 设备，避免在 Worker 里触碰 CUDA
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False

        try:
            # trust_repo=True 是为了防止新版 pytorch 报错
            _GLOBAL_XFEAT = torch.hub.load(
                local_path,
                'XFeat',
                pretrained=True,
                top_k=200,
                trust_repo=True
            )
        except Exception as e:
            print(f"❌ XFeat 加载失败: {e}")
            raise e
        finally:
            # 【重要】恢复 PyTorch 的视力，以免影响主进程的其他操作
            torch.cuda.is_available = original_is_available
        # --- 【核心修改】结束 ---

        # 确保模型在 CPU 上并开启评估模式
        _GLOBAL_XFEAT = _GLOBAL_XFEAT.cpu()
        _GLOBAL_XFEAT.eval()

    return _GLOBAL_XFEAT

def get_superpoint_instance():
    global _GLOBAL_SP
    if _GLOBAL_SP is None:
        # 资深建议：预处理放在 CPU 上跑，把 24G 显存全留给模型训练
        # 权重路径根据你的实际目录调整
        weights = '/mnt/weights/superpoint_v1.pth'
        _GLOBAL_SP = SuperPointFrontend(weights_path=weights, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False)
    return _GLOBAL_SP

def get_akaze_instance():
    global _GLOBAL_AKAZE
    if _GLOBAL_AKAZE is None:
        _GLOBAL_AKAZE = cv2.AKAZE_create()
    return _GLOBAL_AKAZE

def skew(x):
    '''
    converts it into a skew symmetric form, which can be used for crossproduct
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rotateImage(image, angle):
    '''rotate the image'''
    #TODO: check if this is required
    # rotaion might not be necessary
    h, w = image.shape[:2]
    angle_radius = np.abs(angle / 180. * np.pi)
    cos = np.cos(angle_radius)
    sin = np.sin(angle_radius)
    tan = np.tan(angle_radius)
    scale_h = (h / cos + (w - h * tan) * sin) / h
    scale_w = (h / sin + (w - h / tan) * cos) / w
    scale = max(scale_h, scale_w)
    image_center = tuple(np.array(image.shape[1::-1]) / 2.)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    rotation = np.eye(4)
    rotation[:2, :2] = rot_mat[:2, :2]
    return result, rotation

def perspective_transform(img, param=0.001):
    '''do a warp perspective'''
    #TODO: Again I don't think this is necessary
    h, w = img.shape[:2]
    random_state = np.random.RandomState(None)
    M = np.array([[1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand()]])

    dst = cv2.warpPerspective(img, M, (w, h))
    return dst, M


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=3.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def rescale_keypoints(keypoints, size):
    """ Rescale keypoints to fit original image size.
    Inputs
      keypoints: Nx2 numpy array of keypoints.
      size: (H, W) tuple specifying original image size.
    Returns
      rescaled_keypoints: Nx2 numpy array of rescaled keypoints.
    """
    H, W = size
    # 打印正在处理的尺寸
    # print(f"DEBUG - rescale_keypoints 接收到的尺寸: H={H}, W={W}, 类型: {type(H)}, {type(W)}")
    rescaled_keypoints = keypoints.copy()
    rescaled_keypoints[:, 0] = keypoints[:, 0] * W / 640
    rescaled_keypoints[:, 1] = keypoints[:, 1] * H / 480
    return rescaled_keypoints

def preprocess_image(image, size):
    """ Preprocess image before generating keypoints for both akaze (norm_image) and superpoint(grayim)"""
    norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    norm_image = unsharp_mask(norm_image, kernel_size=(3, 3), sigma=0.6, amount=2.0, threshold=0)
    norm_image = norm_image.astype(np.uint8)

    # set up image for superpoint
    interp= cv2.INTER_AREA
    grayim = cv2.resize(norm_image, (640,480), interpolation=interp)
    grayim = grayim.astype(np.float32) / 255.

    return norm_image, grayim


# def generate_query_kpts(img, mode='Xfeat', num_pts=None, h=None, w=None):
#
#     """
#         参数:
#         mode: 选择 'xfeat' (推荐) 或 'superpoint' (旧版)
#         """
#
#     # ----------------------------------------
#     # 第一步：运行 AKAZE (雷打不动，永远存在)
#     # ----------------------------------------
#     akaze = get_akaze_instance()
#     # 获取 AKAZE 需要的 uint8 锐化图
#     norm_image_akaze, grayim_superpoint = preprocess_image(img, (h, w))
#
#     kpa = akaze.detect(norm_image_akaze, None)
#     # 提取 AKAZE 坐标
#     coord_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kpa])
#
#     # ----------------------------------------
#     # 第二步：选择性运行 Deep Model (XFeat 或 SuperPoint)
#     # ----------------------------------------
#     coord_deep = np.zeros((0, 2))
#
#     if mode == 'Superpoint':
#         # >>> 分支 A: 传统的 SuperPoint <<<
#         sp = get_superpoint_instance()
#         # SuperPoint 必须用那个 resize 过的 grayim
#         kps, _, _ = sp.run(grayim_superpoint)
#         # 还原坐标尺度
#         rescaled_kpts = rescale_keypoints(kps.T, (h, w))
#         coord_deep = rescaled_kpts
#
#     elif mode == 'Xfeat':
#         # >>> 分支 B: 极速的 XFeat (推荐) <<<
#         xf = get_xfeat_instance()
#
#         # XFeat 专用预处理 (不做 resize，保持原图分辨率，精度更高)
#         # img 是 float32, 只要归一化到 0-1 即可
#         img_xfeat = img
#         if img_xfeat.max() > 1.0:
#             img_xfeat = img_xfeat / 255.0
#
#         x_tensor = torch.from_numpy(img_xfeat).float()
#         if x_tensor.ndim == 2:
#             x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
#
#         # 极速推理
#         with torch.no_grad():
#             out = xf.detectAndCompute(x_tensor, top_k=num_pts)[0]
#
#         # XFeat 直接返回正确尺度的坐标，不需要 rescale
#         coord_deep = out['keypoints'].cpu().numpy()
#
#     # ----------------------------------------
#     # 第三步：融合 (AKAZE + Deep Features)
#     # ----------------------------------------
#     # 如果任一列表为空，做容错处理
#     # --- 步骤 C: 严格容错阈值判定 ---
#     # 如果 AKAZE 或 深度点 任何一方太少，直接判定为无效图像，返回全零
#     len_kpa = coord_a.shape[0]
#     len_kpsp = coord_s.shape[0]
#
#     if len_kpa < akaze_superpoint_pts or len_kpsp < akaze_superpoint_pts:
#         return np.zeros((1, 2))
#
#     coord = np.vstack((coord_a, coord_deep))
#
#     # ----------------------------------------
#     # 第四步：采样与补齐 (修正版：随机重复采样)
#     # ----------------------------------------
#     # 算法工程师建议：
#     # 放弃 KNN 插值，改用 "Replace=True" 的随机采样。
#     # 这样既保证了 tensor 长度固定，又保证了所有点都是真实的物理特征点。
#
#     total = coord.shape[0]
#
#     # 极端情况兜底：如果一张图连 1 个特征点都提不出来 (比如全黑图)
#     if total == 0:
#         # 返回全零，或者随机噪声，防止报错。
#         # 这种样本通常 Loss 很大，会被网络慢慢学会忽略
#         return np.zeros((num_pts, 2), dtype=np.float32)
#
#     # 核心逻辑：无论是多还是少，直接用 choice 进行采样
#     # replace=True 意味着：
#     # 1. 如果 total > num_pts: 随机选 num_pts 个 (降采样)
#     # 2. 如果 total < num_pts: 随机重复选，直到凑够 num_pts 个 (过采样)
#     # 3. 如果 total == num_pts: 刚好选完
#
#     indices = np.random.choice(total, num_pts, replace=True)
#     coord = coord[indices, :]
#
#     return coord

def generate_query_kpts(img, mode='Superpoint', num_pts=2000, akaze_superpoint_pts=10, h=None, w=None, **kwargs):
    """
    重构逻辑说明：
    1. 严格阈值：若 AKAZE 或深度点数不足 akaze_superpoint_pts，返回 (1, 2) 信号。
    2. 维度对齐：解决 Superpoint (N,3) 与 AKAZE (N,2) 的冲突。
    3. KNN补全：当点数不足 num_pts 时，采用你认可的局部邻域均值插值。
    4. 性能优化：采用 data_utils.py 的单例模式。
    """
    # 获取单例
    akaze = get_akaze_instance()
    norm_image, grayim = preprocess_image(img, (h, w))

    # --- 1. 提取 AKAZE ---
    kpa = akaze.detect(norm_image, None)
    len_kpa = len(kpa)

    # --- 2. 提取深度点 (Superpoint/Xfeat) ---
    coord_s = np.zeros((0, 2))
    if mode == 'Superpoint':
        sp = get_superpoint_instance()
        kps_raw, _, _ = sp.run(grayim)
        rescaled = rescale_keypoints(kps_raw.T, (h, w))
        coord_s = rescaled[:, :2]  # 剔除 confidence 列
    elif mode == 'Xfeat':
        xf = get_xfeat_instance()
        img_xf = img.astype(np.float32)
        if img_xf.max() > 1.0: img_xf /= 255.0
        x_tensor = torch.from_numpy(img_xf).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = xf.detectAndCompute(x_tensor, top_k=150)[0]
        coord_s = out['keypoints'].cpu().numpy()[:, :2]

    len_kpsp = coord_s.shape[0]

    # --- 3. 严格容错判定 (核心：返回 (1, 2) 标志位) ---
    # 如果任何一方点数太少，说明图像质量极差，返回标志位让训练脚本跳过该样本
    if len_kpa < akaze_superpoint_pts or len_kpsp < akaze_superpoint_pts:
        print(f"\n[DEBUG] 特征点提取不足触发 (1, 2) 信号:")
        print(f"  - AKAZE 提取数量: {len_kpa} (阈值: {akaze_superpoint_pts})")
        print(f"  - 深度模型 ({mode}) 提取数量: {len_kpsp} (阈值: {akaze_superpoint_pts})")

        # 明确指出是谁的锅
        if len_kpa < akaze_superpoint_pts:
            print("  >>> 原因: AKAZE 提点太少！")
        if len_kpsp < akaze_superpoint_pts:
            print(f"  >>> 原因: {mode} 提点太少！")
        return np.zeros((1, 2))

        # --- 4. 融合 (XFeat + AKAZE)，不做裁剪或补齐 ---
    coord_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kpa])
    coord = np.vstack((coord_a, coord_s))
    total = coord.shape[0]

    if num_pts > 0:
        if total > num_pts:
            indices = np.random.choice(total, num_pts, replace=False)
            coord = coord[indices, :]
        elif total < num_pts:
            nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(coord)
            _, idx = nbrs.kneighbors(coord)
            needed = num_pts - total
            seed_indices = np.random.randint(low=0, high=total, size=needed)
            gen_kpts = []
            for i in seed_indices:
                mean_pt = np.mean(coord[idx[i]], axis=0)
                gen_kpts.append(mean_pt)
            coord = np.vstack((coord, np.array(gen_kpts)))
            np.random.shuffle(coord)

    return coord


def prune_kpts_sonar(coord1, im2_size, sonar_info, T1, T2):
    """
    极致严格版：只要弧线上有一个采样点出界，就彻底丢弃该关键点。
    """
    H, W = im2_size
    N = coord1.shape[0]
    if N == 0: return np.array([], dtype=bool)

    # 1. 计算相对位姿
    t1_np = T1.cpu().numpy() if hasattr(T1, "cpu") else T1
    t2_np = T2.cpu().numpy() if hasattr(T2, "cpu") else T2
    rel_pose = t2_np @ np.linalg.inv(t1_np)

    # 2. 解析参数 (azi: 130°, elev: 20°)
    r_min, r_max = sonar_info['r_min'], sonar_info['r_max']
    fov_azi = np.deg2rad(sonar_info['azi'])
    fov_elev = np.deg2rad(sonar_info['elev'])

    u, v = coord1[:, 0], coord1[:, 1]

    # 3. 像素 -> 极坐标 (Frame 1)
    bearing = ((W - u) * fov_azi / W) - (fov_azi / 2.0)
    r_val = ((r_max - r_min) / H) * (H - v) + r_min

    # 4. 垂直采样 (保持 10 个点足以覆盖弧线两端)
    num_samples = 100
    phi_samples = np.linspace(-fov_elev / 2, fov_elev / 2, num_samples)

    bearing = bearing[:, np.newaxis]
    r_val = r_val[:, np.newaxis]
    phi = phi_samples[np.newaxis, :]

    # 5. 极坐标 -> 直角坐标 (Frame 1)
    X1 = r_val * np.cos(bearing) * np.cos(phi)
    Y1 = r_val * np.sin(bearing) * np.cos(phi)
    Z1 = -r_val * np.sin(phi)
    ones = np.ones_like(X1)
    P1 = np.stack([X1, Y1, Z1, ones], axis=0).reshape(4, -1)

    # 6. 变换到 Frame 2
    P2 = rel_pose @ P1
    X2, Y2, Z2 = P2[0, :], P2[1, :], P2[2, :]

    # 7. 直角坐标 -> 极坐标 -> 像素 (Frame 2)
    r2 = np.sqrt(X2 ** 2 + Y2 ** 2 + Z2 ** 2 + 1e-6)
    b2 = np.arctan2(Y2, X2)
    u2 = W - (b2 + fov_azi / 2.0) * W / fov_azi
    v2 = H - (r2 - r_min) * H / (r_max - r_min)

    # --- 【核心修改点：极致严格判定】 ---
    u2 = u2.reshape(N, num_samples)
    v2 = v2.reshape(N, num_samples)

    # 判定每个采样点是否都在图像内
    in_fov_mask = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)

    # 使用 np.all：只有这 10 个点全部为 True，才保留该行
    fov_mask = np.all(in_fov_mask, axis=1)

    return fov_mask

# def prune_kpts_sonar(coord1, im2_size, sonar_info, T1, T2):
#     """
#     专门针对声纳的几何剪枝函数 (資深工程師修正版)
#     剔除逻辑：
#     1. 移除 Range Filter：完全信任输入点的深度信息。
#     2. 10点采样：将垂直俯仰角 (Elevation) 细分为 10 个采样点，提高弧线覆盖率。
#     3. 鲁棒性判定：要求弧线上必须有至少 50% (5/10) 的点在图2视野内，才判定为“可见”。
#     """
#     H, W = im2_size
#     N = coord1.shape[0]
#     if N == 0: return np.array([], dtype=bool)
#
#     # 1. 计算相对位姿 T_1->2 = T2 @ inv(T1)
#     try:
#         # 兼容处理：如果传入的是 Tensor，先转为 Numpy
#         t1_np = T1.cpu().numpy() if hasattr(T1, "cpu") else T1
#         t2_np = T2.cpu().numpy() if hasattr(T2, "cpu") else T2
#         rel_pose = t2_np @ np.linalg.inv(t1_np)
#     except Exception as e:
#         print(f"Pose Inversion Error: {e}")
#         return np.ones(N, dtype=bool)
#
#     # 2. 解析声纳参数
#     r_min, r_max = sonar_info['r_min'], sonar_info['r_max']
#     fov_azi = np.deg2rad(sonar_info['azi'])  # 通常 130度
#     fov_elev = np.deg2rad(sonar_info['elev'])  # 通常 20度
#
#     u, v = coord1[:, 0], coord1[:, 1]
#
#     # 3. Pixel -> Polar (Frame 1)
#     bearing = ((W - u) * fov_azi / W) - (fov_azi / 2.0)
#     r_val = ((r_max - r_min) / H) * (H - v) + r_min
#
#     # 4. 垂直方向采样 (Sampling Elevation Phi)
#     # 【修改】：改为 10 个采样点
#     num_samples = 100
#     phi_samples = np.linspace(-fov_elev / 2, fov_elev / 2, num_samples)
#
#     # 准备广播计算 [N, 10]
#     bearing = bearing[:, np.newaxis]
#     r_val = r_val[:, np.newaxis]
#     phi = phi_samples[np.newaxis, :]
#
#     # 5. Polar -> Cartesian (Frame 1) - 前-左-下坐标系
#     X1 = r_val * np.cos(bearing) * np.cos(phi)
#     Y1 = r_val * np.sin(bearing) * np.cos(phi)
#     Z1 = -r_val * np.sin(phi)
#     ones = np.ones_like(X1)
#
#     # 组合点云 [4, N*10]
#     P1 = np.stack([X1, Y1, Z1, ones], axis=0).reshape(4, -1)
#
#     # 6. Transform to Frame 2: P2 = T_rel * P1
#     P2 = rel_pose @ P1
#     X2, Y2, Z2 = P2[0, :], P2[1, :], P2[2, :]
#
#     # 7. Cartesian -> Polar (Frame 2)
#     r2 = np.sqrt(X2 ** 2 + Y2 ** 2 + Z2 ** 2 + 1e-6)
#     b2 = np.arctan2(Y2, X2)
#
#     # 8. Polar -> Pixel (Frame 2)
#     u2 = W - (b2 + fov_azi / 2.0) * W / fov_azi
#     v2 = H - (r2 - r_min) * H / (r_max - r_min)
#
#     # Reshape 回 [N, 10]
#     u2 = u2.reshape(N, num_samples)
#     v2 = v2.reshape(N, num_samples)
#
#     # 9. 【修改判定标准】：统计在图像内的点数
#     in_fov_mask = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
#
#     # 只要满足条件点数 >= 5 即可保留该关键点
#     fov_mask = np.sum(in_fov_mask, axis=1) >= 5
#
#     return fov_mask

def make_matching_figure(
    img0, img1, mkpts0, mkpts1, kpts0=None, kpts1=None, text=[], dpi=96, path=None
):
    # draw image pair
    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    brightness = 2.1 
    # Adjusts the contrast by scaling the pixel values by 2.3
    contrast = 1.3
    img0 = cv2.addWeighted(img0, contrast, np.zeros(img0.shape, img0.dtype), 0, brightness)
    img1 = cv2.addWeighted(img1, contrast, np.zeros(img1.shape, img1.dtype), 0, brightness)

    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=0.8)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="yellow", edgecolor="g", s=1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="yellow", edgecolor="g", s=1)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c="lime",
                linewidth=0.5,
                alpha=0.5,
            )
            for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c="lime", s=7)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c="lime", s=7)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig
