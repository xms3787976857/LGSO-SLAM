import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import torchvision.transforms as transforms
import dataloader.data_utils as data_utils
import random


class SonarParquetLoader:
    def __init__(self, args):
        self.args = args
        # 【修改 1】删除这里写死的 self.split_name，移到 load_data 里动态判断

        # --- 双模自主切换逻辑 ---
        if getattr(args, 'use_local', False):
            import glob
            # 这里依然一次性读取所有文件，但在 load_data 里我们会进行筛选
            self.file_list = glob.glob(os.path.join(args.local_data_path, "**/*.parquet"), recursive=True)

            if not self.file_list:
                raise ValueError(f"❌ 在路径 {args.local_data_path} 下没找到任何 parquet 文件！")
            print(f"✅ [Init] 共发现 {len(self.file_list)} 个 Parquet 文件 (包含训练集和验证集)。")

        else:
            self.file_list = []

    @staticmethod
    def get_extrinsics(default_pose):
        '''change tranlation vector to be homogeneous -rcw as per Foundations of CV
        and the rotation matrix to have the axis direcitons as rows
        '''
        extrinsic = np.eye(4, 4)
        extrinsic[:3, :3] = default_pose[:3, :3].T
        extrinsic[:3, 3] = -default_pose[:3, :3].T @ default_pose[:3, 3]
        return extrinsic


    def _process_data(self, sample):

        sonar_intrinsic = {
            'width': 512,
            'height': 512,
            'r_min': 0.1,
            'r_max': 10,
            'elev': 20,
            'azi': 130
        }
        """
        深度对标原版 __getitem__ 逻辑
        
        
        """
        # 从 Parquet 列提取数据
        im1 = np.array(sample['im1']).astype(np.float32)
        im2 = np.array(sample['im2']).astype(np.float32)

        im1 = np.flip(im1).copy()
        im2 = np.flip(im2).copy()

        # print("Sample keys:", sample.keys())
        if 'pose' in sample:
            return sample  # 注意：此时 sample 应包含 im1, im2, pose 等字段

        # def to_extrinsic(raw_pose_np):
        #     pose = raw_pose_np.reshape(4, 4)
        #     extrinsic = np.eye(4, 4)
        #     # 旋转部分转置
        #     extrinsic[:3, :3] = pose[:3, :3].T
        #     # 平移部分重新计算
        #     extrinsic[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
        #     return torch.from_numpy(extrinsic).float()
        #
        # def to_extrinsic(raw_pose):
        #     # 确保 raw_pose 是 numpy 数组
        #     if isinstance(raw_pose, list):
        #         raw_pose = np.array(raw_pose, dtype=np.float32)
        #     elif not isinstance(raw_pose, np.ndarray):
        #         # 如果已经是 torch.Tensor 或其他类型，也转为 numpy
        #         raw_pose = np.array(raw_pose, dtype=np.float32)
        #
        #     # 重塑为 (4,4)
        #     pose = raw_pose.reshape(4, 4)
        #     return torch.tensor(pose, dtype=torch.float32)
        #
        # def to_extrinsic_1(raw_pose):
        #     if raw_pose is None:
        #         raise ValueError("raw_pose is None")  # 这样会直接暴露问题
        #     pose_np = np.asarray(raw_pose, dtype=np.float32).reshape(4, 4)
        #     return torch.tensor(pose_np, dtype=torch.float32)
        #
        # # # 提取并转换 T1 和 T2 为外参矩阵
        # # extrinsic1 = to_extrinsic(np.array(sample['pose1']))
        # # extrinsic2 = to_extrinsic(np.array(sample['pose2']))
        #
        # # 2. 获取相对位姿（兼容两种存储方式）
        # if 'relative_pose' in sample and sample['relative_pose'] is not None:
        #     relative_pose = to_extrinsic(sample['relative_pose'])
        #     # 转换为 numpy 数组并 reshape
        #     # pose_np = np.asarray(raw_pose, dtype=np.float32).reshape(4, 4)
        #     # relative_pose = torch.tensor(pose_np, dtype=torch.float32)
        # elif 'pose1' in sample and 'pose2' in sample:
        #     # 如果有绝对位姿，则计算相对位姿：relPose = pose2 @ inv(pose1)
        #     extrinsic1 = to_extrinsic(np.array(sample['pose1']))
        #     extrinsic2 = to_extrinsic(np.array(sample['pose2']))
        #     relative_pose = pose2 @ torch.linalg.inv(pose1)
        # else:
        #     raise ValueError("样本中既没有 relative_pose 也没有 pose1/pose2 字段！")
        # # 提取预计算好的相对位姿
        # # relative_pose = torch.from_numpy(np.array(sample['relative_pose'])).float().reshape(4,4)

        # 2. 获取相对位姿（兼容两种存储方式）
        if 'relative_pose' in sample and sample['relative_pose'] is not None:
            # 直接使用相对位姿
            raw_pose = sample['relative_pose']
        elif 'pose1' in sample and 'pose2' in sample:
            # 如果没有相对位姿但有绝对位姿，计算相对位姿：pose2 @ inv(pose1)
            pose1_np = np.asarray(sample['pose1'], dtype=np.float32).reshape(4, 4)
            pose2_np = np.asarray(sample['pose2'], dtype=np.float32).reshape(4, 4)
            pose1 = torch.tensor(pose1_np, dtype=torch.float32)
            pose2 = torch.tensor(pose2_np, dtype=torch.float32)
            relative_pose = pose2 @ torch.linalg.inv(pose1)
            raw_pose = relative_pose.numpy()  # 转为 numpy 以便统一处理
        else:
            raise ValueError("样本中既没有 relative_pose 也没有 pose1/pose2 字段！")

        # 确保 raw_pose 是 numpy 数组并 reshape
        pose_np = np.asarray(raw_pose, dtype=np.float32).reshape(4, 4)
        relative_pose = torch.tensor(pose_np, dtype=torch.float32)
        # --- 图像张量化与归一化 ---
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485), std=(0.229)),
        ])
        im1_tensor = transform(im1)
        im2_tensor = transform(im2)

        # --- 特征点生成 (仅老版 SONIC 需要外部 keypoints) ---
        w, h = im1.shape[:2]
        need_coord1 = getattr(self.args, 'need_external_kpts', False)
        if need_coord1:
            coord1 = data_utils.generate_query_kpts(
                im1, self.args.mode, num_pts=-1,
                akaze_superpoint_pts=self.args.akaze_superpoint_pts, h=h, w=w
            )
            coord1_tensor = torch.from_numpy(coord1).float()

        # --- 核心补充：补齐特征点剪枝逻辑 ---
        # if self.args.prune_kp:
        #     # 调用 data_utils 中的剪枝算法，过滤视野外的点
        #     # 这里使用固化的相对位姿进行投影计算
        #     ind_intersect = data_utils.prune_kpts_sonar(
        #         coord1,
        #         im2.shape[:2],
        #         sonar_intrinsic,
        #         extrinsic1, # T1
        #         extrinsic2  # T2
        #     )
        #     if np.sum(ind_intersect) == 0:
        #         # 保持原版打印提示
        #         print("Warning: All keypoints pruned in this sample.")
        #     coord1 = coord1[ind_intersect]
        #
        #     total_after_prune = coord1.shape[0]
        #     if total_after_prune > 0:
        #         # 再次随机采样回固定的 num_pts
        #         indices = np.random.choice(total_after_prune, self.args.num_pts, replace=True)
        #         coord1 = coord1[indices, :]
        #     else:
        #         print(f"⚠️ Warning: Sample dropped due to 0 valid keypoints.")
        #         return {'is_valid': False}  # <--- 直接返回 None，不要返回 zeros

        # coord1_tensor = torch.from_numpy(coord1).float()

        # scene = sample.get('scene')
        # if scene is None:
        #     scene = ''  # 或 'unknown'
        #
        # orig_path = sample.get('orig_path')
        # if orig_path is None:
        #     orig_path = ''  # 或保持空字符串

        # --- 构造完全等价的输出字典 ---
        out = {
                'im1': im1_tensor,
                'im2': im2_tensor,
                'im1_ori': torch.from_numpy(im1),
                'im2_ori': torch.from_numpy(im2),
                'pose': relative_pose,
                'is_valid': True
        }
        if need_coord1:
            out['coord1'] = coord1_tensor
        return out

    def my_collate(self, batch):
        # 检查 batch 中是否有 None 样本
        for idx, item in enumerate(batch):
            if item is None:
                print(f"Warning: sample {idx} is None")
                continue
            for key, value in item.items():
                if value is None:
                    print(f"Sample {idx} has None for key: {key}")
        # 过滤掉 None 样本（可选）
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None  # 或者返回一个空 batch，但 DataLoader 可能不支持
        return torch.utils.data.dataloader.default_collate(batch)

    # 2. 核心架构变更：创建 DataLoader 的逻辑搬到了这里
    def load_data(self, epoch=0):
        # 【修改 2】动态获取当前应当对应的 split 名称
        # 这样即使你在 train.py 里修改了 args.phase，这里也能实时响应
        current_split = "validation" if self.args.phase in ["val", "validation"] else "train"

        print(f"📂 [Loader] 正在准备 {current_split} 数据集 (Epoch {epoch})...")

        if getattr(self.args, 'use_local', False):
            # --- 本地模式精细化过滤 ---

            # 【修改 3】根据 split 过滤文件列表
            # 假设你的文件夹结构里包含 'train' 或 'val' 关键字
            # 例如: .../data/train/001.parquet 或 .../data/val/002.parquet
            if current_split == "train":
                # 只保留路径里包含 'train' 的文件
                current_epoch_files = [f for f in self.file_list if "train" in f.lower()]
            else:
                # 只保留路径里包含 'val' 或 'validation' 的文件
                current_epoch_files = [f for f in self.file_list if "val" in f.lower()]

            if not current_epoch_files:
                raise ValueError(
                    f"❌ 在模式 {current_split} 下没有过滤到任何文件！请检查路径是否包含 'train'/'val' 文件夹名。")

            # 【修改 4】只在训练时打乱文件顺序 (Shuffle Files)
            if current_split == "train":
                # 只有训练集需要每个 Epoch 随机换文件顺序
                random.seed(self.args.seed + epoch)
                random.shuffle(current_epoch_files)
                print(f"🔄 [Train] 已随机打乱 {len(current_epoch_files)} 个训练文件的顺序")
            else:
                # 验证集文件顺序必须固定！否则每次验证的样本如果不一致，指标波动会很大
                current_epoch_files.sort()
                print(f"🔒 [Val] 锁定 {len(current_epoch_files)} 个验证文件的顺序")

            # 初始化 Dataset
            dataset = load_dataset(
                "parquet",
                data_files=current_epoch_files,
                split="train",  # 注意：对于本地文件列表，这里通常填 "train" 即可，因为我们已经手动筛选了文件
                streaming=True,
                cache_dir=self.args.hf_cache_dir
            )
        else:
            # 远程模式 (Hugging Face 自带 split 管理)
            dataset = load_dataset(
                self.args.hf_repo_id,
                split=current_split,  # 这里直接传 "train" 或 "validation" 给 HF
                streaming=True,
                cache_dir=self.args.hf_cache_dir
            )



        # --- 步骤 3: Buffer 级打乱 (Micro-Shuffle) ---
        # 【修改 5】严格限制只在训练时开启 shuffle
        # 获取原始列名（如果是 streaming，features 可能为 None，但我们可以尝试）
        if dataset.features is not None:
            original_columns = list(dataset.features.keys())
        # else:
        #     # 如果无法获取，可以手动指定已知的所有可能字段
        #     original_columns = ['im1', 'im2', 'relative_pose', 'pose1', 'pose2', 'scene', 'orig_path']
        #
        # # 应用映射并删除原始列
        # dataset = dataset.map(self._process_data, remove_columns=original_columns)
        if current_split == "train":
            buffer_size = 500
            # 训练集：真随机打乱
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=self.args.seed + epoch)
        else:
            # 验证集：千万不要 shuffle！
            # 我们希望验证过程是确定性的 (Deterministic)，这样 Loss 的变化才真实可信
            pass




        dataset = dataset.map(self._process_data)

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            pin_memory=False,
            collate_fn=self.my_collate
        )
