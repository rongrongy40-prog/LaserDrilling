# -*- coding: utf-8 -*-
"""
模块一：极速物理降维预处理 (Dataset 模块)。

功能：实现 GridDiffDrillingDataset，从 samples_info.json 与图片目录加载单孔数据，
      做层内融合 → 帧间绝对差 → 8×8 网格池化，输出 [Seq_Len, 64] 特征序列；提供 collate_fn 与 get_layer_list_from_path。
依赖：samples_info.json（含 sample_path, is_penetrated, penetration_layer 等）；每孔目录下按层命名的 jpg。
输出：单样本 dict 含 data [Seq_Len,64]、label、penetration_layer、layer_list、sample_path。
核心流程：
  1. 层内融合：同层多图求均值 → 每层一张代表图 I_n
  2. 帧间绝对残差：D_n = |I_n - I_{n-1}|
  3. 8×8 网格池化：对 D_n 的 ROI 划分 64 个 Patch，每 Patch 求均值 → 64 维/层
  4. 序列形状：单孔 [Seq_Len, 64]，collate 后 [Batch, 64, Seq_Len] 供 TCN 使用
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入 cv2，若失败则用 PIL（保证 Windows 兼容）
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image


def parse_frame_layer_from_filename(filename):
    """
    从图片文件名解析拍摄序号(帧)与层号。
    约定：去掉扩展名后按 '_' 分段，倒数第二段为拍摄序号，最后一段为层号。
    例：2025_04_29_07_56_50_324_1058_157.jpg → 帧=1058，层=157
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None, None
    try:
        frame_num = int(parts[-2])
        layer_num = int(parts[-1])
        return frame_num, layer_num
    except ValueError:
        return None, None


def parse_layer_from_filename(filename):
    """从文件名解析层号（与 parse_frame_layer_from_filename 一致，仅返回层号）。"""
    _, layer_num = parse_frame_layer_from_filename(filename)
    return layer_num


def get_layer_list_from_path(sample_path):
    """
    仅扫描样本文件夹内 jpg 文件名得到层号列表，不读图，用于快速建表。
    返回按层号排序的 list，若目录无效或为空则返回 []。
    """
    if not sample_path or not os.path.isdir(sample_path):
        return []
    pattern = os.path.join(sample_path, "*.jpg")
    paths = glob(pattern)
    layers = set()
    for p in paths:
        layer = parse_layer_from_filename(os.path.basename(p))
        if layer is not None:
            layers.add(layer)
    return sorted(layers)


def load_image_as_float(path, target_size=None):
    """
    加载单张图片为浮点数组，可选缩放。
    返回 shape: (H, W) 灰度 或 (H, W, 3) 彩色，值域 [0, 1] 或 [0, 255] 统一为 [0, 1]。
    """
    if HAS_CV2:
        img = cv2.imread(path)
        if img is None:
            return None
        # OpenCV 读入 BGR，转 RGB 以便与 PIL 逻辑一致（若后续用 PIL 可统一）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
    else:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    if target_size is not None:
        h, w = img.shape[:2]
        if (w, h) != target_size:
            if HAS_CV2:
                img = cv2.resize(img, target_size)
            else:
                img = np.array(
                    Image.fromarray((img * 255).astype(np.uint8)).resize(target_size),
                    dtype=np.float32
                ) / 255.0
    return img


def layer_mean_images(image_paths_by_layer, target_size=(224, 224), load_workers=1):
    """
    层内融合：对每一层的多张图求均值，得到每层的代表图 I_n。
    load_workers>1 时用多线程并行读图，加快单孔加载。

    image_paths_by_layer: dict, key=层号(int), value=list of 图片路径
    target_size: 统一缩放尺寸 (W, H)
    load_workers: 每层内并行读图线程数，1 为串行

    Returns:
        layer_order: list of int, 按层号排序的层序
        mean_images: list of np.ndarray, 每层一张 (H,W,3)，float32 [0,1]
    """
    layer_order = sorted(image_paths_by_layer.keys())
    mean_images = []

    for layer in layer_order:
        paths = image_paths_by_layer[layer]
        if not paths:
            continue
        if load_workers and load_workers > 1:
            imgs = []
            with ThreadPoolExecutor(max_workers=min(load_workers, len(paths))) as ex:
                futures = [ex.submit(load_image_as_float, p, target_size) for p in paths]
                for f in as_completed(futures):
                    arr = f.result()
                    if arr is not None:
                        imgs.append(arr)
        else:
            imgs = []
            for p in paths:
                arr = load_image_as_float(p, target_size)
                if arr is not None:
                    imgs.append(arr)
        if not imgs:
            continue
        mean_img = np.stack(imgs, axis=0).mean(axis=0).astype(np.float32)
        mean_images.append(mean_img)

    return layer_order, mean_images


def to_grayscale(rgb_image):
    """RGB 转灰度，保留 shape (H, W)。"""
    if rgb_image.ndim == 3:
        return (0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]).astype(np.float32)
    return rgb_image.astype(np.float32)


def temporal_diff_and_grid_pool(mean_images, roi_size=128, grid=(8, 8), use_grayscale=True,
                                 roi_center_yx=(0.5, 0.5), crop_mode="center"):
    """
    帧间绝对残差 + 8x8 网格池化。

    mean_images: list of (H, W, 3)，每层代表图
    roi_size: ROI / CenterCrop 正方形边长
    grid: (rows, cols) 网格数，默认 8x8=64
    use_grayscale: 是否先转灰度再做差分（推荐 True，与论文一致）
    roi_center_yx: (ratio_y, ratio_x)，ROI 中心在整图中的比例（crop_mode="roi" 时用）
    crop_mode: "roi" 表示用 roi_center_yx 比例裁剪，"center" 表示严格中心裁剪
    Returns:
        seq_64: np.ndarray, shape (Seq_Len, 64)，即 (层数, 64)
    """
    n_layers = len(mean_images)
    if n_layers == 0:
        return np.zeros((0, 64), dtype=np.float32)

    grid_r, grid_c = grid
    n_patches = grid_r * grid_c  # 64

    if mean_images[0].ndim == 3:
        h, w = mean_images[0].shape[0], mean_images[0].shape[1]
    else:
        h, w = mean_images[0].shape[0], mean_images[0].shape[1]

    crop = min(roi_size, h, w)
    
    if crop_mode == "center":
        # 严格中心裁剪（DINOv3 第一阶段风格）
        y0 = (h - crop) // 2
        x0 = (w - crop) // 2
    else:
        # 比例偏移裁剪（原来的 ROI 方式）
        ratio_y, ratio_x = roi_center_yx
        center_y = h * float(ratio_y)
        center_x = w * float(ratio_x)
        y0 = int(center_y - crop / 2)
        x0 = int(center_x - crop / 2)
        y0 = max(0, min(h - crop, y0))
        x0 = max(0, min(w - crop, x0))

    # 转为灰度并裁剪 ROI，得到 (n_layers, crop, crop)
    rois = []
    for img in mean_images:
        if use_grayscale:
            g = to_grayscale(img)
        else:
            g = to_grayscale(img) if img.ndim == 3 else img
        roi = g[y0 : y0 + crop, x0 : x0 + crop]
        rois.append(roi)
    rois = np.stack(rois, axis=0)  # (n_layers, crop, crop)

    # 帧间绝对残差：D[0]=0 或 D[0]=rois[0]，D[i] = |rois[i] - rois[i-1]|
    diff = np.zeros_like(rois)
    diff[0] = rois[0]  # 第一层无上一帧，用当前层作为“差分”基或置零；这里用当前层便于量级一致
    for i in range(1, n_layers):
        diff[i] = np.abs(rois[i].astype(np.float32) - rois[i - 1].astype(np.float32))

    # 将 ROI 划分为 grid_r x grid_c，每个 patch 内求均值
    ph, pw = crop // grid_r, crop // grid_c
    seq_64 = np.zeros((n_layers, n_patches), dtype=np.float32)

    for t in range(n_layers):
        d = diff[t]
        idx = 0
        for ri in range(grid_r):
            for ci in range(grid_c):
                sy, ey = ri * ph, (ri + 1) * ph
                sx, ex = ci * pw, (ci + 1) * pw
                patch = d[sy:ey, sx:ex]
                seq_64[t, idx] = float(np.mean(patch))
                idx += 1

    return seq_64


class GridDiffDrillingDataset(Dataset):
    """
    激光钻孔 Grid-Diff 数据集。

    每个样本输出：
      - data: Tensor [Seq_Len, 64]
      - label: 0=未穿透, 1=穿透
      - penetration_layer: 穿透层号（若未穿透则为 -1）
      - layer_list: 该样本对应的层号列表，长度 Seq_Len
      - sample_path: 样本文件夹路径（便于调试）

    训练时可用 collate_fn 将 data 转为 [B, 64, Seq_Len] 供 TCN 使用。
    """

    def __init__(
        self,
        samples_info_path,
        target_size=(224, 224),
        roi_size=128,
        grid=(8, 8),
        base_dir=None,
        load_workers=4,
        roi_center_yx=(0.5, 0.5),
        crop_mode="center",
    ):
        """
        samples_info_path: samples_info.json 路径
        target_size: 层代表图缩放尺寸 (W, H)，做 8x8 网格时用 128 即可加速
        roi_size: ROI / CenterCrop 正方形边长
        grid: (8, 8) 网格
        base_dir: 若样本路径为相对路径，则以此为根目录
        load_workers: 层内并行读图线程数，加快单孔加载
        roi_center_yx: (ratio_y, ratio_x)，crop_mode="roi" 时用
        crop_mode: "center"=严格中心裁剪（默认），"roi"=比例偏移裁剪
        """
        self.target_size = target_size
        self.roi_size = roi_size
        self.grid = grid
        self.base_dir = base_dir or ""
        self.load_workers = load_workers if load_workers else 1
        self.roi_center_yx = roi_center_yx
        self.crop_mode = crop_mode

        with open(samples_info_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 兼容顶层 {"Categories": [...]} 或直接 list
        if isinstance(raw, dict) and "Categories" in raw:
            raw = raw.get("Categories", [])
        if not isinstance(raw, list):
            raw = list(raw) if hasattr(raw, "__iter__") else []

        self.samples = []
        for item in raw:
            path = item.get("sample_path", "")
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            self.samples.append({
                "sample_path": path,
                "is_penetrated": int(item.get("is_penetrated", 0)),
                "penetration_layer": int(item.get("penetration_layer", -1)),
            })

    def __len__(self):
        return len(self.samples)

    def _build_layer_image_dict(self, sample_path):
        """根据样本路径，扫描所有图片并按层号分组。"""
        pattern = os.path.join(sample_path, "*.jpg")
        paths = glob(pattern)
        by_layer = defaultdict(list)
        for p in paths:
            name = os.path.basename(p)
            layer = parse_layer_from_filename(name)
            if layer is not None:
                by_layer[layer].append(p)
        return by_layer

    def __getitem__(self, index):
        """
        返回单孔数据：
          data: [Seq_Len, 64]
          label: 0 或 1
          penetration_layer: int
          layer_list: list of int, 长度 Seq_Len
          sample_path: str
        """
        sample = self.samples[index]
        path = sample["sample_path"]

        if not os.path.isdir(path):
            # 返回空序列，避免崩溃
            return {
                "data": torch.zeros(1, 64, dtype=torch.float32),
                "label": sample["is_penetrated"],
                "penetration_layer": sample["penetration_layer"],
                "layer_list": [0],
                "sample_path": path,
            }

        by_layer = self._build_layer_image_dict(path)
        layer_order, mean_images = layer_mean_images(
            by_layer, self.target_size, load_workers=self.load_workers
        )

        if len(mean_images) == 0:
            return {
                "data": torch.zeros(1, 64, dtype=torch.float32),
                "label": sample["is_penetrated"],
                "penetration_layer": sample["penetration_layer"],
                "layer_list": [0],
                "sample_path": path,
            }

        # 帧间差分 + 8x8 网格池化 → (Seq_Len, 64)
        seq_64 = temporal_diff_and_grid_pool(
            mean_images,
            roi_size=self.roi_size,
            grid=self.grid,
            use_grayscale=True,
            roi_center_yx=self.roi_center_yx,
            crop_mode=self.crop_mode,
        )

        return {
            "data": torch.from_numpy(seq_64).float(),
            "label": sample["is_penetrated"],
            "penetration_layer": sample["penetration_layer"],
            "layer_list": layer_order,
            "sample_path": path,
        }


class PrecomputedGridDiffDataset(Dataset):
    """
    从预计算特征目录加载整孔数据，接口与 GridDiffDrillingDataset 一致。
    用于在训练集上做推理时复用 cache_features_train，避免重复读图。
    命名规则与 precompute_features.py --by_name 及 WindowedDrillingDataset 一致。
    """

    _index_mismatch_warned = False

    def __init__(self, samples_info_path, precomputed_dir, base_dir=None):
        self.base_dir = base_dir or ""
        with open(samples_info_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "Categories" in raw:
            raw = raw.get("Categories", [])
        if not isinstance(raw, list):
            raw = list(raw) if hasattr(raw, "__iter__") else []
        self.samples = []
        for item in raw:
            path = item.get("sample_path", "")
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            self.samples.append({
                "sample_path": path,
                "is_penetrated": int(item.get("is_penetrated", 0)),
                "penetration_layer": int(item.get("penetration_layer", -1)),
            })
        self.precomputed_dir = os.path.abspath(precomputed_dir) if precomputed_dir else None
        used = set()

        def safe_basename(path):
            name = os.path.basename(path or "").strip().replace(os.sep, "_").replace("/", "_") or "unknown"
            return name

        self._precomputed_name_for_idx = []
        for i in range(len(self.samples)):
            path = self.samples[i].get("sample_path", "")
            base = safe_basename(path) or f"sample_{i}"
            name = base
            k = 2
            while name in used:
                name = f"{base}__{k}"
                k += 1
            used.add(name)
            self._precomputed_name_for_idx.append(name + ".pt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample["sample_path"]
        name_pt = os.path.join(self.precomputed_dir, self._precomputed_name_for_idx[index])
        idx_pt = os.path.join(self.precomputed_dir, f"{index}.pt")
        used_index_fallback = False
        if os.path.isfile(name_pt):
            pt_path = name_pt
        elif os.path.isfile(idx_pt):
            pt_path = idx_pt
            used_index_fallback = True
        else:
            pt_path = None
        if pt_path is None:
            return {
                "data": torch.zeros(1, 64, dtype=torch.float32),
                "label": sample["is_penetrated"],
                "penetration_layer": sample["penetration_layer"],
                "layer_list": [0],
                "sample_path": path,
            }
        raw = torch.load(pt_path, map_location="cpu")
        data = raw.get("data")
        layer_list = raw.get("layer_list", [])
        layer_list_norm = []
        for x in layer_list or []:
            if x is None or (isinstance(x, str) and not str(x).strip()):
                continue
            try:
                layer_list_norm.append(int(float(x)))
            except (TypeError, ValueError):
                pass
        layer_list = layer_list_norm
        saved_path = raw.get("sample_path")
        if saved_path and path and os.path.normpath(str(saved_path)) != os.path.normpath(str(path)):
            import warnings
            if used_index_fallback:
                if not PrecomputedGridDiffDataset._index_mismatch_warned:
                    PrecomputedGridDiffDataset._index_mismatch_warned = True
                    warnings.warn(
                        "预计算缓存与当前 samples_info 按索引对不上（例：全量列表预计算后只用训练 JSON）。"
                        "请对当前 JSON 重新运行: python precompute_features.py --samples_info ... --out_dir ... --by_name",
                        UserWarning,
                        stacklevel=2,
                    )
                return {
                    "data": torch.zeros(1, 64, dtype=torch.float32),
                    "label": sample["is_penetrated"],
                    "penetration_layer": sample["penetration_layer"],
                    "layer_list": [0],
                    "sample_path": path,
                }
            warnings.warn(
                f"预计算 sample_path 与当前不一致: pt内={saved_path!r} 当前={path!r}",
                UserWarning,
                stacklevel=2,
            )
        if data is not None and not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float() if hasattr(data, "__array__") else torch.zeros(1, 64, dtype=torch.float32)
        if data is None:
            data = torch.zeros(1, 64, dtype=torch.float32)
        return {
            "data": data,
            "label": sample["is_penetrated"],
            "penetration_layer": sample["penetration_layer"],
            "layer_list": layer_list,
            "sample_path": path,
        }


def collate_fn(batch):
    """
    将多个样本拼成 batch。
    每个样本 data 为 [Seq_Len_i, 64]，序列长度可能不同。
    处理方式：padding 到当前 batch 内最大长度，并转置为 TCN 输入 [B, 64, Seq_Len]。
    """
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    penetration_layers = [b["penetration_layer"] for b in batch]
    layer_lists = [b["layer_list"] for b in batch]
    paths = [b["sample_path"] for b in batch]

    data_list = [b["data"] for b in batch]
    max_len = max(d.size(0) for d in data_list)
    # Padding: 在时间维（第 0 维）右侧补 0
    padded = []
    for d in data_list:
        if d.size(0) < max_len:
            pad = torch.zeros(max_len - d.size(0), d.size(1), dtype=d.dtype)
            d = torch.cat([d, pad], dim=0)
        padded.append(d)

    # stack 后 (B, Seq_Len, 64)，转置为 (B, 64, Seq_Len) 供 TCN
    data = torch.stack(padded, dim=0)
    data = data.transpose(1, 2)
    # 此时 data: [Batch, 64, Seq_Len]

    return {
        "data": data,
        "label": labels,
        "penetration_layer": penetration_layers,
        "layer_list": layer_lists,
        "sample_path": paths,
    }
