# -*- coding: utf-8 -*-
"""
模块三：抗不平衡训练逻辑 (Train 模块)。

功能：基于 WindowedDrillingDataset 做窗口级训练；支持按孔划分 train/val、平衡采样或按孔组 batch；
      Focal Loss / 负样本子采样 / 辅助定位损失；每轮窗口级验证，可选按孔验证与最佳权重保存；支持 AMP。
依赖：dataset.py, tcn_model.py, inference.run_inference；samples_info（或 samples_info_train.json）；可选 precomputed_dir。
输出：训练日志；最佳或最终模型权重保存到 --save（默认 grid_diff_tcn.pt）。
主要参数：--samples_info, --precomputed_dir, --save, --epochs, --batch_size, --simple（简单模式）, --no_amp（关 AMP）, --val_ratio。
示例：python train.py --samples_info ../data_drilling/samples_info_train.json --precomputed_dir ./cache_features_train --simple --device cuda
"""

import os
import sys
import json
import argparse
import random
import queue
import threading
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# 将当前目录加入路径，便于同目录下 import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import GridDiffDrillingDataset, collate_fn, get_layer_list_from_path
from tcn_model import GridDiffTCN
from transformer_tcn_model import build_tcn_or_transformer, GridDiffTCNWithTransformer
from inference import run_inference_topkmedian


# ---------------------------------------------------------------------------
# 窗口采样：将整孔 [Seq_Len, 64] 截成固定长度窗口，并给出该窗口的标签
# ---------------------------------------------------------------------------

class WindowedDrillingDataset(Dataset):
    """
    在 GridDiffDrillingDataset 的基础上做窗口采样：
    - 正样本（穿透）：围绕真实穿透层截取 window_len 层，且穿透点落在窗口中后部（例如后 1/3）
    - 负样本（未穿透）：随机截取 window_len 层
    每个窗口一个样本，可能从一个孔产生多个窗口（负样本时随机多次截取）。
    """

    def __init__(
        self,
        samples_info_path,
        target_size=(224, 224),
        roi_size=128,
        grid=(8, 8),
        base_dir=None,
        window_len=60,
        penetration_in_window_tail_ratio=0.4,
        neg_random_windows_per_hole=1,
        skip_first_layers=30,
        max_samples=None,
        load_workers=6,
        roi_center_yx=(0.5, 0.5),
        crop_mode="center",
        penetration_radius=0,
        precomputed_dir=None,
    ):
        """
        precomputed_dir: 若指定，从该目录加载 {base_idx}.pt（含 data、layer_list），跳过读图与计算，大幅提速
        window_len: 窗口长度（层数）
        penetration_in_window_tail_ratio: 正样本中，穿透层需落在窗口内后部比例，如 0.4 表示穿透层在窗口 [0.6*L, L] 内
        neg_random_windows_per_hole: 每个负样本孔随机截取几个窗口
        skip_first_layers: 与推理一致，前若干层不参与（可选，这里主要用于确定有效起点）
        load_workers: 层内并行读图线程数，用于加速单孔加载
        roi_center_yx: (ratio_y, ratio_x)，ROI 中心在画面中的比例，孔不在正中时可改，如 (0.4, 0.5) 表示偏上
        crop_mode: "center"=严格中心裁剪（默认），"roi"=比例偏移裁剪
        penetration_radius: 穿透层前后多少层也标为 1（软标签），0=单点，2=穿透层±2 共 5 个正类，缓解不平衡与单点过严
        """
        self.base_dataset = GridDiffDrillingDataset(
            samples_info_path,
            target_size=target_size,
            roi_size=roi_size,
            grid=grid,
            base_dir=base_dir,
            load_workers=load_workers,
            roi_center_yx=roi_center_yx,
            crop_mode=crop_mode,
        )
        self.window_len = window_len
        self.penetration_in_window_tail_ratio = penetration_in_window_tail_ratio
        self.neg_random_windows_per_hole = neg_random_windows_per_hole
        self.skip_first_layers = skip_first_layers
        self.penetration_radius = max(0, int(penetration_radius))
        self.precomputed_dir = (os.path.abspath(precomputed_dir) if precomputed_dir else None)
        n_base = len(self.base_dataset)
        if max_samples is not None:
            n_base = min(n_base, max_samples)

        # 建表时不读图：仅用文件名扫描得到层列表，避免在 __init__ 里加载所有孔的所有图片
        self.samples = []
        for idx in tqdm(range(n_base), desc="建表(扫描层数)", leave=False):
            sample = self.base_dataset.samples[idx]
            path = sample["sample_path"]
            is_penetrated = sample["is_penetrated"]
            penetration_layer = sample["penetration_layer"]

            if not os.path.isdir(path):
                continue

            layer_list = get_layer_list_from_path(path)
            seq_len = len(layer_list)
            if seq_len < window_len:
                continue

            if is_penetrated and penetration_layer >= 0:
                try:
                    pen_idx = layer_list.index(penetration_layer)
                except ValueError:
                    pen_idx = min(seq_len - 1, max(0, penetration_layer))
                tail_start_in_window = int(window_len * (1 - penetration_in_window_tail_ratio))
                start_min_idx = max(0, pen_idx - window_len + 1)
                start_max_idx = min(seq_len - window_len, pen_idx - tail_start_in_window)
                if start_max_idx < start_min_idx:
                    start_max_idx = start_min_idx
                for start_idx in range(start_min_idx, start_max_idx + 1):
                    if start_idx + window_len <= seq_len:
                        self.samples.append({
                            "base_idx": idx,
                            "start": start_idx,
                            "label": 1,
                            "penetration_layer": penetration_layer,
                        })
                if not any(s["base_idx"] == idx for s in self.samples):
                    start_idx = max(0, min(pen_idx - window_len // 2, seq_len - window_len))
                    if start_idx + window_len <= seq_len:
                        self.samples.append({
                            "base_idx": idx,
                            "start": start_idx,
                            "label": 1,
                            "penetration_layer": penetration_layer,
                        })
            else:
                start_min = skip_first_layers
                if seq_len - start_min >= window_len:
                    for _ in range(neg_random_windows_per_hole):
                        start_max = seq_len - window_len + 1
                        start_idx = np.random.randint(start_min, start_max)
                        self.samples.append({
                            "base_idx": idx,
                            "start": start_idx,
                            "label": 0,
                            "penetration_layer": -1,
                        })

        # 正/负样本索引，供平衡 batch 采样使用
        self._pos_indices = [i for i, s in enumerate(self.samples) if s["label"] == 1]
        self._neg_indices = [i for i, s in enumerate(self.samples) if s["label"] == 0]

        # 按孔缓存：同一孔只算一次 [Seq_Len, 64]，后续窗口只做切片，避免重复读图与计算
        self._cache = {}
        # 预计算按样本名加载时的文件名（与 precompute_features.py --by_name 一致）
        if self.precomputed_dir:
            def safe_basename(path):
                name = os.path.basename(path or "").strip().replace(os.sep, "_").replace("/", "_") or "unknown"
                return name
            used = set()
            self._precomputed_name_for_idx = []
            for i in range(len(self.base_dataset.samples)):
                path = self.base_dataset.samples[i].get("sample_path", "")
                base = safe_basename(path) or f"sample_{i}"
                name = base
                k = 2
                while name in used:
                    name = f"{base}__{k}"
                    k += 1
                used.add(name)
                self._precomputed_name_for_idx.append(name + ".pt")
        else:
            self._precomputed_name_for_idx = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ent = self.samples[index]
        base_idx = ent["base_idx"]
        start = ent["start"]
        if base_idx not in self._cache:
            if self.precomputed_dir:
                # 先按样本名找，再按 base_idx.pt（兼容旧缓存）
                pt_path = os.path.join(self.precomputed_dir, self._precomputed_name_for_idx[base_idx])
                if not os.path.isfile(pt_path):
                    pt_path = os.path.join(self.precomputed_dir, f"{base_idx}.pt")
                if os.path.isfile(pt_path):
                    raw = torch.load(pt_path, map_location="cpu")
                    raw["sample_path"] = self.base_dataset.samples[base_idx].get("sample_path", "")
                    self._cache[base_idx] = raw
                else:
                    self._cache[base_idx] = self.base_dataset[base_idx]
            else:
                self._cache[base_idx] = self.base_dataset[base_idx]
        raw = self._cache[base_idx]
        data = raw["data"]
        layer_list = raw["layer_list"]
        window_data = data[start : start + self.window_len]
        layer_window = layer_list[start : start + self.window_len]

        # === 逐层标签：正样本穿透层（及±radius）为1，负样本全0 ===
        seq_label = torch.zeros(self.window_len, dtype=torch.long)
        if ent["label"] == 1 and ent["penetration_layer"] >= 0:
            try:
                pen_pos = layer_window.index(ent["penetration_layer"])
                r = self.penetration_radius
                for j in range(max(0, pen_pos - r), min(self.window_len, pen_pos + r + 1)):
                    seq_label[j] = 1  # 单点(r=0)或软窗口(r>0)
            except ValueError:
                pass

        return {
            "data": window_data,          # [T, 64]
            "label": ent["label"],        # 窗口级标签（0/1）
            "seq_label": seq_label,       # [T] 逐层标签（单点1）
            "penetration_layer": ent["penetration_layer"],
            "layer_list": layer_window,
            "sample_path": raw.get("sample_path", ""),
        }


def window_collate(batch):
    """窗口数据已等长，直接 stack 并转置为 (B, 64, T)；seq_label 拼成 (B, T)。"""
    data = torch.stack([b["data"] for b in batch], dim=0)
    data = data.transpose(1, 2)  # (B, T, 64) -> (B, 64, T)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    seq_label = torch.stack([b["seq_label"] for b in batch], dim=0)  # (B, T)
    return {
        "data": data,
        "label": labels,
        "seq_label": seq_label,
    }


# ---------------------------------------------------------------------------
# 平衡 Batch 采样：每 batch 保证至少 pos_per_batch 个正样本
# ---------------------------------------------------------------------------

class BalancedBatchSampler(Sampler):
    """
    每个 batch 由 pos_per_batch 个正样本 + (batch_size - pos_per_batch) 个负样本组成；
    每 epoch 打乱正/负列表后依次取 batch，不足时用另一类循环补足。
    """

    def __init__(self, pos_indices, neg_indices, batch_size, pos_per_batch):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.batch_size = batch_size
        self.pos_per_batch = min(max(0, pos_per_batch), batch_size)
        self.neg_per_batch = self.batch_size - self.pos_per_batch
        n_pos, n_neg = len(self.pos_indices), len(self.neg_indices)
        if n_pos == 0 and n_neg == 0:
            self.num_batches = 0
        elif self.pos_per_batch == 0 or n_pos == 0:
            self.num_batches = max(1, (n_pos + n_neg) // self.batch_size)
        else:
            from_pos = n_pos // self.pos_per_batch
            from_neg = n_neg // max(1, self.neg_per_batch)
            self.num_batches = max(1, min(from_pos, from_neg))

    def __iter__(self):
        pos = list(self.pos_indices)
        neg = list(self.neg_indices)
        random.shuffle(pos)
        random.shuffle(neg)
        n_pos, n_neg = len(pos), len(neg)
        pos_per, neg_per = self.pos_per_batch, self.neg_per_batch
        pos_idx, neg_idx = 0, 0
        for _ in range(self.num_batches):
            batch = []
            for _ in range(pos_per):
                batch.append(pos[pos_idx % n_pos])
                pos_idx += 1
            for _ in range(neg_per):
                batch.append(neg[neg_idx % n_neg])
                neg_idx += 1
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ---------------------------------------------------------------------------
# 按孔组 batch：同一 batch 内样本均来自同一孔，最后一格不足时用该孔内样本重复填充
# ---------------------------------------------------------------------------

class SameHoleBatchSampler(Sampler):
    """
    每个 batch 内的窗口样本均来自同一孔（base_idx）。
    - 按图片/层顺序（start）裁剪成 batch_size 的 batch，最后一格不足用该孔内样本重复填充。
    - 若 pos_inject_ratio>0：在所有「无穿透」的 batch 中随机选 neg_batch_inject_ratio 比例，用该孔穿透样本随机替换 batch 中 pos_inject_ratio 比例的样本，保证这些 batch 也有穿透图。
    """

    def __init__(
        self,
        dataset,
        train_indices,
        batch_size,
        shuffle=True,
        seed=42,
        pos_inject_ratio=0.0,
        neg_batch_inject_ratio=0.0,
    ):
        """
        pos_inject_ratio: 对“被选中的全负 batch”中，用穿透样本替换的比例（占 batch_size），如 0.2 表示 20%。
        neg_batch_inject_ratio: 全负 batch 中被注入穿透样本的比例，如 0.5 表示 50%。
        """
        self.dataset = dataset
        self.train_indices = train_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pos_inject_ratio = max(0.0, min(1.0, pos_inject_ratio))
        self.neg_batch_inject_ratio = max(0.0, min(1.0, neg_batch_inject_ratio))

        hole_to_windows = defaultdict(list)
        for j in range(len(train_indices)):
            idx = train_indices[j]
            base_idx = dataset.samples[idx]["base_idx"]
            hole_to_windows[base_idx].append(j)
        self.hole_to_windows = dict(hole_to_windows)
        # 每孔按 start 排序后的下标、以及该孔穿透样本下标列表（用于注入）
        self._hole_sorted = {}
        self._hole_pos = {}
        for base_idx, j_list in self.hole_to_windows.items():
            sorted_j = sorted(
                j_list,
                key=lambda j: dataset.samples[train_indices[j]].get("start", 0),
            )
            self._hole_sorted[base_idx] = sorted_j
            self._hole_pos[base_idx] = [
                j for j in sorted_j
                if dataset.samples[train_indices[j]].get("label", 0) == 1
            ]
        self._num_batches = 0
        for indices in self._hole_sorted.values():
            n = len(indices)
            self._num_batches += (n + batch_size - 1) // batch_size
        self._epoch = 0

    def set_epoch(self, epoch):
        """每 epoch 使用不同随机序，便于打乱孔顺序。"""
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        keys = list(self._hole_sorted.keys())
        if self.shuffle:
            rng.shuffle(keys)
        for base_idx in keys:
            # 按层序（start）裁剪 batch，不打乱孔内顺序
            indices = list(self._hole_sorted[base_idx])
            pos_j = self._hole_pos.get(base_idx, [])
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                while len(batch) < self.batch_size:
                    batch.extend(indices)
                batch = batch[: self.batch_size]
                # 若开启注入：当前 batch 无穿透且该孔有穿透样本时，以 neg_batch_inject_ratio 概率注入
                if (
                    self.pos_inject_ratio > 0
                    and self.neg_batch_inject_ratio > 0
                    and len(pos_j) > 0
                ):
                    has_pos = any(
                        self.dataset.samples[self.train_indices[j]].get("label", 0) == 1
                        for j in batch
                    )
                    if not has_pos and rng.random() < self.neg_batch_inject_ratio:
                        n_replace = max(1, int(self.pos_inject_ratio * self.batch_size))
                        replace_positions = rng.sample(range(len(batch)), min(n_replace, len(batch)))
                        for pos in replace_positions:
                            batch[pos] = rng.choice(pos_j)
                yield batch

    def __len__(self):
        return self._num_batches


# ---------------------------------------------------------------------------
# 不平衡友好：Focal Loss + 可选负样本子采样
# ---------------------------------------------------------------------------

def focal_cross_entropy(logits_bt, seq_y_bt, gamma=2.0, alpha_pos=0.75, device=None):
    """Focal loss：压低易分负样本的梯度，缓解 1:59 不平衡。alpha_pos 为正类权重。"""
    probs = F.softmax(logits_bt, dim=1)
    pt = probs.gather(1, seq_y_bt.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    alpha_t = torch.where(seq_y_bt == 1, torch.tensor(alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
                          torch.tensor(1.0 - alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype))
    focal_w = (1 - pt).pow(gamma)
    return (focal_w * (-pt.log()) * alpha_t)


def train_one_epoch(model, loader, optimizer, device, epoch=0, total_epochs=1,
                    use_focal=True, focal_gamma=2.0, focal_alpha=0.75,
                    weight_pos=10.0, subsample_neg=0, loc_loss_weight=0.0,
                    kl_weight=0.0, scaler=None, use_amp=False):
    """
    逐层训练。use_focal=True 时用 Focal 压低易分负样本；否则用加权 CE。
    use_amp 且 scaler 非空时使用混合精度。
    """
    model.train()
    total_loss = 0.0
    correct_layers, total_layers = 0, 0
    pen_correct, pen_total = 0, 0
    loc_mae_sum, loc_mae_n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=True)
    for batch in pbar:
        x = batch["data"].to(device, non_blocking=True)
        seq_y = batch["seq_label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                if isinstance(out, tuple):
                    logits, extra = out
                else:
                    logits, extra = out, {}
                B, _, T = logits.shape
                logits_bt = logits.permute(0, 2, 1).reshape(-1, 2)
                seq_y_bt = seq_y.reshape(-1)
                if use_focal:
                    step_loss = focal_cross_entropy(logits_bt, seq_y_bt, gamma=focal_gamma, alpha_pos=focal_alpha, device=device)
                else:
                    w = torch.where(seq_y_bt == 1,
                                    torch.tensor(weight_pos, device=device, dtype=logits.dtype),
                                    torch.tensor(1.0, device=device, dtype=logits.dtype))
                    step_loss = F.cross_entropy(logits_bt, seq_y_bt, reduction="none") * w
                if subsample_neg > 0:
                    pos_mask = seq_y_bt == 1
                    neg_mask = torch.zeros(B * T, dtype=torch.bool, device=device)
                    for b in range(B):
                        neg_local = (seq_y[b] == 0).nonzero(as_tuple=True)[0]
                        if neg_local.numel() > subsample_neg:
                            perm = torch.randperm(neg_local.numel(), device=device)[:subsample_neg]
                            neg_global = b * T + neg_local[perm]
                        else:
                            neg_global = b * T + neg_local
                        neg_mask[neg_global] = True
                    step_mask = pos_mask | neg_mask
                    loss = step_loss[step_mask].mean()
                else:
                    loss = step_loss.mean()
                if loc_loss_weight > 0:
                    has_pen = (seq_y.sum(dim=1) > 0)
                    if has_pen.any():
                        arange_t = torch.arange(T, device=logits.device, dtype=logits.dtype)
                        prob_1 = F.softmax(logits[:, 1, :], dim=1)
                        pred_soft = (prob_1 * arange_t.unsqueeze(0)).sum(dim=1)
                        true_layer = seq_y.argmax(dim=1).float()
                        loss_loc = F.smooth_l1_loss(pred_soft[has_pen], true_layer[has_pen])
                        loss = loss + loc_loss_weight * loss_loc
                if kl_weight > 0.0 and isinstance(extra, dict) and "kl_loss" in extra:
                    loss = loss + kl_weight * extra["kl_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            if isinstance(out, tuple):
                logits, extra = out
            else:
                logits, extra = out, {}
            B, _, T = logits.shape
            logits_bt = logits.permute(0, 2, 1).reshape(-1, 2)
            seq_y_bt = seq_y.reshape(-1)
            if use_focal:
                step_loss = focal_cross_entropy(logits_bt, seq_y_bt, gamma=focal_gamma, alpha_pos=focal_alpha, device=device)
            else:
                w = torch.where(seq_y_bt == 1,
                                torch.tensor(weight_pos, device=device, dtype=logits.dtype),
                                torch.tensor(1.0, device=device, dtype=logits.dtype))
                step_loss = F.cross_entropy(logits_bt, seq_y_bt, reduction="none") * w
            if subsample_neg > 0:
                pos_mask = seq_y_bt == 1
                neg_mask = torch.zeros(B * T, dtype=torch.bool, device=device)
                for b in range(B):
                    neg_local = (seq_y[b] == 0).nonzero(as_tuple=True)[0]
                    if neg_local.numel() > subsample_neg:
                        perm = torch.randperm(neg_local.numel(), device=device)[:subsample_neg]
                        neg_global = b * T + neg_local[perm]
                    else:
                        neg_global = b * T + neg_local
                    neg_mask[neg_global] = True
                step_mask = pos_mask | neg_mask
                loss = step_loss[step_mask].mean()
            else:
                loss = step_loss.mean()
            if loc_loss_weight > 0:
                has_pen = (seq_y.sum(dim=1) > 0)
                if has_pen.any():
                    arange_t = torch.arange(T, device=logits.device, dtype=logits.dtype)
                    prob_1 = F.softmax(logits[:, 1, :], dim=1)
                    pred_soft = (prob_1 * arange_t.unsqueeze(0)).sum(dim=1)
                    true_layer = seq_y.argmax(dim=1).float()
                    loss_loc = F.smooth_l1_loss(pred_soft[has_pen], true_layer[has_pen])
                    loss = loss + loc_loss_weight * loss_loc
            if kl_weight > 0.0 and isinstance(extra, dict) and "kl_loss" in extra:
                loss = loss + kl_weight * extra["kl_loss"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct_layers += (pred == seq_y).sum().item()
        total_layers += seq_y.numel()

        # 定位指标：有穿透的样本（至少一个 1）
        has_pen = (seq_y.sum(dim=1) > 0)
        if has_pen.any():
            pred_layer = logits[:, 1, :].argmax(dim=1)
            true_layer = seq_y.argmax(dim=1)
            loc_mae_sum += (pred_layer[has_pen].float() - true_layer[has_pen].float()).abs().sum().item()
            loc_mae_n += has_pen.sum().item()
            within_2 = (pred_layer[has_pen] - true_layer[has_pen]).abs() <= 2
            pen_correct += within_2.sum().item()
            pen_total += has_pen.sum().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct_layers / total_layers:.2%}" if total_layers else "0%",
            pen_r=f"{pen_correct / pen_total:.2%}" if pen_total else "-",
            mae=f"{loc_mae_sum / loc_mae_n:.1f}" if loc_mae_n else "-",
        )
    n = len(loader) if loader else 1
    pen_recall = pen_correct / pen_total if pen_total else 0.0
    loc_mae = loc_mae_sum / loc_mae_n if loc_mae_n else 0.0
    return total_loss / n, (correct_layers / total_layers if total_layers else 0.0), pen_recall, loc_mae


def evaluate(model, loader, device, use_focal=True, focal_gamma=2.0, focal_alpha=0.75,
            weight_pos=10.0, subsample_neg=0, loc_loss_weight=0.0, kl_weight=0.0, use_amp=False):
    """在验证集上计算 Loss、Acc、PenRecall(±2)、LocMAE，不反传梯度。use_amp 时用 autocast 加速。"""
    model.eval()
    total_loss = 0.0
    correct_layers, total_layers = 0, 0
    pen_correct, pen_total = 0, 0
    loc_mae_sum, loc_mae_n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            seq_y = batch["seq_label"].to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(x)
                    if isinstance(out, tuple):
                        logits, extra = out
                    else:
                        logits, extra = out, {}
            else:
                out = model(x)
                if isinstance(out, tuple):
                    logits, extra = out
                else:
                    logits, extra = out, {}
            B, _, T = logits.shape
            logits_bt = logits.permute(0, 2, 1).reshape(-1, 2)
            seq_y_bt = seq_y.reshape(-1)

            if use_focal:
                step_loss = focal_cross_entropy(logits_bt, seq_y_bt, gamma=focal_gamma, alpha_pos=focal_alpha, device=device)
            else:
                w = torch.where(seq_y_bt == 1,
                                torch.tensor(weight_pos, device=device, dtype=logits.dtype),
                                torch.tensor(1.0, device=device, dtype=logits.dtype))
                step_loss = F.cross_entropy(logits_bt, seq_y_bt, reduction="none") * w

            if subsample_neg > 0:
                pos_mask = seq_y_bt == 1
                neg_mask = torch.zeros(B * T, dtype=torch.bool, device=device)
                for b in range(B):
                    neg_local = (seq_y[b] == 0).nonzero(as_tuple=True)[0]
                    if neg_local.numel() > subsample_neg:
                        perm = torch.randperm(neg_local.numel(), device=device)[:subsample_neg]
                        neg_global = b * T + neg_local[perm]
                    else:
                        neg_global = b * T + neg_local
                    neg_mask[neg_global] = True
                step_mask = pos_mask | neg_mask
                loss = step_loss[step_mask].mean()
            else:
                loss = step_loss.mean()

            if loc_loss_weight > 0:
                has_pen = (seq_y.sum(dim=1) > 0)
                if has_pen.any():
                    arange_t = torch.arange(T, device=logits.device, dtype=logits.dtype)
                    prob_1 = F.softmax(logits[:, 1, :], dim=1)
                    pred_soft = (prob_1 * arange_t.unsqueeze(0)).sum(dim=1)
                    true_layer = seq_y.argmax(dim=1).float()
                    loss_loc = F.smooth_l1_loss(pred_soft[has_pen], true_layer[has_pen])
                    loss = loss + loc_loss_weight * loss_loc
            if kl_weight > 0.0 and isinstance(extra, dict) and "kl_loss" in extra:
                loss = loss + kl_weight * extra["kl_loss"]

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct_layers += (pred == seq_y).sum().item()
            total_layers += seq_y.numel()
            has_pen = (seq_y.sum(dim=1) > 0)
            if has_pen.any():
                pred_layer = logits[:, 1, :].argmax(dim=1)
                true_layer = seq_y.argmax(dim=1)
                loc_mae_sum += (pred_layer[has_pen].float() - true_layer[has_pen].float()).abs().sum().item()
                loc_mae_n += has_pen.sum().item()
                within_2 = (pred_layer[has_pen] - true_layer[has_pen]).abs() <= 2
                pen_correct += within_2.sum().item()
                pen_total += has_pen.sum().item()

    n = len(loader) if loader else 1
    pen_recall = pen_correct / pen_total if pen_total else 0.0
    loc_mae = loc_mae_sum / loc_mae_n if loc_mae_n else 0.0
    return total_loss / n, (correct_layers / total_layers if total_layers else 0.0), pen_recall, loc_mae


def _load_one_hole(dataset, base_idx):
    """在 worker 或主线程中加载单孔数据，返回 (data, layer_list) 或 (None, None)。"""
    base_dataset = dataset.base_dataset
    if dataset.precomputed_dir:
        pt_path = os.path.join(dataset.precomputed_dir, dataset._precomputed_name_for_idx[base_idx])
        if not os.path.isfile(pt_path):
            pt_path = os.path.join(dataset.precomputed_dir, f"{base_idx}.pt")
        if os.path.isfile(pt_path):
            raw = torch.load(pt_path, map_location="cpu")
            return raw["data"], raw["layer_list"]
        raw = base_dataset[base_idx]
        return raw["data"], raw.get("layer_list", [])
    raw = base_dataset[base_idx]
    return raw["data"], raw.get("layer_list", [])


def evaluate_hole_level(
    model,
    dataset,
    val_base_ids,
    device,
    lock_layers=30,
    prefetch_workers=8,
    k=9,
    min_thresh=0.4,
    use_amp=True,
    unc_samples=1,
    use_uncertainty_gate=False,
    unc_var_median_thresh=0.05,
):
    """
    按孔计算验证指标：仅对标注为穿透的孔，比较预测穿透层与真实穿透层（序列下标差）。
    使用后台线程预取下一孔数据，与当前孔推理重叠，加速验证。
    unc_samples==1 时与原先一致；>1 时 Transformer 模型多次采样，可选不确定性门控（见 inference.run_inference_topkmedian）。
    """
    model.eval()
    base_dataset = dataset.base_dataset
    n_within_3, n_within_5, n_over_10, n_penetrated = 0, 0, 0, 0
    val_base_list = [b for b in val_base_ids if base_dataset.samples[b].get("penetration_layer", -1) >= 0]
    if not val_base_list:
        return {"n_penetrated": 0, "pct_within_3": 0.0, "pct_within_5": 0.0, "pct_over_10": 0.0}

    loaded_queue = queue.Queue(maxsize=prefetch_workers)

    def _producer():
        for base_idx in val_base_list:
            try:
                data, layer_list = _load_one_hole(dataset, base_idx)
                true_layer = base_dataset.samples[base_idx].get("penetration_layer", -1)
                loaded_queue.put((base_idx, data, layer_list, true_layer))
            except Exception:
                loaded_queue.put((base_idx, None, [], -1))
        loaded_queue.put(None)

    prod = threading.Thread(target=_producer, daemon=True)
    prod.start()

    use_amp = bool(use_amp) and (device.type == "cuda")
    with tqdm(total=len(val_base_list), desc="Val(按孔)", leave=False) as pbar:
        while True:
            item = loaded_queue.get()
            if item is None:
                break
            pbar.update(1)
            base_idx, data, layer_list, true_layer = item
            n_penetrated += 1
            if data is None or (hasattr(data, "numel") and data.numel() == 0) or len(layer_list) == 0:
                n_over_10 += 1
                continue
            try:
                true_idx = layer_list.index(true_layer) if true_layer in layer_list else -1
            except (ValueError, TypeError):
                true_idx = -1
            if true_idx < 0:
                n_over_10 += 1
                continue
            tkm_kw = dict(
                lock_layers=lock_layers,
                k=k,
                min_thresh=min_thresh,
                unc_samples=max(1, int(unc_samples)),
                use_uncertainty_gate=bool(use_uncertainty_gate),
                unc_var_median_thresh=float(unc_var_median_thresh),
            )
            with torch.inference_mode():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = run_inference_topkmedian(model, data, layer_list, device, **tkm_kw)
                else:
                    out = run_inference_topkmedian(model, data, layer_list, device, **tkm_kw)
            pred_idx = out.get("penetration_layer_index")
            if pred_idx is None:
                error = 999
            else:
                error = abs(pred_idx - true_idx)
            if error <= 3:
                n_within_3 += 1
            if error <= 5:
                n_within_5 += 1
            if error > 10:
                n_over_10 += 1

    pct_3 = (n_within_3 / n_penetrated * 100) if n_penetrated else 0.0
    pct_5 = (n_within_5 / n_penetrated * 100) if n_penetrated else 0.0
    pct_over10 = (n_over_10 / n_penetrated * 100) if n_penetrated else 0.0
    return {"n_penetrated": n_penetrated, "pct_within_3": pct_3, "pct_within_5": pct_5, "pct_over_10": pct_over10}


def main():
    parser = argparse.ArgumentParser(description="Grid-Diff TCN 训练")
    parser.add_argument("--samples_info", type=str, default=None, help="samples_info.json 路径")
    parser.add_argument("--base_dir", type=str, default=None, help="数据根目录，若样本路径为相对路径则使用")
    parser.add_argument("--window_len", type=int, default=60, help="窗口长度（层数）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_neg", type=float, default=1.0, help="未穿透类权重")
    parser.add_argument("--weight_pos", type=float, default=10.0, help="穿透类权重")
    parser.add_argument("--save", type=str, default="grid_diff_tcn.pt", help="模型保存路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="最多使用样本数（用于快速测试），默认全部")
    parser.add_argument("--load_workers", type=int, default=6, help="层内并行读图线程数，越大预热越快，建议 4～8")
    parser.add_argument("--img_size", type=int, default=128, help="读图缩放边长（正方形），128 已够 8x8 网格且更快")
    parser.add_argument("--roi_size", type=int, default=None, help="中心 ROI 裁剪边长，需≤img_size 且被 8 整除；默认 img_size 表示不裁，设 96 即从中心取 96×96")
    parser.add_argument("--roi_cy", type=float, default=0.5, help="ROI 中心在画面中的垂直比例，0.5=正中，孔偏上可设 0.4")
    parser.add_argument("--roi_cx", type=float, default=0.5, help="ROI 中心在画面中的水平比例，0.5=正中，孔偏左可设 0.4")
    parser.add_argument("--crop_mode", type=str, default="center", choices=["center", "roi"], help="裁剪模式：center=严格中心，roi=比例偏移")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 进程数；用 --precomputed_dir 时建议 4")
    parser.add_argument("--precomputed_dir", type=str, default="/home/student2025/wudf2025/dinov3-main/grid_diff_tcn/cache_features_train", help="预计算特征目录（见 precompute_features.py），使用后训练显著加速")
    parser.add_argument("--no_focal", action="store_true", help="关闭 Focal Loss，改用加权 CE（默认使用 Focal）")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal 的 gamma，越大越压制易分样本")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Focal 正类 alpha（0.5～0.9 更偏正类）")
    parser.add_argument("--subsample_neg", type=int, default=10, help="每样本参与损失的负时间步数，0=全部；10 可缓解不平衡")
    parser.add_argument("--penetration_radius", type=int, default=2, help="穿透层前后多少层也标 1（软标签），0=单点")
    parser.add_argument("--pos_per_batch", type=int, default=None, help="每 batch 最少正样本数，默认 batch_size//2；0=不平衡采样")
    parser.add_argument("--loc_loss_weight", type=float, default=0.5, help="辅助定位损失权重（软预测层 vs 真实层），0=不加")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="从训练集中划分验证集的比例，默认 0.2（20%%）")
    parser.add_argument("--val_seed", type=int, default=42, help="划分 train/val 的随机种子")
    # 按孔验证：每轮固定抽样 50 个验证孔（用户需求），不再使用 val_hole_every / val_hole_ratio
    parser.add_argument("--use_transformer", action="store_true", help="使用带概率注意力的 Transformer-TCN 模型")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Transformer 层数")
    parser.add_argument("--attn_dim", type=int, default=64, help="Transformer 隐层维度 d_model")
    parser.add_argument("--num_heads", type=int, default=4, help="自注意力头数")
    parser.add_argument("--kl_weight", type=float, default=0.0, help="概率注意力 KL 正则权重，0 表示不加")
    parser.add_argument("--val_holes_per_epoch", type=int, default=200, help="每轮按孔验证抽样孔数，默认 50")
    parser.add_argument("--val_prefetch_workers", type=int, default=8, help="按孔验证预取线程数，默认 8")
    parser.add_argument("--val_unc_samples", type=int, default=1, help="验证时 TopKMedian 前向次数；1=与原先一致，>1 时仅 Transformer 有多样性方差")
    parser.add_argument("--val_unc_gate", action="store_true", help="验证时若 TopKMedian 判穿透，但 top-k 方差中位数过大则改判未穿透")
    parser.add_argument("--val_unc_var_median_thresh", type=float, default=0.05, help="与 --val_unc_gate 配合：方差中位数超过此阈值则否决穿透")
    parser.add_argument("--no_batch_by_hole", action="store_true", help="关闭按孔组 batch，改用混合/平衡采样（默认同一 batch 来自同一孔）")
    parser.add_argument("--simple", action="store_true", help="简单模式：混合/平衡采样，训练中不做按孔验证（仅结束时做一次报告），按 Val Loss 保存最佳")
    parser.add_argument("--no_amp", action="store_true", help="关闭混合精度（AMP），默认 CUDA 下启用 AMP 加速")
    parser.add_argument("--pos_inject_ratio", type=float, default=0.2, help="按孔组 batch 时，全负 batch 中被选中注入穿透样本的比例（占 batch_size），默认 0.2；设 0 关闭注入")
    parser.add_argument("--neg_batch_inject_ratio", type=float, default=0.5, help="全负 batch 中有多少比例会被注入穿透样本，默认 0.5")
    args = parser.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(SCRIPT_DIR, "..", "data_drilling", "samples_info.json")
    # 统一路径分隔符：命令行若写 Windows 风格 ..\data_drilling\... 在 Linux 下会找不到，此处规范化
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if not os.path.isfile(args.samples_info):
        print("未找到 samples_info.json，请通过 --samples_info 指定。当前路径:", os.path.abspath(args.samples_info))
        return

    device = torch.device(args.device)
    img_size = max(64, args.img_size)
    roi_size = args.roi_size if args.roi_size is not None else min(96, img_size)
    roi_size = min(roi_size, img_size)
    if roi_size % 8 != 0:
        roi_size = (roi_size // 8) * 8
    if roi_size < 8:
        roi_size = 8
    roi_center_yx = (args.roi_cy, args.roi_cx)
    print(f"读图尺寸 {img_size}×{img_size}，ROI {roi_size}×{roi_size}，中心比例 (cy={args.roi_cy}, cx={args.roi_cx})，模式={args.crop_mode}")
    dataset = WindowedDrillingDataset(
        args.samples_info,
        target_size=(img_size, img_size),
        roi_size=roi_size,
        grid=(8, 8),
        base_dir=args.base_dir,
        window_len=args.window_len,
        penetration_in_window_tail_ratio=0.4,
        neg_random_windows_per_hole=2,
        skip_first_layers=30,
        max_samples=args.max_samples,
        load_workers=args.load_workers,
        roi_center_yx=roi_center_yx,
        crop_mode=args.crop_mode,
        penetration_radius=args.penetration_radius,
        precomputed_dir=os.path.normpath(args.precomputed_dir) if args.precomputed_dir else None,
    )
    if len(dataset) == 0:
        print("窗口采样后样本数为 0，请检查数据路径与 window_len")
        return

    # 按样本（孔）级别划分 train/val：先对 base_idx（孔）做 80/20 划分，再将该孔下所有窗口归入同一集合
    val_ratio = max(0.0, min(1.0, getattr(args, "val_ratio", 0.2)))
    val_seed = getattr(args, "val_seed", 42)
    if val_ratio > 0 and val_ratio < 1:
        unique_holes = sorted(set(s["base_idx"] for s in dataset.samples))
        rng = random.Random(val_seed)
        rng.shuffle(unique_holes)
        n_val_holes = max(0, int(len(unique_holes) * val_ratio))
        n_train_holes = len(unique_holes) - n_val_holes
        train_holes = set(unique_holes[:n_train_holes])
        val_holes = set(unique_holes[n_train_holes:])
        train_indices = [i for i in range(len(dataset.samples)) if dataset.samples[i]["base_idx"] in train_holes]
        val_indices = [i for i in range(len(dataset.samples)) if dataset.samples[i]["base_idx"] in val_holes]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        train_pos = [j for j in range(len(train_indices)) if dataset.samples[train_indices[j]]["label"] == 1]
        train_neg = [j for j in range(len(train_indices)) if dataset.samples[train_indices[j]]["label"] == 0]
        print(f"划分 train/val（按孔）：{n_train_holes} 孔/{len(train_ds)} 窗口 训练，{n_val_holes} 孔/{len(val_ds)} 窗口 验证（val_ratio={val_ratio}, seed={val_seed}）")
    else:
        train_ds = dataset
        val_ds = None
        train_indices = list(range(len(dataset)))
        train_pos = dataset._pos_indices
        train_neg = dataset._neg_indices
        print(f"未划分验证集（val_ratio=0 或 1），全部用于训练")

    val_base_ids = set(dataset.samples[i]["base_idx"] for i in val_indices) if val_ds is not None else set()

    if args.precomputed_dir:
        print(f"数据集就绪：共 {len(train_ds)} 个训练窗口，从预计算目录加载（可设 --num_workers 4 进一步加速）")
    else:
        print(f"数据集就绪：共 {len(train_ds)} 个训练窗口，按需读图（建议先运行 precompute_features.py 再指定 --precomputed_dir 提速）")

    # 按孔组 batch：同一 batch 内仅含同一孔的窗口，不足时用该孔内样本重复填充；--simple 时强制混合/平衡采样
    use_same_hole = not getattr(args, "no_batch_by_hole", False) and not getattr(args, "simple", False)
    if use_same_hole:
        pos_inject = getattr(args, "pos_inject_ratio", 0.2)
        neg_inject = getattr(args, "neg_batch_inject_ratio", 0.5)
        same_hole_sampler = SameHoleBatchSampler(
            dataset,
            train_indices,
            args.batch_size,
            shuffle=True,
            seed=val_seed,
            pos_inject_ratio=pos_inject,
            neg_batch_inject_ratio=neg_inject,
        )
        msg = f"按孔组 batch：同一 batch 来自同一孔，共 {len(same_hole_sampler)} batch（按层序裁剪，不足已填充）"
        if pos_inject > 0 and neg_inject > 0:
            msg += f"；全负 batch 中 {neg_inject:.0%} 注入 {pos_inject:.0%} 穿透样本"
        print(msg)
        loader = DataLoader(
            train_ds,
            batch_sampler=same_hole_sampler,
            num_workers=args.num_workers,
            collate_fn=window_collate,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
    else:
        pos_per_batch = args.pos_per_batch if args.pos_per_batch is not None else (args.batch_size // 2)
        use_balanced = pos_per_batch > 0 and len(train_pos) > 0 and len(train_neg) > 0
        if use_balanced:
            batch_sampler = BalancedBatchSampler(
                train_pos,
                train_neg,
                args.batch_size,
                min(pos_per_batch, args.batch_size),
            )
            print(f"平衡采样：每 batch 正样本 {batch_sampler.pos_per_batch}，共 {len(batch_sampler)} batch")
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                collate_fn=window_collate,
                pin_memory=(args.device == "cuda"),
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2 if args.num_workers > 0 else None,
            )
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=window_collate,
                pin_memory=(args.device == "cuda"),
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2 if args.num_workers > 0 else None,
            )

    # 验证仅做“按孔抽样”评估，不再做窗口级 val_loader（节省时间与内存）

    # 根据参数选择原始 TCN 或带 Transformer 的版本
    use_transformer = getattr(args, "use_transformer", False)
    if use_transformer:
        print(f"使用 GridDiffTCNWithTransformer: layers={args.num_transformer_layers}, d_model={args.attn_dim}, heads={args.num_heads}, kl_weight={args.kl_weight}")
    model = build_tcn_or_transformer(
        use_transformer=use_transformer,
        in_channels=64,
        out_channels=2,
        tcn_channels=(64, 64, 64, 64),
        kernel_size=3,
        d_model=getattr(args, "attn_dim", 64),
        nhead=getattr(args, "num_heads", 4),
        num_layers=getattr(args, "num_transformer_layers", 2),
        dim_feedforward=256,
        dropout=0.1,
        add_kl=True,
        return_kl=(getattr(args, "kl_weight", 0.0) > 0.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_focal = not getattr(args, "no_focal", False)
    use_amp = device.type == "cuda" and not getattr(args, "no_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    if getattr(args, "simple", False):
        print("简单模式：混合/平衡采样，按孔验证仅结束时做一次，按 Val Loss 保存最佳")
    print(f"Focal={use_focal} gamma={args.focal_gamma} alpha={args.focal_alpha} subsample_neg={args.subsample_neg} pen_radius={args.penetration_radius} loc_loss_w={args.loc_loss_weight}  AMP={use_amp}")

    best_pct_5 = -1.0
    best_val_loss = float("inf")
    for ep in range(args.epochs):
        if use_same_hole and hasattr(getattr(loader, "batch_sampler", None), "set_epoch"):
            loader.batch_sampler.set_epoch(ep)
        loss, acc, pen_recall, loc_mae = train_one_epoch(
            model, loader, optimizer, device,
            epoch=ep + 1, total_epochs=args.epochs,
            use_focal=use_focal,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            weight_pos=args.weight_pos,
            subsample_neg=args.subsample_neg,
            loc_loss_weight=args.loc_loss_weight,
            kl_weight=getattr(args, "kl_weight", 0.0),
            scaler=scaler,
            use_amp=use_amp,
        )
        # 每轮只做按孔验证：从验证集中抽样 50 个孔，计算 3 个指标（≤3、≤5、>10）
        line = f"Epoch {ep+1}/{args.epochs}  Train  Loss: {loss:.4f}  Acc: {acc:.4f}  PenRecall(±2): {pen_recall:.4f}  LocMAE: {loc_mae:.2f}"
        print(line)
        if val_ds is not None and len(val_base_ids) > 0:
            rng = random.Random(val_seed + ep)
            val_base_list = list(val_base_ids)
            n_use = min(int(getattr(args, "val_holes_per_epoch", 50)), len(val_base_list))
            subset_ids = set(rng.sample(val_base_list, n_use))
            vs = max(1, int(getattr(args, "val_unc_samples", 1)))
            vg = bool(getattr(args, "val_unc_gate", False))
            vgt = float(getattr(args, "val_unc_var_median_thresh", 0.05))
            hole_metrics = evaluate_hole_level(
                model,
                dataset,
                subset_ids,
                device,
                lock_layers=30,
                prefetch_workers=int(getattr(args, "val_prefetch_workers", 8)),
                k=9,
                min_thresh=0.4,
                unc_samples=vs,
                use_uncertainty_gate=vg,
                unc_var_median_thresh=vgt,
            )
            unc_note = ""
            if vs > 1:
                unc_note = f" unc_samples={vs}"
            if vg:
                unc_note += f" unc_gate(th={vgt})"
            print(
                f"       Val(按孔,抽样{n_use}孔,TopKMedian k=9 min_thresh=0.4{unc_note}) "
                f"穿透孔数={hole_metrics['n_penetrated']}  "
                f"误差≤3层: {hole_metrics['pct_within_3']:.1f}%  "
                f"误差≤5层: {hole_metrics['pct_within_5']:.1f}%  "
                f"误差>10层: {hole_metrics['pct_over_10']:.1f}%"
            )
            if hole_metrics["pct_within_5"] > best_pct_5:
                best_pct_5 = hole_metrics["pct_within_5"]
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "config": {"in_channels": 64, "out_channels": 2}}, args.save)
                print(f"       → 已保存最佳权重 (Val 误差≤5层: {best_pct_5:.1f}%)")
    if (not getattr(args, "simple", False) and best_pct_5 < 0) or (
        getattr(args, "simple", False) and (val_loader is None or best_val_loss == float("inf"))
    ):
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "config": {"in_channels": 64, "out_channels": 2}}, args.save)
        print(f"模型已保存: {args.save}")


if __name__ == "__main__":
    main()
