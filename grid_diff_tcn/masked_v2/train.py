# -*- coding: utf-8 -*-
"""
Two-stage training for masked image modeling (v2 - trainable encoder).

Stage 1: Pre-train encoder + decoder via MIM (encoder UNFROZEN, classifier frozen)
Stage 2: Fine-tune classifier (encoder frozen or fine-tuned)

Key difference from v1 (masked/):
  - Encoder is TRAINABLE during Stage 1 MIM, learning domain-specific features
  - This means DINOv3 features are adapted to the drilling hole domain before classification
"""

import os
import json
import math
import random
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from grid_diff_tcn.masked_v2.model import MaskedPixelModel, load_masked_model


def window_ce_weights(seq_y: torch.Tensor, window_radius: int = 5, in_window_weight: float = 2.0) -> torch.Tensor:
    """
    给穿透孔的序列位置加权：真值层±window_radius 的 timestep 权重更高，促进 <=5。
    因果约束：t=0 无法预测，只对 t>0 加权。
    返回: (B,T) float
    """
    b, t = seq_y.shape
    w = torch.ones((b, t), device=seq_y.device, dtype=torch.float32)
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return w
    true_idx = seq_y.float().argmax(dim=1)  # (B,)
    idx = torch.arange(t, device=seq_y.device).unsqueeze(0).expand(b, -1)
    # causal: only t > 0 and within window of true layer
    band = (idx - true_idx.unsqueeze(1)).abs() <= int(window_radius)
    causal_band = idx > 0
    w = torch.where(
        has_pen.unsqueeze(1) & band & causal_band,
        torch.full_like(w, float(in_window_weight)),
        w,
    )
    return w


def layer_index_loss(
    pred_idx: torch.Tensor,
    seq_y: torch.Tensor,
    frame_mask: torch.Tensor,
    lock_layers: int = 30,
    weight: float = 1.0,
    clip_pred: float = 300.0,
) -> torch.Tensor | None:
    """
    穿透层层索引回归 loss。

    pred_idx: (B,) — LearnedDecisionHead 预测的 0-based 层索引（连续值）
    seq_y:    (B, T) — ground truth label per timestep (0/1)
    frame_mask: (B, T, F) — 帧有效掩码
    lock_layers: 低于此索引的真实层不计入（安全锁）
    weight:   loss 权重系数

    Loss 组成：
    1. MAE 损失：直接惩罚预测层与真实层的距离
    2. Hinge 损失：对 >3 的误差额外惩罚（鼓励 <=3）

    只有 is_pen=1 且 pen_layer >= lock_layers 的样本才计入。
    """
    b, t = seq_y.shape
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return None

    true_idx = seq_y.float().argmax(dim=1)  # (B,)
    pen_idx = true_idx  # same thing

    # 安全锁：pen_layer 必须在 lock_layers 之后
    valid = has_pen & (pen_idx >= lock_layers)
    if not valid.any():
        return None

    # 限制预测值范围（防止极端值）
    pred_idx_clipped = pred_idx.clamp(-0.5, float(t) - 0.5)

    error = (pred_idx_clipped - pen_idx).abs()  # (B,)

    # MAE
    mae = (error * valid.float()).sum() / valid.float().sum().clamp(min=1)

    # Hinge: 额外惩罚 >3 的误差
    hinge = F.relu(error - 3.0)
    hinge_loss = (hinge * valid.float()).sum() / valid.float().sum().clamp(min=1)

    loss = mae + 0.5 * hinge_loss
    return weight * loss


def focal_cross_entropy(logits_bt, seq_y_bt, mask_bt, gamma=2.0, alpha_pos=0.75,
                        timestep_weights=None, label_smoothing=0.0):
    """
    logits_bt: (B, 2, T) classifier output
    seq_y_bt: (B, T) ground truth per time step
    mask_bt: (B, T, F) valid frame mask
    timestep_weights: (B, T) optional per-timestep weight
    label_smoothing: soft label blending factor, 0=hard labels, >0=smoother. Default 0.0.

    Causal: skips t=0 (no prediction possible for the first layer).
    """
    # logits: (B, 2, T) -> (B, T, 2)
    logits_bt = logits_bt.transpose(1, 2)
    # mask_bt has shape (B, T, F), take any valid F since labels are per T
    mask_2d = mask_bt.any(dim=2)  # (B, T) - True if any frame valid
    ignore_pad = (seq_y_bt == -100)  # (B, T)

    # Causal: only allow t > 0 (can predict from t-1 context)
    causal_mask = torch.ones_like(mask_2d)
    causal_mask[:, 0] = False

    valid_mask = mask_2d & ~ignore_pad & causal_mask
    hard_y = seq_y_bt.clamp(0, 1)
    # Label smoothing: blend hard labels with uniform distribution
    if label_smoothing > 0:
        soft_y = hard_y * (1 - label_smoothing) + label_smoothing * 0.5
    else:
        soft_y = hard_y
    valid_logits = logits_bt[valid_mask]  # (N_valid, 2)
    valid_y = seq_y_bt[valid_mask]  # (N_valid,) hard labels for alpha
    valid_soft_y = soft_y[valid_mask]  # (N_valid,) smooth labels for pt
    probs = F.softmax(valid_logits, dim=1)
    # With smoothing, pt is the model's predicted probability of the (soft) true class
    pt = probs.gather(1, valid_y.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    alpha_t = torch.where(
        valid_y == 1,
        torch.tensor(alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
        torch.tensor(1.0 - alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
    )
    focal_w = (1 - pt).pow(gamma)
    step_loss = focal_w * (-pt.log()) * alpha_t

    if timestep_weights is not None:
        tw = timestep_weights[valid_mask]
        step_loss = step_loss * tw

    return step_loss


def gaussian_peak_loss(
    logits: torch.Tensor,
    seq_y: torch.Tensor,
    sigma: float = 3.0,
    causal: bool = True,
) -> torch.Tensor | None:
    """
    logits: (B, 2, T)  — 渗漏概率取 softmax[:,1,:]
    seq_y:  (B, T)     —  label-encoded ground truth
    sigma:  高斯标准差，越小要求越精确。默认 3.0 (±3 层内都有较高目标值)
    causal: 若为 True，只对 t >= true_idx 的位置建目标（因果约束）

    鼓励预测概率在真值层附近形成高斯峰，只在渗漏孔上计算。
    全程可导，和 focal CE 梯度方向一致。
    """
    b, _, t = logits.shape
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return None

    # 真值层 index (B,)
    true_idx = seq_y.float().argmax(dim=1)

    # 高斯软目标：(B, T)
    idx = torch.arange(t, device=logits.device, dtype=torch.float32)
    idx_exp = idx.unsqueeze(0)          # (1, T)
    mu_exp  = true_idx.unsqueeze(1)     # (B, 1)
    target = torch.exp(-((idx_exp - mu_exp) ** 2) / (2 * sigma ** 2))

    if causal:
        # 因果掩码：只对 t >= true_idx 建目标
        ar = torch.arange(t, device=logits.device).unsqueeze(0).expand(b, -1)  # (B, T)
        causal_mask = (ar >= true_idx.unsqueeze(1)).float()
        target = target * causal_mask
        target = target / (target.sum(dim=1, keepdim=True).clamp(min=1e-8))   # 归一化

    prob = F.softmax(logits, dim=1)[:, 1]  # (B, T)
    loss = F.mse_loss(prob[has_pen], target[has_pen])
    return loss


def temporal_smoothness_loss(
    logits: torch.Tensor,
    seq_y: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor | None:
    """
    时间平滑损失：惩罚预测概率的剧烈波动，促进连续预测。
    只在有穿透的样本上计算。
    
    logits: (B, 2, T)
    seq_y: (B, T)
    """
    b, _, t = logits.shape
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return None
    
    prob = F.softmax(logits, dim=1)[:, 1]  # (B, T)
    # 一阶差分
    diff = prob[:, 1:] - prob[:, :-1]  # (B, T-1)
    # 对有穿透的样本计算L2损失
    loss = (diff ** 2).mean()
    return weight * loss


def boundary_aware_loss(
    logits: torch.Tensor,
    seq_y: torch.Tensor,
    boundary_weight: float = 0.15,
) -> torch.Tensor | None:
    """
    边界感知损失：鼓励预测概率在真值层附近上升，在其他区域保持平稳。
    只在穿透样本上计算。
    
    设计思路：
    - 对t < true_idx的位置：概率应该较低（因为还没有穿透）
    - 对t >= true_idx的位置：概率应该较高（已经穿透）
    这是一个软边界约束，与高斯峰损失互补。
    """
    b, _, t = logits.shape
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return None
    
    true_idx = seq_y.float().argmax(dim=1)  # (B,)
    prob = F.softmax(logits, dim=1)[:, 1]  # (B, T)
    
    # 构建软目标：t >= true_idx 时目标为1，否则为0
    idx = torch.arange(t, device=logits.device).unsqueeze(0).expand(b, -1)  # (B, T)
    target = (idx >= true_idx.unsqueeze(1)).float()
    
    # 对有穿透的样本计算MSE
    loss = F.mse_loss(prob[has_pen], target[has_pen])
    return boundary_weight * loss


def train_stage1(
    model: MaskedPixelModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: GradScaler | None = None,
) -> dict:
    """Train stage 1: encoder + decoder MIM pre-training (classifier frozen)."""
    # Freeze classifier: only encoder + decoder should be trained
    model.freeze_classifier()
    # Ensure encoder is trainable (freeze_encoder=False by default in v2)
    model.set_encoder_trainable(True)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}")
        for batch in pbar:
            frame_data = batch["frame_data"].to(device)  # (B, T, F, 3, H, W)
            B = frame_data.shape[0]
            T = frame_data.shape[1]
            F_frames = frame_data.shape[2]
            H = frame_data.shape[4]
            W = frame_data.shape[5]
            
            # 整个 batch 一次前向：flatten (B,T,F,3,H,W) -> (B*T*F, 3, H, W)
            B, T, F_frames = frame_data.shape[0], frame_data.shape[1], frame_data.shape[2]
            chunk_flat = frame_data.reshape(B * T * F_frames, 3, H, W)
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    result = model.forward_stage1(chunk_flat)
                    loss = result["loss"]
                scaler.scale(loss).backward()
            else:
                result = model.forward_stage1(chunk_flat)
                loss = result["loss"]
                loss.backward()
            
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        if val_loader is not None:
            val_loss = evaluate_stage1(model, val_loader, device, args)
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                if args.save:
                    save_checkpoint(model, optimizer, epoch, args, stage=1)
        else:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                if args.save:
                    save_checkpoint(model, optimizer, epoch, args, stage=1)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    return {"best_loss": best_loss}


def evaluate_stage1(
    model: MaskedPixelModel,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """Evaluate stage 1 model."""
    model.eval()
    # Freeze classifier to match training behavior
    model.freeze_classifier()
    total_loss = 0.0
    num_batches = 0
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Evaluating"):
            frame_data = batch["frame_data"].to(device)  # (B, T, F, 3, H, W)
            B, T, F_frames = frame_data.shape[0], frame_data.shape[1], frame_data.shape[2]
            H, W = frame_data.shape[4], frame_data.shape[5]
            
            chunk_flat = frame_data.reshape(B * T * F_frames, 3, H, W)
            result = model.forward_stage1(chunk_flat)
            total_loss += result["loss"].item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train_stage2(
    model: MaskedPixelModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    scaler: GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> dict:
    """Train stage 2: classification fine-tuning (encoder frozen or fine-tuned)."""
    model.stage = 2

    # Unfreeze classifier for training
    model.unfreeze_classifier()

    # Encoder strategy:
    #   --unfreeze_encoder_stage2=True  → fine-tune the Stage 1 trained encoder
    #   --unfreeze_encoder_stage2=False → keep encoder frozen (use Stage 1 learned features as-is)
    unfreeze_stage2 = getattr(args, "unfreeze_encoder_stage2", False)
    if unfreeze_stage2:
        model.set_encoder_trainable(True)
        print("  [Stage2] Encoder fine-tuning ENABLED (unfreeze_encoder_stage2=True)")
    else:
        model.set_encoder_trainable(False)
        print("  [Stage2] Encoder FROZEN (using Stage 1 learned features, unfreeze_encoder_stage2=False)")
    
    # Training hyperparameters
    weight_pos = getattr(args, "weight_pos", 3.0)
    peak_loss_weight = getattr(args, "peak_loss_weight", 0.2)
    label_smoothing = getattr(args, "label_smoothing", 0.05)
    use_learned_decision = getattr(args, "use_learned_decision", False)
    index_loss_weight = getattr(args, "index_loss_weight", 1.0)
    lock_layers = getattr(args, "lock_layers", 30)
    smoothness_weight = getattr(args, "smoothness_weight", 0.1)
    boundary_weight = getattr(args, "boundary_weight", 0.15)
    
    # Gradient accumulation for Stage 2
    accum_steps = getattr(args, "accum_steps_stage2", 1)
    if accum_steps > 1:
        print(f"  [Stage2] Using gradient accumulation: accum_steps={accum_steps}")

    best_metric = -1.0  # maximise <=3 pct via s3wd grid search
    patience_counter = 0
    patience = getattr(args, "patience", 10)
    
    # Initialize accum loss tracking
    accum_loss = 0.0
    accum_peak_loss = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_peak_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            frame_data = batch["frame_data"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            seq_label = batch["seq_label"].to(device)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    result = model.forward(
                        frame_data, frame_mask=frame_mask,
                        return_decision_idx=use_learned_decision,
                    )
                    logits = result["logits"]

                    lock_layers = getattr(args, "lock_layers", 30)
                    if lock_layers > 0:
                        logits[:, 1, :lock_layers] = float("-inf")
                        seq_label_local = seq_label.clone()
                        seq_label_local[:, :lock_layers] = 0
                    else:
                        seq_label_local = seq_label.clone()

                    seq_label_safe = seq_label_local.clone()
                    seq_label_safe[seq_label_safe == -100] = 0

                    tw = window_ce_weights(seq_label_safe).to(device)
                    step_loss = focal_cross_entropy(
                        logits, seq_label_local, frame_mask,
                        alpha_pos=weight_pos / (weight_pos + 1.0),
                        timestep_weights=tw,
                        label_smoothing=label_smoothing,
                    )
                    focal_loss = step_loss.mean()
                    total_loss = focal_loss

                    # Gaussian peak loss (定位软约束)
                    if peak_loss_weight > 0:
                        pk_loss = gaussian_peak_loss(logits, seq_label_safe, sigma=3.0, causal=True)
                        if pk_loss is not None:
                            total_loss = total_loss + float(peak_loss_weight) * pk_loss
                            accum_peak_loss += pk_loss.item()

                    # Temporal smoothness loss (时间平滑)
                    if smoothness_weight > 0:
                        sm_loss = temporal_smoothness_loss(logits, seq_label_safe, weight=smoothness_weight)
                        if sm_loss is not None:
                            total_loss = total_loss + sm_loss

                    # Boundary-aware loss (边界感知)
                    if boundary_weight > 0:
                        bd_loss = boundary_aware_loss(logits, seq_label_safe, boundary_weight=boundary_weight)
                        if bd_loss is not None:
                            total_loss = total_loss + bd_loss

                    if use_learned_decision and index_loss_weight > 0:
                        decision_idx = result.get("decision_idx")
                        if decision_idx is not None:
                            idx_loss = layer_index_loss(
                                decision_idx, seq_label_safe, frame_mask,
                                lock_layers=lock_layers, weight=index_loss_weight,
                            )
                            if idx_loss is not None:
                                total_loss = total_loss + idx_loss

                # Scale loss for gradient accumulation
                loss_for_backward = total_loss / accum_steps
                scaler.scale(loss_for_backward).backward()
            else:
                result = model.forward(
                    frame_data, frame_mask=frame_mask,
                    return_decision_idx=use_learned_decision,
                )
                logits = result["logits"]

                lock_layers = getattr(args, "lock_layers", 30)
                if lock_layers > 0:
                    logits[:, 1, :lock_layers] = float("-inf")
                    seq_label_local = seq_label.clone()
                    seq_label_local[:, :lock_layers] = 0
                else:
                    seq_label_local = seq_label.clone()

                seq_label_safe = seq_label_local.clone()
                seq_label_safe[seq_label_safe == -100] = 0

                tw = window_ce_weights(seq_label_safe).to(device)
                step_loss = focal_cross_entropy(
                    logits, seq_label_local, frame_mask,
                    alpha_pos=weight_pos / (weight_pos + 1.0),
                    timestep_weights=tw,
                    label_smoothing=label_smoothing,
                )
                focal_loss = step_loss.mean()
                total_loss = focal_loss

                # Gaussian peak loss (定位软约束)
                if peak_loss_weight > 0:
                    pk_loss = gaussian_peak_loss(logits, seq_label_safe, sigma=3.0, causal=True)
                    if pk_loss is not None:
                        total_loss = total_loss + float(peak_loss_weight) * pk_loss
                        accum_peak_loss += pk_loss.item()

                # Temporal smoothness loss (时间平滑)
                if smoothness_weight > 0:
                    sm_loss = temporal_smoothness_loss(logits, seq_label_safe, weight=smoothness_weight)
                    if sm_loss is not None:
                        total_loss = total_loss + sm_loss

                # Boundary-aware loss (边界感知)
                if boundary_weight > 0:
                    bd_loss = boundary_aware_loss(logits, seq_label_safe, boundary_weight=boundary_weight)
                    if bd_loss is not None:
                        total_loss = total_loss + bd_loss

                if use_learned_decision and index_loss_weight > 0:
                    decision_idx = result.get("decision_idx")
                    if decision_idx is not None:
                        idx_loss = layer_index_loss(
                            decision_idx, seq_label_safe, frame_mask,
                            lock_layers=lock_layers, weight=index_loss_weight,
                        )
                        if idx_loss is not None:
                            total_loss = total_loss + idx_loss

                # Scale loss for gradient accumulation
                loss_for_backward = total_loss / accum_steps
                loss_for_backward.backward()

            accum_loss += focal_loss.item()
            
            # Update weights every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            num_batches += 1
            # Display running average loss
            running_loss = accum_loss / num_batches
            pbar.set_postfix({"loss": f"{running_loss:.4f}"})
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        avg_loss = accum_loss / max(num_batches, 1)
        avg_peak_loss = accum_peak_loss / max(num_batches, 1)
        
        # Reset accum tracking for next epoch
        accum_loss = 0.0
        accum_peak_loss = 0.0
        
        if val_loader is not None:
            metrics = evaluate_stage2(model, val_loader, device, args, compute_loss=True, debug=True)
            val_loss = metrics.get("avg_val_loss") or 0.0
            s3wd_info = metrics.get("best_s3wd_params", {})
            # Combined metric: pct_within_3 + pct_within_5
            s3wd_combined = metrics.get("best_s3wd_metric", -1)
            s3wd_within_3 = s3wd_info.get("pct_within_3", 0)
            s3wd_within_5 = s3wd_info.get("pct_within_5", 0)
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}(pk={avg_peak_loss:.4f}), val_loss={val_loss:.4f}, ")
            if metrics.get("learned_decision"):
                print(f"  [LearnedDecision] <=3:{s3wd_within_3*100:.1f}%, <=5:{s3wd_within_5*100:.1f}%, "
                      f"<=3+<=5={s3wd_combined*100:.1f}%, >=10:{s3wd_info.get('pct_over_10',0)*100:.1f}%")
            else:
                print(f"  [S3WD grid] best: wait={s3wd_info.get('wait','?')}, "
                      f"thresh={s3wd_info.get('threshold','?')}, "
                      f"accept={s3wd_info.get('accept','?')}, "
                      f"<=3:{s3wd_within_3*100:.1f}%, <=5:{s3wd_within_5*100:.1f}%, "
                      f"<=3+<=5={s3wd_combined*100:.1f}%, "
                      f">=10:{s3wd_info.get('pct_over_10',0)*100:.1f}%")

            if s3wd_combined > best_metric:
                best_metric = s3wd_combined
                patience_counter = 0
                if args.save:
                    save_checkpoint(model, optimizer, epoch, args, stage=2,
                                    s3wd_params=s3wd_info)
                    if metrics.get("learned_decision"):
                        print(f"  [Best] Saved — learned_decision <=3+<=5: {s3wd_combined*100:.1f}%")
                    else:
                        print(f"  [Best] Saved — s3wd <=3+<=5: {s3wd_combined*100:.1f}% "
                              f"(wait={s3wd_info.get('wait')}, thresh={s3wd_info.get('threshold')}, "
                              f"accept={s3wd_info.get('accept')})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  [LR] {current_lr:.2e}")

        else:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
    
    return {"best_metric": best_metric}


def evaluate_stage2(
    model: MaskedPixelModel,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    compute_loss: bool = False,
    debug: bool = False,
) -> dict:
    """Evaluate stage 2 model.

    When use_learned_decision=True, uses the model's LearnedDecisionHead
    directly (decision_idx) and skips S3WD grid search.
    """
    model.eval()

    use_learned_decision = getattr(args, "use_learned_decision", False)

    all_preds = []
    all_labels = []
    all_penetration_layers = []
    all_layer_lists = []
    all_frame_probs = []
    total_val_loss = 0.0
    num_val_batches = 0

    weight_pos = getattr(args, "weight_pos", 3.0)
    peak_loss_weight = getattr(args, "peak_loss_weight", 0.2)
    smoothness_weight = getattr(args, "smoothness_weight", 0.1)
    boundary_weight = getattr(args, "boundary_weight", 0.15)
    label_smoothing = getattr(args, "label_smoothing", 0.05)
    lock_layers = getattr(args, "lock_layers", 30)

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Evaluating"):
            frame_data = batch["frame_data"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            labels = batch["labels"].to(device)
            penetration_layers = batch["penetration_layers"]
            layer_lists = batch["layer_lists"]

            result = model.forward(
                frame_data, frame_mask=frame_mask,
                return_decision_idx=use_learned_decision,
            )
            logits = result["logits"]

            if lock_layers > 0:
                logits[:, 1, :lock_layers] = float("-inf")
                if compute_loss:
                    seq_label = batch["seq_label"].to(device)
                    seq_label = seq_label.clone()
                    seq_label[:, :lock_layers] = 0
            else:
                seq_label = batch["seq_label"].to(device) if compute_loss else None

            if compute_loss:
                seq_label_safe = seq_label.clone()
                seq_label_safe[seq_label_safe == -100] = 0
                tw = window_ce_weights(seq_label_safe).to(device)
                step_loss = focal_cross_entropy(
                    logits, seq_label, frame_mask,
                    alpha_pos=weight_pos / (weight_pos + 1.0),
                    timestep_weights=tw,
                    label_smoothing=label_smoothing,
                )
                val_loss = step_loss.mean()
                if peak_loss_weight > 0:
                    pk_loss = gaussian_peak_loss(logits, seq_label_safe, sigma=3.0, causal=True)
                    if pk_loss is not None:
                        val_loss = val_loss + float(peak_loss_weight) * pk_loss
                total_val_loss += val_loss.item()
                num_val_batches += 1

            if use_learned_decision:
                # Use learned decision head directly
                decision_idx = result.get("decision_idx")
                pred_layers = []
                for bi in range(logits.shape[0]):
                    label = labels[bi].item()
                    if decision_idx is not None and label == 1:
                        layer_list = layer_lists[bi]
                        raw_pred_idx = decision_idx[bi].item()
                        # Clamp to valid layer list range
                        pred_idx_clamped = max(0, min(int(round(raw_pred_idx)), len(layer_list) - 1))
                        pred_layer = layer_list[pred_idx_clamped]
                    else:
                        pred_layer = -1
                    pred_layers.append(pred_layer)
            else:
                # Argmax prediction
                probs = F.softmax(logits, dim=1)[:, 1]
                batch_probs = []
                for bi in range(logits.shape[0]):
                    prob_t = probs[bi]
                    mask_bi = frame_mask[bi]
                    mask_t = mask_bi.any(dim=1)
                    batch_probs.append((prob_t.cpu(), mask_t.cpu()))
                all_frame_probs.extend(batch_probs)

                pred_layers = []
                for bi in range(logits.shape[0]):
                    prob_t = probs[bi]
                    mask_bi = frame_mask[bi]
                    mask_t = mask_bi.any(dim=1)
                    if mask_t.sum() == 0:
                        pred_layers.append(-1)
                        continue
                    valid_probs = prob_t[mask_t]
                    valid_idx = torch.where(mask_t)[0]
                    if valid_probs.numel() == 0:
                        pred_layers.append(-1)
                        continue
                    max_idx = valid_probs.argmax()
                    pred_layer = valid_idx[max_idx].item()
                    pred_layers.append(pred_layer)

            all_preds.extend(pred_layers)
            all_labels.extend(labels.cpu().tolist())
            all_penetration_layers.extend(penetration_layers.cpu().tolist())
            all_layer_lists.extend(layer_lists)

    metrics = compute_metrics(
        all_preds, all_labels, all_penetration_layers, all_layer_lists,
        lock_layers=lock_layers,
        debug=debug,
    )

    if compute_loss and num_val_batches > 0:
        metrics["avg_val_loss"] = total_val_loss / num_val_batches
    else:
        metrics["avg_val_loss"] = None

    if use_learned_decision:
        metrics["best_s3wd_metric"] = metrics.get("pct_within_3", 0) + metrics.get("pct_within_5", 0)
        metrics["best_s3wd_params"] = {
            "mode": "learned_decision",
            "pct_within_3": metrics.get("pct_within_3", 0),
            "pct_within_5": metrics.get("pct_within_5", 0),
            "pct_over_10": metrics.get("pct_over_10", 0),
            "total": metrics.get("total", 0),
        }
        metrics["learned_decision"] = True
    else:
        # S3WD grid search over (threshold, accept, wait)
        s3wd_thresholds = [ 0.6]
        s3wd_accept_vals = [0.4]
        s3wd_wait_vals = [3]

        best_s3wd_metric = -1
        best_s3wd_params = {}

        for wait in s3wd_wait_vals:
            for thresh in s3wd_thresholds:
                for accept in s3wd_accept_vals:
                    s3wd_preds = []
                    for (prob_t, mask_t), label, pen_layer, layers in zip(
                            all_frame_probs, all_labels, all_penetration_layers, all_layer_lists):
                        if label != 1:
                            s3wd_preds.append(-1)
                            continue
                        if pen_layer < 0 or pen_layer not in layers:
                            s3wd_preds.append(-1)
                            continue
                        pen_idx = layers.index(pen_layer)
                        if pen_idx < lock_layers:
                            s3wd_preds.append(-1)
                            continue

                        consecutive_high = 0
                        pred_layer = -1
                        for ti in range(len(prob_t)):
                            if not mask_t[ti]:
                                continue
                            if prob_t[ti] >= thresh:
                                consecutive_high += 1
                                if consecutive_high >= wait:
                                    pred_layer = ti
                                    break
                            else:
                                consecutive_high = 0
                        s3wd_preds.append(pred_layer)

                    s3wd_metrics = compute_metrics(
                        s3wd_preds, all_labels, all_penetration_layers, all_layer_lists,
                        lock_layers=lock_layers,
                    )
                    combined = s3wd_metrics["pct_within_3"] + s3wd_metrics["pct_within_5"]
                    if combined >= best_s3wd_metric:
                        best_s3wd_metric = combined
                        best_s3wd_params = {"wait": wait, "threshold": thresh, "accept": accept,
                                            "pct_within_3": s3wd_metrics["pct_within_3"],
                                            "pct_within_5": s3wd_metrics["pct_within_5"],
                                            "pct_over_10": s3wd_metrics["pct_over_10"],
                                            "total": s3wd_metrics["total"]}

        metrics["best_s3wd_metric"] = best_s3wd_metric
        metrics["best_s3wd_params"] = best_s3wd_params
        metrics["learned_decision"] = False

    return metrics


def compute_metrics(
    preds: list,
    labels: list,
    penetration_layers: list,
    layer_lists: list,
    lock_layers: int = 30,
    debug: bool = False,
) -> dict:
    """Compute evaluation metrics. Samples with penetration_layer in first lock_layers layers are excluded."""
    within_3 = 0
    within_5 = 0
    over_10 = 0
    total = 0
    skipped_locked = 0
    skipped_invalid = 0
    
    for pred, label, pen_layer, layers in zip(preds, labels, penetration_layers, layer_lists):
        if label != 1:
            continue
        if pen_layer < 0 or pen_layer not in layers:
            skipped_invalid += 1
            continue
        pen_idx = layers.index(pen_layer)
        if pen_idx < lock_layers:
            skipped_locked += 1
            continue
        if pred < 0 or pred not in layers:
            skipped_invalid += 1
            continue
        
        true_idx = layers.index(pen_layer)
        pred_idx = layers.index(pred)
        error = abs(pred_idx - true_idx)
        
        total += 1
        if error <= 3:
            within_3 += 1
        if error <= 5:
            within_5 += 1
        if error > 10:
            over_10 += 1
    
    if debug and total == 0:
        print(f"  [DEBUG] compute_metrics: total=0, skipped_locked={skipped_locked}, skipped_invalid={skipped_invalid}")
    
    if total == 0:
        return {"pct_within_3": 0.0, "pct_within_5": 0.0, "pct_over_10": 0.0, "total": 0}
    
    return {
        "pct_within_3": within_3 / total,
        "pct_within_5": within_5 / total,
        "pct_over_10": over_10 / total,
        "total": total,
    }


def save_checkpoint(
    model: MaskedPixelModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    stage: int,
    s3wd_params: dict | None = None,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "config": {
            "dinov3_model": args.dinov3_model,
            "dinov3_feat_dim": args.dinov3_feat_dim,
            "d_model": args.d_model,
            "freeze_encoder": args.freeze_encoder,
            "mask_ratio": args.mask_ratio,
        },
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if s3wd_params:
        checkpoint["s3wd_params"] = s3wd_params
    torch.save(checkpoint, args.save)
    print(f"Saved checkpoint to {args.save}")


def main():
    parser = argparse.ArgumentParser(description="Train masked image model")
    
    parser.add_argument("--samples_info", type=str, required=True)
    parser.add_argument("--val_samples_info", type=str,
                        default="data_drilling/samples_info_val.json",
                        help="Validation samples info. Default: data_drilling/samples_info_val.json")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    
    parser.add_argument("--dinov3_model", type=str, default="vit_small")
    parser.add_argument("--dinov3_feat_dim", type=int, default=384)
    parser.add_argument("--dinov3_roi_size", type=int, default=224)
    
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    
    parser.add_argument("--freeze_encoder", type=lambda x: x.lower() == "true", default=False,
                        help="[v2] Freeze DINOv3 encoder during training. Default False (encoder is trainable in Stage 1).")
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--mask_shape", type=str, default="circle")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-6,
                        help="Learning rate for encoder during Stage 1. Default 1e-6 (10x lower than decoder).")
    parser.add_argument("--encoder_lr_stage2", type=float, default=1e-6,
                        help="[v2] Learning rate for encoder in Stage 2 (only used when --unfreeze_encoder_stage2=True). Default 1e-6.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--max_frames_per_layer", type=int, default=8)
    parser.add_argument("--dinov3_chunk_size", type=int, default=4,
                        help="Number of images processed by DINOv3 at once (lower = less GPU memory)")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps for Stage 1")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="Directory with pre-extracted DINOv3 features (.pt per sample). "
                             "Must be used together with --use_cached_features. "
                             "Cache generated with: python -m grid_diff_tcn.masked.extract")
    parser.add_argument("--use_cached_features", action="store_true",
                        help="Skip DINOv3 forward pass during training by loading pre-extracted "
                             "features from --precomputed_dir. Requires --precomputed_dir.")
    parser.add_argument("--crop_cache_dir", type=str, default=None,
                        help="Directory with pre-cropped ROI .pt files (generated by pre_crop.py). "
                             "If set, uses CropCacheDataset — training reads .pt directly, "
                             "bypassing all image processing. infer.py always uses online cropping.")
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--preload", action="store_true",
                        help="Load all data into RAM at startup. "
                             "For CropCacheDataset: load .pt files. "
                             "For MaskedDrillingDataset: scan dirs only. "
                             "Eliminates disk I/O during training.")
    parser.add_argument("--preload_workers", type=int, default=8,
                        help="Number of workers for parallel ROI extraction (preload mode).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Debug: limit dataset to N samples for quick dry-run. "
                             "E.g. --max_samples 5 runs on 5 samples only.")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--weight_pos", type=float, default=3.0,
                        help="Positive class weight for focal CE. Default 3.0 (was 10.0).")
    parser.add_argument("--peak_loss_weight", type=float, default=0.2,
                        help="Weight for gaussian_peak_loss (定位软约束). Default 0.2. Set to 0 to disable.")
    parser.add_argument("--smoothness_weight", type=float, default=0.1,
                        help="Weight for temporal_smoothness_loss (时间平滑). Default 0.1. Set to 0 to disable.")
    parser.add_argument("--boundary_weight", type=float, default=0.15,
                        help="Weight for boundary_aware_loss (边界感知). Default 0.15. Set to 0 to disable.")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label smoothing for focal CE. Default 0.05 (blends hard labels with 0.5).")
    parser.add_argument("--stage2_scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"],
                        help="LR scheduler for Stage 2. Default cosine.")
    parser.add_argument("--lock_layers", type=int, default=30,
                        help="Training layers before this index are forced to 0 penetration probability. "
                             "Inference will also treat them as non-penetrated. Default: 30.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to Stage 1 checkpoint to resume/finetune from in Stage 2. "
                             "Classifier is re-initialized, encoder weights are loaded.")
    parser.add_argument("--unfreeze_encoder", type=lambda x: x.lower() == "true", default=False,
                        help="If resuming from Stage 1, unfreeze encoder after loading (Stage 2 only). Default False.")
    parser.add_argument("--finetune_classifier", type=lambda x: x.lower() == "true", default=True,
                        help="If resuming from Stage 1, try to load and finetune classifier weights. Default True.")
    parser.add_argument("--unfreeze_encoder_stage2", type=lambda x: x.lower() == "true", default=False,
                        help="[v2] In Stage 2, continue training the encoder (not just classifier). "
                             "Use when you want to fine-tune the Stage 1 learned encoder. Default False (frozen).")
    parser.add_argument("--use_learned_decision", action="store_true",
                        help="Enable learned decision head (layer index regression). "
                             "Adds a LearnedDecisionHead that directly predicts penetration layer index, "
                             "eliminating the need for S3WD threshold tuning. Default: disabled.")
    parser.add_argument("--index_loss_weight", type=float, default=1.0,
                        help="Weight for layer_index_loss when use_learned_decision=True. Default 1.0. "
                             "Set to 0 to disable index loss while keeping the head (for ablation).")
    parser.add_argument("--accum_steps_stage2", type=int, default=4,
                        help="Gradient accumulation steps for Stage 2. Default 4.")

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from grid_diff_tcn.masked_v2.dataset import (
        MaskedDrillingDataset,
        CropCacheDataset,
        collate_masked_batch,
    )

    use_crop_cache = bool(args.crop_cache_dir)

    if use_crop_cache:
        print(f"[main] Using CropCacheDataset (cache={args.crop_cache_dir})")
        train_dataset = CropCacheDataset(
            samples_info_path=args.samples_info,
            cache_dir=args.crop_cache_dir,
            roi_size=args.dinov3_roi_size,
            max_layers=args.max_layers,
            max_frames_per_layer=args.max_frames_per_layer,
            preload=args.preload,
            precomputed_dir=args.precomputed_dir,
            max_samples=args.max_samples,
        )
        val_samples_info = args.val_samples_info
        if os.path.exists(val_samples_info):
            val_dataset = CropCacheDataset(
                samples_info_path=val_samples_info,
                cache_dir=args.crop_cache_dir,
                roi_size=args.dinov3_roi_size,
                max_layers=args.max_layers,
                max_frames_per_layer=args.max_frames_per_layer,
                preload=args.preload,
                precomputed_dir=args.precomputed_dir,
                max_samples=args.max_samples,
            )
        else:
            val_dataset = None
        # .pt 文件读取无图像处理开销，num_workers=0 足够
        dl_workers = 0
    else:
        print(f"[main] Using MaskedDrillingDataset (online ROI extraction)")
        train_dataset = MaskedDrillingDataset(
            samples_info_path=args.samples_info,
            roi_size=args.dinov3_roi_size,
            max_layers=args.max_layers,
            max_frames_per_layer=args.max_frames_per_layer,
            preload=args.preload,
            preload_workers=args.preload_workers,
            max_samples=args.max_samples,
        )
        val_samples_info = args.val_samples_info
        if os.path.exists(val_samples_info):
            val_dataset = MaskedDrillingDataset(
                samples_info_path=val_samples_info,
                roi_size=args.dinov3_roi_size,
                max_layers=args.max_layers,
                max_frames_per_layer=args.max_frames_per_layer,
                preload=args.preload,
                preload_workers=args.preload_workers,
                max_samples=args.max_samples,
            )
        else:
            val_dataset = None
        dl_workers = args.num_workers

    collate_fn = collate_masked_batch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dl_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # batch contains layer_lists (Python list) + sample_paths (str) — cannot pin
        persistent_workers=True if dl_workers > 0 else False,
        prefetch_factor=2 if dl_workers > 0 else None,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=dl_workers,
            collate_fn=collate_fn,
            pin_memory=False,  # batch contains layer_lists (Python list) + sample_paths (str) — cannot pin
            persistent_workers=True if dl_workers > 0 else False,
            prefetch_factor=2 if dl_workers > 0 else None,
        )

  

    model_kwargs = dict(
        dinov3_model=args.dinov3_model,
        dinov3_feat_dim=args.dinov3_feat_dim,
        dinov3_roi_size=args.dinov3_roi_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_layers,
        freeze_encoder=args.freeze_encoder,
        mask_ratio=args.mask_ratio,
        mask_shape=args.mask_shape,
        use_cached_features=args.use_cached_features,
        dinov3_chunk_size=args.dinov3_chunk_size,
    )

    if args.use_cached_features and args.precomputed_dir:
        print(f"[main] use_cached_features=True, will skip DINOv3 forward (features from {args.precomputed_dir})")

    # Stage 2: 优先从 --resume_from 加载 Stage 1 权重
    if args.stage == 2 and args.resume_from and os.path.exists(args.resume_from):
        print(f"[Stage2] 从 Stage1 权重恢复: {args.resume_from}")
        finetune_classifier = getattr(args, "finetune_classifier", True)
        model = load_masked_model(
            args.resume_from,
            stage=2,
            unfreeze_encoder=args.unfreeze_encoder,
            finetune_classifier=finetune_classifier,
            **model_kwargs,
        )
    else:
        model = MaskedPixelModel(**model_kwargs)

    model = model.to(device)

    # Create optimizer: strategy depends on stage and encoder freeze setting
    if args.stage == 1:
        # Stage 1: encoder + decoder trained together (classifier frozen)
        # Model is initialized with freeze_encoder=False by default in v2
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if "dinov3" in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": decoder_params, "lr": args.lr},
        ], weight_decay=0.01)
        scaler = GradScaler()
    else:
        # Stage 2: classifier trained (encoder frozen by default)
        unfreeze_stage2 = getattr(args, "unfreeze_encoder_stage2", False)
        if unfreeze_stage2:
            encoder_params = list(model.dinov3_extractor.parameters())
            classifier_params = list(model.classifier.parameters())
            decoder_params = list(model.pixel_decoder.parameters()) if hasattr(model, "pixel_decoder") else []
            encoder_lr_s2 = getattr(args, "encoder_lr_stage2", 1e-6)
            optimizer = torch.optim.AdamW(
                [{"params": encoder_params, "lr": encoder_lr_s2},
                 {"params": classifier_params, "lr": args.lr}]
                + ([{"params": decoder_params, "lr": args.lr}] if decoder_params else []),
                weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=0.01)

        stage2_scheduler = getattr(args, "stage2_scheduler", "cosine")
        if args.stage == 2 and stage2_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        elif args.stage == 2 and stage2_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3)
        else:
            scheduler = None

        scaler = GradScaler()

    if args.stage == 1:
        train_stage1(model, train_loader, val_loader, optimizer, device, args, scaler)
    else:
        train_stage2(model, train_loader, val_loader, optimizer, device, args, scaler, scheduler)


if __name__ == "__main__":
    main()