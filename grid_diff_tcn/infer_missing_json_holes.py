# -*- coding: utf-8 -*-
"""
对“无标注 JSON”的孔目录做穿透推理。

功能：在 data_drilling/train 下扫描有 jpg 但无 Auto.json/Project.json 的孔目录，构建临时 samples_info，
      加载 GridDiffTCN 权重并调用 inference.run_inference 逐孔推理，打印是否穿透及穿透层（可选 --topk 限制数量）。
依赖：dataset, tcn_model, inference；训练好的 grid_diff_tcn.pt；data_drilling/train 目录结构。
输出：终端打印每孔推理结果（是否穿透、穿透层号等）。
主要参数：--ckpt, --train_root, --topk。
示例：python infer_missing_json_holes.py --ckpt ./grid_diff_tcn.pt --topk 5
"""

import os
import sys
import glob
import argparse
import random

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import GridDiffDrillingDataset
from tcn_model import GridDiffTCN
from inference import run_inference


def find_missing_json_holes(train_root: str):
    jpgs = glob.glob(os.path.join(train_root, "**", "*.jpg"), recursive=True)
    hole_dirs = sorted(set(os.path.dirname(p) for p in jpgs))
    missing = []
    for d in hole_dirs:
        if not (os.path.isfile(os.path.join(d, "Auto.json")) or os.path.isfile(os.path.join(d, "Project.json"))):
            missing.append(d)
    return missing


def build_temp_samples_info(out_path: str, hole_dirs):
    # GridDiffDrillingDataset 只需要 sample_path 存在并包含 jpg 即可
    samples = []
    for d in hole_dirs:
        samples.append({
            "category": os.path.basename(os.path.dirname(os.path.dirname(d))) if d else "unknown",
            "sample_path": d,
            "is_penetrated": 0,
            "penetration_layer": -1,
        })
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="对缺少 json 标注的孔做推理")
    ap.add_argument("--train_root", type=str, default=None, help="data_drilling/train 路径")
    ap.add_argument("--ckpt", type=str, default="grid_diff_tcn.pt", help="模型权重路径")
    ap.add_argument("--topk", type=int, default=5, help="挑选多少个孔推理")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--roi_size", type=int, default=96)
    ap.add_argument("--roi_cy", type=float, default=0.5)
    ap.add_argument("--roi_cx", type=float, default=0.5)
    ap.add_argument("--crop_mode", type=str, default="center", choices=["center", "roi"])
    ap.add_argument("--lock_layers", type=int, default=30)
    args = ap.parse_args()

    if args.train_root is None:
        args.train_root = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data_drilling", "train"))
    train_root = os.path.abspath(args.train_root)
    if not os.path.isdir(train_root):
        raise SystemExit(f"train_root 不存在: {train_root}")
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"ckpt 不存在: {args.ckpt}")

    missing = find_missing_json_holes(train_root)
    print(f"缺少 Auto/Project.json 的孔目录数: {len(missing)}")
    if not missing:
        return

    rng = random.Random(args.seed)
    chosen = missing[:] if len(missing) <= args.topk else rng.sample(missing, args.topk)
    print("将推理这些孔：")
    for d in chosen:
        print(" -", d)

    # 建一个临时 samples_info，让 dataset 复用现有预处理逻辑
    tmp_samples = os.path.join(SCRIPT_DIR, "_tmp_missing_json_samples_info.json")
    build_temp_samples_info(tmp_samples, chosen)

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GridDiffTCN(in_channels=64, out_channels=2, num_channels=(64, 64, 64, 64), kernel_size=3)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model = model.to(device)

    dataset = GridDiffDrillingDataset(
        tmp_samples,
        target_size=(args.img_size, args.img_size),
        roi_size=args.roi_size,
        grid=(8, 8),
        base_dir=None,
        roi_center_yx=(args.roi_cy, args.roi_cx),
        crop_mode=args.crop_mode,
    )

    print("\n推理结果：")
    for i in range(len(dataset)):
        sample = dataset[i]
        out = run_inference(model, sample["data"], sample["layer_list"], device, lock_layers=args.lock_layers)
        print(f"[{i}] penetrated={out['is_penetrated']}  pen_layer={out['penetration_layer_number']}  path={chosen[i]}")



if __name__ == "__main__":
    main()

