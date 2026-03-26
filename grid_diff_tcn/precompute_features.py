# -*- coding: utf-8 -*-
"""
预计算每个孔的 [Seq_Len, 64] 网格差分特征并保存到磁盘。

功能：按 samples_info 遍历每孔，读图并做层内融合、帧间差、8×8 网格池化，得到 [Seq_Len, 64] 张量；
      每孔保存为一份 .pt 文件（含 data、layer_list），训练时指定 --precomputed_dir 可跳过读图，大幅提速。
依赖：dataset.GridDiffDrillingDataset；samples_info.json 与对应图片目录。
输出：out_dir 下每孔一个 .pt 文件；--by_name 时按样本文件夹名命名（如 xxx.pt），否则按索引 0.pt, 1.pt, …。
主要参数：--samples_info, --out_dir, --by_name, --img_size, --roi_size。
示例：python precompute_features.py --samples_info ../data_drilling/samples_info_train.json --out_dir ./cache_features_train --by_name
"""

import os
import sys
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import GridDiffDrillingDataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def main():
    parser = argparse.ArgumentParser(description="预计算 Grid-Diff 特征 [Seq_Len, 64] 到磁盘")
    parser.add_argument("--samples_info", type=str, default=None, help="samples_info.json 路径")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="cache_features", help="输出目录，每孔一个 {idx}.pt")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--roi_size", type=int, default=None)
    parser.add_argument("--roi_cy", type=float, default=0.5)
    parser.add_argument("--roi_cx", type=float, default=0.5)
    parser.add_argument("--crop_mode", type=str, default="center", choices=["center", "roi"])
    parser.add_argument("--load_workers", type=int, default=6)
    parser.add_argument("--by_name", action="store_true", help="按样本文件夹名保存为 xxx.pt，否则按 0.pt 1.pt …")
    args = parser.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(SCRIPT_DIR, "..", "data_drilling", "samples_info.json")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if not os.path.isfile(args.samples_info):
        print("未找到 samples_info.json:", args.samples_info)
        return

    img_size = max(64, args.img_size)
    roi_size = args.roi_size if args.roi_size is not None else min(96, img_size)
    roi_size = min(roi_size, img_size)
    if roi_size % 8 != 0:
        roi_size = (roi_size // 8) * 8
    if roi_size < 8:
        roi_size = 8
    roi_center_yx = (args.roi_cy, args.roi_cx)

    dataset = GridDiffDrillingDataset(
        args.samples_info,
        target_size=(img_size, img_size),
        roi_size=roi_size,
        grid=(8, 8),
        base_dir=args.base_dir,
        load_workers=args.load_workers,
        roi_center_yx=roi_center_yx,
        crop_mode=args.crop_mode,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    n = len(dataset)

    def safe_basename(path):
        name = os.path.basename(path or "").strip()
        return name.replace(os.sep, "_").replace("/", "_") or "unknown"

    if args.by_name:
        used = set()
        def unique_name(idx):
            path = dataset.samples[idx].get("sample_path", "")
            base = safe_basename(path) or f"sample_{idx}"
            name = base
            k = 2
            while name in used:
                name = f"{base}__{k}"
                k += 1
            used.add(name)
            return name + ".pt"
        name_for_idx = [unique_name(i) for i in range(n)]
        print(f"预计算 {n} 个孔，保存到 {args.out_dir}（按样本名），参数 img={img_size} roi={roi_size}")
    else:
        name_for_idx = [f"{i}.pt" for i in range(n)]
        print(f"预计算 {n} 个孔，保存到 {args.out_dir}，参数 img={img_size} roi={roi_size}")

    for idx in tqdm(range(n), desc="预计算"):
        out_path = os.path.join(args.out_dir, name_for_idx[idx])
        if os.path.isfile(out_path):
            continue
        try:
            raw = dataset[idx]
            data = raw["data"]
            layer_list = raw["layer_list"]
            sample_path = raw.get("sample_path") or dataset.samples[idx].get("sample_path", "")
            torch.save(
                {"data": data, "layer_list": layer_list, "sample_path": sample_path},
                out_path,
            )
        except Exception as e:
            print(f"  [{idx}] 失败: {e}")

    if args.by_name:
        print(f"完成（按样本名）。训练时使用: --precomputed_dir {os.path.abspath(args.out_dir)} --num_workers 4")
    else:
        print(f"完成。训练时使用: --precomputed_dir {os.path.abspath(args.out_dir)} --num_workers 4")


if __name__ == "__main__":
    main()
