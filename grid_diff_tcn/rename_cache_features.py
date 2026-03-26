# -*- coding: utf-8 -*-
"""
将预计算特征目录中的 {idx}.pt 重命名为样本文件夹名.pt。

功能：按 samples_info 顺序，将 cache_dir 下的 0.pt, 1.pt, … 重命名为 basename(sample_path).pt；
      若重名则自动追加 __2, __3, …。可选 --dry_run 仅打印不执行。
依赖：samples_info JSON 与 cache_dir 下对应索引的 .pt 文件。
输出：原地重命名 cache_dir 内文件。
主要参数：--samples_info, --cache_dir, --dry_run。
示例：python rename_cache_features.py --samples_info ../data_drilling/samples_info_train.json --cache_dir ./cache_features_train
"""

import os
import json
import argparse


def safe_name(name: str) -> str:
    # 只做最小清洗：去掉路径分隔符（理论上 basename 不含，但防御一下）
    return name.replace(os.sep, "_").replace("/", "_").strip()


def main():
    ap = argparse.ArgumentParser(description="重命名 cache_features 的 pt 文件为样本名")
    ap.add_argument("--samples_info", type=str, required=True, help="samples_info_train.json 路径")
    ap.add_argument("--cache_dir", type=str, required=True, help="cache_features_train 目录")
    ap.add_argument("--dry_run", action="store_true", help="只打印不改名")
    args = ap.parse_args()

    samples_info = os.path.abspath(os.path.normpath(args.samples_info))
    cache_dir = os.path.abspath(os.path.normpath(args.cache_dir))
    if not os.path.isfile(samples_info):
        raise SystemExit(f"samples_info 不存在: {samples_info}")
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"cache_dir 不存在: {cache_dir}")

    with open(samples_info, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise SystemExit("samples_info 不是 list")

    # 统计已有 idx.pt
    renamed = 0
    skipped = 0
    collisions = 0

    used = set()
    for idx, s in enumerate(samples):
        src = os.path.join(cache_dir, f"{idx}.pt")
        if not os.path.isfile(src):
            skipped += 1
            continue
        sp = str(s.get("sample_path", ""))
        base = safe_name(os.path.basename(sp)) or f"sample_{idx}"
        dst = os.path.join(cache_dir, f"{base}.pt")
        if dst in used or os.path.isfile(dst):
            # resolve collision
            k = 2
            while True:
                dst2 = os.path.join(cache_dir, f"{base}__{k}.pt")
                if (dst2 not in used) and (not os.path.isfile(dst2)):
                    dst = dst2
                    collisions += 1
                    break
                k += 1
        used.add(dst)
        if args.dry_run:
            print(f"mv {os.path.basename(src)} -> {os.path.basename(dst)}")
        else:
            os.rename(src, dst)
        renamed += 1

    print(f"cache_dir: {cache_dir}")
    print(f"renamed: {renamed}  skipped(no idx.pt): {skipped}  collisions: {collisions}")


if __name__ == "__main__":
    main()

