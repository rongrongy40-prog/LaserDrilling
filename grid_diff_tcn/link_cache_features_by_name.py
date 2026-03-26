# -*- coding: utf-8 -*-
"""
为预计算特征文件建立“样本名.pt”的硬链接或副本。

功能：根据 samples_info 中的 sample_path，在 cache_dir 内为已有 {idx}.pt 创建以样本文件夹名命名的链接（或复制），
      保留原 idx.pt 不变，便于按样本名访问特征且与 precompute_features --by_name 输出一致。
依赖：samples_info 列表与 cache_dir 下已存在的 .pt 文件（通常为 0.pt, 1.pt, …）。
输出：cache_dir 下新增 样本名.pt（硬链接或复制）。
主要参数：--samples_info, --cache_dir, --mode（link|copy）。
示例：python link_cache_features_by_name.py --samples_info ../data_drilling/samples_info_train.json --cache_dir ./cache_features_train
"""

import os
import json
import argparse
import shutil


def safe_name(name: str) -> str:
    return name.replace(os.sep, "_").replace("/", "_").strip()


def main():
    ap = argparse.ArgumentParser(description="为 idx.pt 创建 样本名.pt 链接/副本")
    ap.add_argument("--samples_info", type=str, required=True)
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="link", choices=["link", "copy"], help="link=硬链接(省空间)，copy=复制")
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

    created = 0
    skipped = 0
    collisions = 0

    for idx, s in enumerate(samples):
        src = os.path.join(cache_dir, f"{idx}.pt")
        if not os.path.isfile(src):
            skipped += 1
            continue
        sp = str(s.get("sample_path", ""))
        base = safe_name(os.path.basename(sp)) or f"sample_{idx}"
        dst = os.path.join(cache_dir, f"{base}.pt")
        if os.path.abspath(dst) == os.path.abspath(src):
            skipped += 1
            continue
        if os.path.exists(dst):
            # resolve collision
            k = 2
            while True:
                dst2 = os.path.join(cache_dir, f"{base}__{k}.pt")
                if not os.path.exists(dst2):
                    dst = dst2
                    collisions += 1
                    break
                k += 1
        try:
            if args.mode == "link":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            created += 1
        except OSError:
            # 有些文件系统不支持硬链接，自动降级为 copy
            shutil.copy2(src, dst)
            created += 1

    print(f"cache_dir: {cache_dir}")
    print(f"created: {created}  skipped(no idx.pt yet): {skipped}  collisions: {collisions}")


if __name__ == "__main__":
    main()

