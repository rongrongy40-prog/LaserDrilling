# -*- coding: utf-8 -*-
"""
将 samples_info.json 按比例划分为训练集和测试集。

功能：按 --ratio 与 --seed 随机划分样本列表，写入 samples_info_train.json 与 samples_info_test.json 到 out_dir（默认与 samples_info 同目录）；
      划分后需用训练集跑 precompute_features，再训练；推理时用测试集 JSON。
依赖：无（仅读写 JSON）。
输出：samples_info_train.json、samples_info_test.json。
主要参数：--samples_info, --out_dir, --ratio（训练集比例，默认 0.8）, --seed。
示例：python split_train_test.py --samples_info ../data_drilling/samples_info.json --ratio 0.8 --seed 42
"""

import os
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="划分 samples_info 为 train/test")
    parser.add_argument("--samples_info", type=str, default=None, help="原始 samples_info.json 路径")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录，默认与 samples_info 同目录")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(SCRIPT_DIR, "..", "data_drilling", "samples_info.json")
    args.samples_info = os.path.normpath(args.samples_info)
    if not os.path.isfile(args.samples_info):
        print("未找到:", args.samples_info)
        return

    out_dir = args.out_dir or os.path.dirname(args.samples_info)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.samples_info, "r", encoding="utf-8") as f:
        samples = json.load(f)
    n = len(samples)
    if n == 0:
        print("样本数为 0")
        return

    rng = __import__("random").Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = max(1, int(n * args.ratio))
    n_test = n - n_train
    train_idx = set(indices[:n_train])
    train_list = [samples[i] for i in range(n) if i in train_idx]
    test_list = [samples[i] for i in range(n) if i not in train_idx]

    train_path = os.path.join(out_dir, "samples_info_train.json")
    test_path = os.path.join(out_dir, "samples_info_test.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=2)

    print(f"总数: {n}  训练: {len(train_list)}  测试: {len(test_list)}")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")


if __name__ == "__main__":
    main()
