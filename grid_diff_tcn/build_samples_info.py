# -*- coding: utf-8 -*-
"""
从 data_drilling/train 扫描各孔目录，生成 samples_info.json。

约定：jpg 文件名去掉扩展名后，最后一个「_」后为层号（与 dataset.parse_layer_from_filename 一致）。
穿透信息：读取 Project.json 的 Categories 或 Auto.json 列表，取首条 Classification==1 的 ImagePath，
         从文件名解析 penetration_layer。

示例：
  python3 build_samples_info.py --train_root ./train --output ./samples_info.json
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_DIFF = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "grid_diff_tcn"))
if _GRID_DIFF not in sys.path:
    sys.path.insert(0, _GRID_DIFF)

from dataset import parse_layer_from_filename  # noqa: E402


def _load_categories(hole_dir):
    """返回标注条目列表，或 None。"""
    pj = os.path.join(hole_dir, "Project.json")
    aj = os.path.join(hole_dir, "Auto.json")
    for path in (pj, aj):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "Categories" in data:
            return data["Categories"]
        if isinstance(data, list):
            return data
    return None


def _first_penetration_layer_from_categories(categories):
    if not categories:
        return 0, -1
    for item in categories:
        if int(item.get("Classification", 0)) != 1:
            continue
        raw_path = item.get("ImagePath") or ""
        base = os.path.basename(raw_path.replace("\\", os.sep))
        layer = parse_layer_from_filename(base)
        if layer is None:
            continue
        return 1, layer
    return 0, -1


def collect_hole_dirs(train_root):
    train_root = os.path.abspath(train_root)
    holes = []
    for dirpath, _dirnames, filenames in os.walk(train_root):
        if "Project.json" not in filenames and "Auto.json" not in filenames:
            continue
        holes.append(dirpath)
    return sorted(holes)


def build_entries(train_root):
    train_root = os.path.abspath(train_root)
    entries = []
    for hole_dir in collect_hole_dirs(train_root):
        rel = os.path.relpath(hole_dir, train_root)
        parts = rel.split(os.sep)
        category = parts[0] if parts else "unknown"

        cats = _load_categories(hole_dir)
        is_pen, pen_layer = _first_penetration_layer_from_categories(cats)

        entries.append({
            "category": category,
            "sample_path": hole_dir,
            "is_penetrated": is_pen,
            "penetration_layer": pen_layer,
        })
    entries.sort(key=lambda x: x["sample_path"])
    return entries


def main():
    ap = argparse.ArgumentParser(description="从 train 目录生成 samples_info.json")
    ap.add_argument(
        "--train_root",
        type=str,
        default=os.path.join(SCRIPT_DIR, "train"),
        help="训练数据根目录（内含 category/.../孔文件夹）",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "samples_info.json"),
        help="输出 JSON 路径",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.train_root):
        print("train_root 不存在:", args.train_root)
        return 1

    entries = build_entries(args.train_root)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"孔数: {len(entries)}  写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
