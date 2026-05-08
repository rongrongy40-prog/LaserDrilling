#!/usr/bin/env python3
"""
将旧版 roi_cache 缓存从 float32/128x128 转换为 uint8/64x64。
转换后体积：约 46GB（367GB → ~46GB，压缩 ~8×）。

用法:
    python convert_roi_cache.py [--workers 4] [--target_size 64]
    # 或直接运行（默认）
    python convert_roi_cache.py

旧文件会被直接覆盖（原地转换，读取→转换→写入同一文件路径）。
磁盘占用峰值约 2× 单文件大小（先读后写），建议 workers=2。
"""
import argparse
import glob
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm


def convert_file(path: str, target_size: int) -> dict:
    """读取一个 .pt，转换为 uint8/64x64 并覆盖保存。"""
    d = torch.load(path, map_location="cpu", weights_only=False)
    frames = d["frames"]  # (T, F, 3, H, W) float32

    # Convert float32 → uint8, downsample 128→64
    if frames.dtype == torch.uint8:
        is_already_new = d.get("_uint8", False)
        if is_already_new and d.get("_roi_size", 64) == target_size:
            return {"status": "skipped_already_new", "path": path}

        # Already uint8 but wrong size: just resize in-place
        stored_size = d.get("_roi_size", 64)
        frames = frames.float() / 255.0
        T, F_ch, C, H, W = frames.shape
        flat = frames.permute(0, 1, 3, 4, 2).reshape(T * F_ch, C, H, W)
        flat = F.interpolate(flat, size=(target_size, target_size),
                             mode="bilinear", align_corners=False)
        frames = (flat.reshape(T, F_ch, 3, target_size, target_size) * 255).round().to(torch.uint8)
    else:
        T, F_ch, C, H, W = frames.shape
        frames = frames.float()  # ensure float
        if H != target_size or W != target_size:
            flat = frames.permute(0, 1, 3, 4, 2).reshape(T * F_ch, C, H, W)
            flat = F.interpolate(flat, size=(target_size, target_size),
                                 mode="bilinear", align_corners=False)
            frames = flat.reshape(T, F_ch, 3, target_size, target_size)
        frames = (frames.clamp(0, 1) * 255).round().to(torch.uint8)

    d["frames"] = frames
    d["_uint8"] = True
    d["_roi_size"] = target_size

    orig_bytes = os.path.getsize(path) if os.path.exists(path) else 0  # before replace
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)
    new_bytes = os.path.getsize(path)
    return {"status": "converted", "path": path,
            "saved_mb": (orig_bytes - new_bytes) / 1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data_drilling/roi_cache")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发转换的进程数（默认4，建议2以避免磁盘争抢）")
    parser.add_argument("--target_size", type=int, default=64,
                        help="目标分辨率（默认64）")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if not os.path.isdir(cache_dir):
        print(f"Error: {cache_dir} not found")
        return

    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not files:
        print("No .pt files found.")
        return

    print(f"Found {len(files)} cache files in {cache_dir}")
    print(f"Target size: {args.target_size}×{args.target_size}, dtype: uint8")
    print(f"Workers: {args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    converted = skipped = errors = 0
    total_saved = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_file, f, args.target_size): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(files), desc="Converting"):
            try:
                res = fut.result()
                if res["status"] == "converted":
                    converted += 1
                    total_saved += res.get("saved_mb", 0)
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                print(f"\nError: {e}")

    print(f"\nDone: {converted} converted, {skipped} skipped, {errors} errors")
    print(f"Total space saved: {total_saved/1e3:.1f} GB")


if __name__ == "__main__":
    main()
