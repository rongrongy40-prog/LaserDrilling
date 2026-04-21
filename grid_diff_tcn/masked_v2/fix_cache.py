#!/usr/bin/env python3
"""快速修复缓存：遍历旧缓存文件，用正确的层号重新生成 layer_list"""
import os, json, glob, torch
from collections import defaultdict
from tqdm import tqdm

def fix_cache_file(cache_path):
    """重新加载旧的 .pt，找出正确的层号"""
    d = torch.load(cache_path, weights_only=False, map_location='cpu')
    frames = d['frames']  # (T, F, 3, H, W)
    old_layers = d.get('layers', [])
    
    # 旧缓存中 layers 只有一层 (如 [9])，这是错的
    # 真正的层号需要从 sample_path 重新解析
    sample_path = d.get('sample_path', '')
    if not sample_path or not os.path.isdir(sample_path):
        return None
    
    # 重新扫描目录获取正确的层号
    by_layer = defaultdict(list)
    for p in glob.glob(os.path.join(sample_path, "*.jpg")):
        fn = os.path.basename(p)
        # 用正确的解析方式
        parts = fn.replace('.jpg', '').split('_')
        if len(parts) >= 2:
            try:
                layer = int(parts[-1])
                frame = int(parts[-2])
                by_layer[layer].append((frame, p))
            except:
                pass
    
    layer_list = sorted(by_layer.keys())
    if not layer_list:
        return None
    
    # 更新 layers
    d['layers'] = layer_list
    return d

# 遍历所有 .pt
cache_dir = 'data_drilling/roi_cache'
pts = sorted(glob.glob(os.path.join(cache_dir, '*.pt')))
print(f"Found {len(pts)} cache files")

for p in tqdm(pts, desc="Fixing cache"):
    # 跳过临时文件
    if p.endswith('.tmp'):
        continue
    
    # 检查 layers 是否已经是正确的 (T>1)
    try:
        d = torch.load(p, weights_only=False, map_location='cpu')
        if isinstance(d.get('layers'), list) and len(d.get('layers', [])) > 1:
            continue  # 已经是正确的，跳过
    except:
        continue
    
    fixed = fix_cache_file(p)
    if fixed:
        torch.save(fixed, p)
        print(f"Fixed: {os.path.basename(p)} -> layers: {fixed['layers'][:5]}...")

print("Done!")