#!/usr/bin/env bash
# ============================================================
# 从 master_annotations.json 中删除 category 字段
# ============================================================
set -euo pipefail

SRC="${1:-data_drilling/master_annotations.json}"

source ~/miniconda3/bin/activate my_py310

python3 -c "
import json

src = '$SRC'

with open(src, 'r', encoding='utf-8') as f:
    raw = json.load(f)

if isinstance(raw, list):
    for item in raw:
        item.pop('category', None)
elif isinstance(raw, dict) and 'Categories' in raw:
    for item in raw['Categories']:
        item.pop('category', None)

with open(src, 'w', encoding='utf-8') as f:
    json.dump(raw, f, ensure_ascii=False, indent=2)

print(f\"Done: removed 'category' field from {src}\")
"
