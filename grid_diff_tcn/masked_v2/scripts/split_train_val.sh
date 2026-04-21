#!/usr/bin/env bash
# ============================================================
# 从 samples_info_train.json 中划分出验证集
# ============================================================
set -euo pipefail

SRC="${1:-data_drilling/samples_info_train.json}"
OUT_DIR="${2:-data_drilling}"
VAL_RATIO="${3:-0.15}"
SEED="${4:-42}"

source ~/miniconda3/bin/activate my_py310

python3 - <<'EOF'
import json, random, sys, os

src = sys.argv[1]
out_dir = sys.argv[2]
val_ratio = float(sys.argv[3])
seed = int(sys.argv[4])

with open(src, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, dict) and "Categories" in raw:
    items = raw["Categories"]
else:
    items = raw

random.seed(seed)
random.shuffle(items)

n_val = max(1, int(len(items) * val_ratio))
val_items = items[:n_val]
train_items = items[n_val:]

os.makedirs(out_dir, exist_ok=True)

def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, dict):
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, ensure_ascii=False, indent=2)

out_train = os.path.join(out_dir, "samples_info_train_split.json")
out_val = os.path.join(out_dir, "samples_info_val.json")

save(out_train, train_items)
save(out_val, val_items)

print(f"train: {len(train_items)}  val: {len(val_items)}  ({val_ratio*100:.0f}%)")
print(f"  -> {out_train}")
print(f"  -> {out_val}")
EOF
