#!/usr/bin/env bash
# ============================================================
# 从 master_annotations.json 中划分出训练集、验证集、测试集
# ============================================================
set -euo pipefail

SRC="${1:-data_drilling/master_annotations.json}"
OUT_DIR="${2:-data_drilling}"
VAL_RATIO="${3:-0.15}"
TEST_RATIO="${4:-0.15}"
SEED="${5:-42}"

source ~/miniconda3/bin/activate my_py310

python3 -c "
import json, random, os

src = '$SRC'
out_dir = '$OUT_DIR'
val_ratio = float('$VAL_RATIO')
test_ratio = float('$TEST_RATIO')
seed = int('$SEED')

with open(src, 'r', encoding='utf-8') as f:
    raw = json.load(f)

if isinstance(raw, dict) and 'Categories' in raw:
    items = raw['Categories']
else:
    items = raw

random.seed(seed)
random.shuffle(items)

n = len(items)
n_test = max(1, int(n * test_ratio))
n_val = max(1, int(n * val_ratio))

test_items = items[:n_test]
val_items = items[n_test:n_test + n_val]
train_items = items[n_test + n_val:]

os.makedirs(out_dir, exist_ok=True)

def save(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

out_train = os.path.join(out_dir, 'samples_info_train_split.json')
out_val = os.path.join(out_dir, 'samples_info_val.json')
out_test = os.path.join(out_dir, 'samples_info_test_split.json')

save(out_train, train_items)
save(out_val, val_items)
save(out_test, test_items)

print(f'train: {len(train_items)}  val: {len(val_items)}  test: {len(test_items)}')
print(f'  -> {out_train}')
print(f'  -> {out_val}')
print(f'  -> {out_test}')
"
