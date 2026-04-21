# -*- coding: utf-8 -*-
"""
Batch feature extraction for masked_v2.
Supports extracting features from:
  1. Pre-trained DINOv3 (default): frozen, generic features
  2. Trained masked_v2 checkpoint: Stage 1 fine-tuned encoder features

Using Stage 1 fine-tuned encoder features for Stage 2 training/inference
is the key improvement of the v2 pipeline.

Performance: each sample ~1400 frames, processed in chunks via dinov3_chunk_size
to maximize GPU utilization.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from grid_diff_tcn.hier.frame_layer.dinov3_features import DinoV3FeatureExtractor
from grid_diff_tcn.masked_v2.model import MaskedPixelModel, load_masked_model
from grid_diff_tcn.masked_v2.dataset import MaskedDrillingDataset, CropCacheDataset, collate_masked_batch


def build_feature_extractor(
    dinov3_model: str,
    dinov3_roi_size: int,
    checkpoint_path: str | None,
    dinov3_feat_dim: int,
) -> tuple[torch.nn.Module, bool]:
    """
    Build a feature extractor for batch inference.

    Returns:
        (model, is_masked_pixel_model):
            - If checkpoint provided: MaskedPixelModel (stage 2)
            - If no checkpoint: DinoV3FeatureExtractor directly
    """
    if checkpoint_path:
        model = load_masked_model(
            checkpoint_path,
            stage=2,
            dinov3_model=dinov3_model,
            dinov3_feat_dim=dinov3_feat_dim,
            dinov3_roi_size=dinov3_roi_size,
            use_cached_features=False,
        )
        return model, True
    else:
        extractor = DinoV3FeatureExtractor(
            model_name=dinov3_model,
            pretrained=True,
            image_size=dinov3_roi_size,
        )
        return extractor, False


def _extract_single(
    extractor: torch.nn.Module,
    is_model: bool,
    frame_data: torch.Tensor,
    dinov3_roi_size: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Run DINOv3 forward on a (B, T, F, 3, H, W) tensor.
    Returns (B, T, F, feat_dim) features.

    Key optimization: flatten the full (B*T*F,) batch, process in chunks,
    then reshape back. This amortizes Python loop overhead across the
    full batch rather than per-sample.

    Always calls dinov3_extractor directly so that chunk_size is fully
    controlled from here — not from the model's internal self.dinov3_chunk_size.
    """
    B, T, F, C, H, W = frame_data.shape
    flat = frame_data.reshape(B * T * F, C, H, W)

    all_feats = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            batch_chunk = flat[start:end]
            # Always go through dinov3_extractor directly:
            # - MaskedPixelModel path: extractor.dinov3_extractor
            # - No-checkpoint path: extractor itself IS the DinoV3FeatureExtractor
            dinov3 = extractor.dinov3_extractor if is_model else extractor
            feat = dinov3(batch_chunk)
            all_feats.append(feat.float())

    feats_flat = torch.cat(all_feats, dim=0)              # (B*T*F, feat_dim)
    feats = feats_flat.reshape(B, T, F, -1)              # (B, T, F, feat_dim)
    return feats


def extract_features(
    samples_info: str,
    out_dir: str,
    dinov3_model: str,
    dinov3_feat_dim: int,
    dinov3_roi_size: int,
    dinov3_chunk_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    checkpoint_path: str | None,
    crop_cache_dir: str | None = None,
):
    if crop_cache_dir:
        print(f"[extract] Using CropCacheDataset (cache={crop_cache_dir})")
        dataset = CropCacheDataset(
            samples_info_path=samples_info,
            cache_dir=crop_cache_dir,
            roi_size=dinov3_roi_size,
        )
    else:
        print("[extract] Using MaskedDrillingDataset (online ROI extraction)")
        dataset = MaskedDrillingDataset(
            samples_info_path=samples_info,
            roi_size=dinov3_roi_size,
        )

    # batch_size=1 means 1 sample per DataLoader step, but each sample has ~1400 frames.
    # The _extract_single function flattens the full (1, T, F) per sample and processes
    # it in dinov3_chunk_size chunks. This is already optimized per-sample.
    # Increase batch_size if you have more GPU memory to further amortize overhead.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_masked_batch,
        pin_memory=device.type == "cuda",
    )

    extractor, is_model = build_feature_extractor(
        dinov3_model=dinov3_model,
        dinov3_roi_size=dinov3_roi_size,
        checkpoint_path=checkpoint_path,
        dinov3_feat_dim=dinov3_feat_dim,
    )
    extractor = extractor.to(device)
    extractor.eval()

    os.makedirs(out_dir, exist_ok=True)

    # Build set of already-extracted files for fast skip
    existing = set()
    if os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            if f.endswith(".pt"):
                existing.add(f)

    done = 0
    skipped = 0

    # torch.compile + multi-step forward loop causes CUDA Graphs tensor-overwrite issues
    # with this codebase's DinoV3 forward. The chunk_size optimization already gives
    # the main speedup, so compile is disabled for safety.
    use_compile = False
    if use_compile:
        print("[extract] Applying torch.compile to DinoV3FeatureExtractor...")
        extractor = torch.compile(extractor, mode="reduce-overhead")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="提取特征")):
            frame_data = batch["frame_data"].to(device)   # (B, T, F, 3, H, W)
            sample_paths = batch["sample_paths"]

            # Warmup compile on first batch (only used when compile is enabled)
            if use_compile and batch_idx == 0:
                _ = _extract_single(
                    extractor, is_model, frame_data[:1],
                    dinov3_roi_size, dinov3_chunk_size,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            feats = _extract_single(
                extractor, is_model, frame_data,
                dinov3_roi_size, dinov3_chunk_size,
            )
            feats = feats.cpu()

            for bi, sp in enumerate(sample_paths):
                key = sp.replace(os.sep, "__").replace("/", "__").replace(".", "_") + ".pt"
                if key in existing:
                    skipped += 1
                    continue
                torch.save(
                    {"features": feats[bi].float(), "sample_path": sp},
                    os.path.join(out_dir, key),
                )
                existing.add(key)
                done += 1

    print(f"完成: {done} 个样本 (跳过 {skipped} 个已有文件)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: path to trained MaskedPixelModel checkpoint. "
                             "If omitted, uses the pretrained DINOv3 encoder directly.")
    parser.add_argument("--samples_info", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted features (.pt per sample).")
    parser.add_argument("--dinov3_model", type=str, default="vit_small")
    parser.add_argument("--dinov3_feat_dim", type=int, default=384)
    parser.add_argument("--dinov3_roi_size", type=int, default=224)
    parser.add_argument("--dinov3_chunk_size", type=int, default=256,
                        help="Frames per DINOv3 forward pass. "
                             "Higher = fewer forward passes, more GPU utilization. "
                             "Default 256; try 512 if GPU memory allows. "
                             "Each sample has ~1400 frames.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="DataLoader batch size (number of samples per step). "
                             "Default 2. Increase if GPU memory allows.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_cache_dir", type=str, default=None,
                        help="Directory with pre-cropped ROI .pt files (from pre_crop.py). "
                             "If set, reads from CropCacheDataset instead of MaskedDrillingDataset. "
                             "Much faster when ROI crops are already cached.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_features(
        samples_info=args.samples_info,
        out_dir=args.output_dir,
        dinov3_model=args.dinov3_model,
        dinov3_feat_dim=args.dinov3_feat_dim,
        dinov3_roi_size=args.dinov3_roi_size,
        dinov3_chunk_size=args.dinov3_chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        checkpoint_path=args.checkpoint,
        crop_cache_dir=args.crop_cache_dir,
    )


if __name__ == "__main__":
    main()
