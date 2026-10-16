#!/usr/bin/env python3
"""对完整验证集进行固定噪声的 EMA 去噪评价。

衡量的是模型的 epsilon 预测目标，不是生成图视觉质量或通用文本理解。
只有显式运行脚本时才会启动计算。
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
LIMITATIONS = [
    "Denoising MSE is the epsilon-prediction training objective, not a perceptual-quality, image-generation, or semantic-understanding score.",
    "The validation split contains held-out renders of the same character and the same finite condition vocabulary, not unseen subjects or an open-domain benchmark.",
    "Each image uses one fixed uniformly sampled diffusion timestep and one Gaussian noise tensor; this is a reproducible Monte Carlo estimate, not an exact expectation over all timesteps/noise.",
    "Shuffled-factor captions are sampled with replacement from distinct training captions, excluding the original caption. They can change multiple factors simultaneously; the paired difference does not isolate an individual factor.",
    "Positive shuffled-minus-matched MSE indicates that the correct caption helps this fixed-noise denoising test. It does not establish faithful view control, human-perceived quality, or general text understanding.",
    "Training-versus-validation differences are descriptive for this chosen training subset and fixed noise/timesteps. They cannot establish absence of memorization or broad generalization.",
    "FP32, deterministic settings and CPU-generated noise improve reproducibility; bitwise agreement across PyTorch versions, devices or batch sizes is not guaranteed.",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def read_manifest(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError(f"Missing caption at {path}:{line_number}")
        name = row.get("image", row.get("file_name"))
        if not isinstance(name, str) or not name:
            raise ValueError(f"Missing image at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def pixels(rows, manifest, image_size):
    result = []
    for row in rows:
        image_path = Path(row.get("image", row.get("file_name")))
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        with Image.open(image_path) as image:
            if image.size != (image_size, image_size):
                raise ValueError(f"Evaluation requires original {image_size}px images; got {image.size}: {image_path}")
            result.append(np.array(image.convert("RGB"), dtype=np.uint8, copy=True))
    return torch.from_numpy(np.stack(result)).permute(0, 3, 1, 2).contiguous()


def shuffled_conditions(rows, caption_pool, seed):
    generator = random.Random(seed)
    result = []
    for row in rows:
        choices = [caption for caption in caption_pool if caption != row["caption"]]
        if not choices:
            raise ValueError("At least two distinct training captions are needed for the shuffled-condition control")
        result.append(generator.choice(choices))
    return result


def summarize(records):
    if not records:
        return {"image_count": 0, "matched_mse": None, "shuffled_mse": None,
                "paired_shuffled_minus_matched_mean": None, "paired_difference_median": None,
                "paired_difference_std": None, "fraction_matched_lower_mse": None, "zero_predictor_mse": None}
    matched = np.asarray([r["matched_mse"] for r in records], dtype=np.float64)
    shuffled = np.asarray([r["shuffled_mse"] for r in records], dtype=np.float64)
    delta = shuffled - matched
    return {"image_count": len(records), "matched_mse": float(matched.mean()), "shuffled_mse": float(shuffled.mean()),
            "paired_shuffled_minus_matched_mean": float(delta.mean()), "paired_difference_median": float(np.median(delta)),
            "paired_difference_std": float(delta.std(ddof=1)) if len(delta) > 1 else None,
            "fraction_matched_lower_mse": float((delta > 0).mean()),
            "zero_predictor_mse": float(np.mean([r["zero_predictor_mse"] for r in records]))}


@torch.inference_mode()
def evaluate_split(name, rows, original_indices, manifest, model, tokenizer, schedule, t_all, noise_all,
                   caption_pool, caption_to_factors, args):
    started = time.monotonic()
    image_data = pixels(rows, manifest, model.cfg.image_size)
    incorrect = shuffled_conditions(rows, caption_pool, args.seed)
    unknown = {caption: tokenizer.unknown_characters(caption) for caption in set([r["caption"] for r in rows] + incorrect)}
    unknown = {key: value for key, value in unknown.items() if value}
    if unknown:
        raise ValueError(f"Evaluation captions contain unknown training characters: {unknown}")
    matched_tokens = tokenizer.encode([r["caption"] for r in rows])
    shuffled_tokens = tokenizer.encode(incorrect)
    device = torch.device(args.device)
    records = []
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        x = image_data[start:end].to(device).float().div(127.5).sub(1)
        t, noise = t_all[start:end].to(device), noise_all[start:end].to(device)
        noisy = schedule.add_noise(x, noise, t)
        # 配对前向共享同一图片、噪声、时间步和权重，仅替换条件文本。
        # 此处刻意用 FP32，作为诊断计算，不使用采样时的混合精度。
        matched = model(noisy, t, matched_tokens[start:end].to(device))
        shuffled = model(noisy, t, shuffled_tokens[start:end].to(device))
        losses = (matched.float() - noise).square().flatten(1).mean(1).double().cpu().tolist()
        controls = (shuffled.float() - noise).square().flatten(1).mean(1).double().cpu().tolist()
        baselines = noise.square().flatten(1).mean(1).double().cpu().tolist()
        if not all(np.isfinite(losses)) or not all(np.isfinite(controls)):
            raise RuntimeError(f"Non-finite denoising MSE in {name} at image {start}")
        for offset, (loss, control, baseline) in enumerate(zip(losses, controls, baselines)):
            i = start + offset
            row = rows[i]
            replacement = incorrect[i]
            original_factors = {key: row.get(key) for key in ("view", "framing", "background")}
            shuffled_factors = caption_to_factors.get(replacement, {})
            changed = [key for key, value in original_factors.items()
                       if value is not None and key in shuffled_factors and value != shuffled_factors[key]]
            records.append({"dataset_index": original_indices[i], "image": row.get("image", row.get("file_name")),
                            "caption": row["caption"], "shuffled_caption": replacement,
                            "factors": original_factors, "shuffled_factors": shuffled_factors,
                            "changed_factors": changed, "timestep": int(t_all[i]),
                            "matched_mse": loss, "shuffled_mse": control,
                            "paired_shuffled_minus_matched": control - loss, "zero_predictor_mse": baseline})
        print(json.dumps({"event": "evaluation_progress", "split": name, "images": end, "total": len(rows)}), flush=True)
    segments = []
    for i in range(5):
        lower, upper = (i * schedule.steps) // 5, ((i + 1) * schedule.steps) // 5
        selection = [r for r in records if lower <= r["timestep"] < upper]
        segments.append({"t_min_inclusive": lower, "t_max_inclusive": upper - 1, **summarize(selection)})
    return {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "summary": summarize(records), "timestep_segments": segments,
            "elapsed_seconds": round(time.monotonic() - started, 3), "images": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train", type=Path, default=PROJECT / "data/train.jsonl")
    parser.add_argument("--val", type=Path, default=PROJECT / "data/val.jsonl")
    parser.add_argument("--code-dir", type=Path, default=PROJECT / "model_code")
    parser.add_argument("--out", type=Path, default=HERE / "full_validation_report.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    args.train, args.val, args.checkpoint = args.train.resolve(), args.val.resolve(), args.checkpoint.resolve()
    sys.path.insert(0, str(args.code_dir.resolve()))
    from model import checkpoint_model
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a CPU run")
    train_rows, val_rows = read_manifest(args.train), read_manifest(args.val)
    if len(train_rows) < len(val_rows):
        raise ValueError("The training split must have at least as many images as the full validation split")
    selected = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(args.seed))[:len(val_rows)].tolist()
    train_subset = [train_rows[i] for i in selected]
    caption_pool = sorted({row["caption"] for row in train_rows})
    caption_to_factors = {row["caption"]: {key: row[key] for key in ("view", "framing", "background") if key in row} for row in train_rows}
    model, tokenizer, schedule, checkpoint = checkpoint_model(args.checkpoint, args.device, use_ema=True)
    # 各部分使用独立、固定 seed 的随机数生成器，使其可单独复现，
    # 并避免依赖评价 batch 大小或训练子集的抽样顺序。
    count = len(val_rows)
    t_all = torch.randint(schedule.steps, (count,), generator=torch.Generator().manual_seed(args.seed))
    noise_all = torch.randn((count, 3, model.cfg.image_size, model.cfg.image_size), generator=torch.Generator().manual_seed(args.seed))
    validation = evaluate_split("full_validation", val_rows, list(range(count)), args.val, model, tokenizer, schedule,
                                t_all, noise_all, caption_pool, caption_to_factors, args)
    training = evaluate_split("fixed_training_subset", train_subset, selected, args.train, model, tokenizer, schedule,
                              t_all, noise_all, caption_pool, caption_to_factors, args)
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
              "training_step": checkpoint.get("step"), "weights": "EMA", "precision": "FP32", "seed": args.seed,
              "device": str(device), "torch_version": torch.__version__, "batch_size": args.batch_size,
              "resolution": model.cfg.image_size, "diffusion_steps": schedule.steps,
              "metric": "mean((epsilon - epsilon_theta(x_t,t,caption))^2), over all RGB pixels; no CFG",
              "noise_hash_sha256": tensor_hash(noise_all), "timestep_hash_sha256": tensor_hash(t_all),
              "randomness": {"training_subset": "CPU torch.randperm(seed), without replacement; count equals full validation split",
                             "timesteps": "CPU torch.randint(0,T,seed), one per image; shared between train subset and validation by position",
                             "noise": "CPU torch.randn(seed), one Gaussian tensor per image; shared between train subset and validation by position",
                             "shuffled_captions": "Python random.Random(seed), uniform over sorted distinct training captions excluding each original, with replacement",
                             "images": "manifest order for full validation; fixed randperm order for training subset; no augmentation or resize"},
              "caption_pool_size": len(caption_pool), "training_manifest_images": len(train_rows),
              "validation": validation, "training_subset": training,
              "descriptive_train_validation_comparison": {
                  "validation_minus_training_matched_mse": validation["summary"]["matched_mse"] - training["summary"]["matched_mse"],
                  "validation_over_training_matched_mse": validation["summary"]["matched_mse"] / training["summary"]["matched_mse"] if training["summary"]["matched_mse"] else None},
              "limitations": LIMITATIONS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out.resolve()), "training_step": checkpoint.get("step"),
                      "validation": validation["summary"], "training_subset": training["summary"],
                      "limitations": LIMITATIONS}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
