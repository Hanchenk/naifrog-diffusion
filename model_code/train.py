#!/usr/bin/env python3
"""从随机权重开始训练 64×64 文本条件 DDPM；命令行示例见仓库 README。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from dataclasses import asdict
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from model import ModelConfig, CharacterTokenizer, ConditionalUNet, DiffusionSchedule, sample_ddim


def read_dataset(path, image_size):
    path = Path(path).resolve()
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    images = []
    for row in rows:
        image_path = Path(row.get("image", row.get("file_name", "")))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            images.append(np.array(image, dtype=np.uint8, copy=True))
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError(f"Missing nonempty caption: {image_path}")
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    return rows, images


def save_grid(images, path, columns=4, scale=3):
    array = images.detach().float().clamp(-1, 1).add(1).mul(127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
    n, h, w, _ = array.shape
    columns = min(columns, n)
    canvas = Image.new("RGB", (columns*w, math.ceil(n/columns)*h), "white")
    for i, a in enumerate(array):
        canvas.paste(Image.fromarray(a), ((i%columns)*w, (i//columns)*h))
    canvas.resize((canvas.width*scale, canvas.height*scale), Image.Resampling.NEAREST).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL; image paths relative to its directory")
    parser.add_argument("--val-data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--text-width", type=int, default=128)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--condition-dropout", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=1000, help="0 disables periodic image sampling")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-guidance", type=float, default=2.5)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--resume", help="Checkpoint path; --steps is the total desired step count")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.base_channels % 16 or args.text_width % 4:
        raise ValueError("base-channels must be divisible by 16; text-width by 4")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu for a smoke run")
    out = Path(args.out).resolve()
    for folder in [out, out/"checkpoints", out/"samples"]:
        folder.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    cfg = ModelConfig(**checkpoint["model_config"]) if checkpoint else ModelConfig(base_channels=args.base_channels,
        text_width=args.text_width, text_layers=args.text_layers, max_length=args.max_length, diffusion_steps=args.diffusion_steps)
    rows, images = read_dataset(args.data, cfg.image_size)
    captions = [r["caption"] for r in rows]
    tokenizer = CharacterTokenizer(checkpoint["vocab"], cfg.max_length) if checkpoint else CharacterTokenizer.build(captions, cfg.max_length)
    tokens = tokenizer.encode(captions)
    cache_device = device if args.cache == "gpu" else torch.device("cpu")
    images, tokens = images.to(cache_device), tokens.to(cache_device)
    if cache_device.type == "cpu" and device.type == "cuda":
        images, tokens = images.pin_memory(), tokens.pin_memory()
    val = None
    if args.val_data:
        val_rows, val_images = read_dataset(args.val_data, cfg.image_size)
        val = (val_images.to(cache_device), tokenizer.encode([r["caption"] for r in val_rows]).to(cache_device))
    model = ConditionalUNet(cfg, len(tokenizer.vocab)).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    schedule = DiffusionSchedule(cfg.diffusion_steps, device)
    start_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]
        torch.set_rng_state(checkpoint["rng_cpu"])
        if device.type == "cuda" and checkpoint.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(checkpoint["rng_cuda"], device)
    parameters = sum(p.numel() for p in model.parameters())
    metadata = {"args": vars(args), "model_config": asdict(cfg), "vocab": tokenizer.vocab,
        "parameters": parameters, "train_images": len(rows), "validation_images": 0 if val is None else len(val[0]),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "architecture": "random-init pixel-space text-conditional UNet; learned character transformer; low-resolution cross-attention",
        "objective": "E[||epsilon - epsilon_theta(x_t,t,text)||^2]; 15% null conditioning by default",
        "schedule": "1000-step cosine alpha-bar, beta cap .999; EMA; DDIM eta=0 for preview",
        "torch_version": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
    (out/"config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"vocab.json").write_text(json.dumps(tokenizer.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    prompts = [
        "奶蛙捧腹大笑，正面视角，全身，蓝色背景",
        "奶蛙捧腹大笑，左前方视角，全身，粉色背景",
        "奶蛙捧腹大笑，右前方视角，全身，绿色背景",
        "奶蛙捧腹大笑，正面视角，全身，白色背景",
        "奶蛙捧腹大笑，正面视角，近景，粉色背景",
        "奶蛙捧腹大笑，左前方视角，近景，蓝色背景",
        "奶蛙捧腹大笑，右前方视角，近景，白色背景",
        "奶蛙捧腹大笑，正面视角，近景，绿色背景"]
    # 若换成其他数据集，用其真实 captions 采样，避免固定提示词含未知字符。
    if any(tokenizer.unknown_characters(p) for p in prompts):
        prompts = list(dict.fromkeys(captions))[:8]
    (out/"samples"/"prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_tokens = tokenizer.encode(prompts, device)
    null_tokens = tokenizer.encode([""]*args.batch_size, device)
    model_params, ema_params = list(model.parameters()), list(ema.parameters())
    print(json.dumps({"event":"initialized", "parameters":parameters, "images":len(rows), "start_step":start_step,
          "steps":args.steps, "device":str(device), "cache":str(cache_device)}, ensure_ascii=False), flush=True)

    def save(step):
        payload = {"step":step, "model":model.state_dict(), "ema":ema.state_dict(),
            "optimizer":optimizer.state_dict(), "scaler":scaler.state_dict(), "model_config":asdict(cfg),
            "vocab":tokenizer.vocab, "train_args":vars(args), "rng_cpu":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        latest = out/"checkpoints"/"latest.pt"
        temporary = latest.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, latest)
        dated = out/"checkpoints"/f"step-{step:07d}.pt"
        if dated.exists():
            dated.unlink()
        os.link(latest, dated)
        old = sorted((out/"checkpoints").glob("step-*.pt"))
        for path in old[:-max(1, args.keep_last)]:
            path.unlink()

    @torch.inference_mode()
    def evaluate():
        if val is None:
            return None
        # 每次验证使用相同的留出图片、高斯噪声与时间步，便于比较训练进展。
        generator = torch.Generator(device=device).manual_seed(args.seed+98765)
        total, count = 0.0, 0
        for k in range(args.val_batches):
            start = (k*args.batch_size) % len(val[0])
            ids = torch.arange(start, min(start+args.batch_size, len(val[0])), device=cache_device)
            x = val[0][ids].to(device).float().div(127.5).sub(1)
            tok = val[1][ids].to(device)
            t = torch.randint(schedule.steps, (len(x),), generator=generator, device=device)
            noise = torch.randn(x.shape, generator=generator, device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                pred = ema(schedule.add_noise(x, noise, t), t, tok)
            total += F.mse_loss(pred.float(), noise, reduction="sum").item()
            count += noise.numel()
        return total/count

    metrics = open(out/"metrics.jsonl", "a", encoding="utf-8", buffering=1)
    started = time.perf_counter()
    window_started = started
    losses = []
    for step in range(start_step+1, args.steps+1):
        model.train()
        ids = torch.randint(len(images), (args.batch_size,), device=cache_device)
        x = images[ids].to(device, non_blocking=True).float().div_(127.5).sub_(1)
        tok = tokens[ids].to(device, non_blocking=True)
        drop = torch.rand((len(x),), device=device) < args.condition_dropout
        tok = torch.where(drop[:, None], null_tokens, tok)
        t = torch.randint(schedule.steps, (len(x),), device=device)
        noise = torch.randn_like(x)
        # 真正训练步骤：采样真实 epsilon，加到图片，再监督网络预测这份 epsilon。
        noisy = schedule.add_noise(x, noise, t)
        warmup = min(1.0, step/max(1, args.warmup))
        progress = min(1.0, max(0.0, (step-args.warmup)/max(1, args.steps-args.warmup)))
        lr = args.lr * warmup * (0.1 + 0.9*0.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            prediction = model(noisy, t, tok)
            loss = F.mse_loss(prediction.float(), noise)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        # 反向传播更新图像 U-Net 与文本编码器的所有可训练参数。
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = scaler.get_scale() >= previous_scale
        if optimizer_updated:
            # 训练早期降低 EMA decay，让预览尽快摆脱随机初始化；之后逐渐接近设定值。
            decay = min(args.ema_decay, (1+step)/(10+step))
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, model_params, 1-decay)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            record = {"step":step, "train_mse":sum(losses)/len(losses), "lr":lr,
                "grad_norm":float(grad_norm), "steps_per_second":len(losses)/(now-window_started),
                "elapsed_seconds":now-started, "optimizer_updated":optimizer_updated,
                "amp_scale":scaler.get_scale()}
            if device.type == "cuda":
                record["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated(device)/1024**3
            metrics.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
            losses, window_started = [], now
        if args.eval_every and val is not None and (step % args.eval_every == 0 or step == args.steps):
            value = evaluate()
            record = {"step":step, "validation_ema_mse":value}
            metrics.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
        if args.sample_every and (step % args.sample_every == 0 or step == args.steps):
            samples = sample_ddim(ema, schedule, sample_tokens, tokenizer.encode([""]*len(prompts), device),
                steps=args.sample_steps, guidance=args.sample_guidance, seed=args.seed+123, amp=use_amp)
            save_grid(samples, out/"samples"/f"step-{step:07d}.png")
            print(json.dumps({"event":"samples_saved", "step":step}), flush=True)
        if (args.save_every and step % args.save_every == 0) or step == args.steps:
            save(step)
            print(json.dumps({"event":"checkpoint_saved", "step":step}), flush=True)
    metrics.close()
    print(json.dumps({"event":"complete", "step":args.steps, "checkpoint":str(out/"checkpoints"/"latest.pt")}), flush=True)


if __name__ == "__main__":
    main()
