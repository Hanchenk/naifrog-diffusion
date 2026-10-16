#!/usr/bin/env python3
"""加载本地训练权重，按中文提示词生成 RGB 图片。"""
import argparse
import json
import math
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from model import checkpoint_model, sample_ddim


def to_image(tensor):
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", action="append", required=True, help="Can be repeated for batch generation")
    p.add_argument("--out", required=True, help="PNG; for multiple prompts append -000, -001, ...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4, help="CPU intra-op threads; small batches benefit from a modest count")
    p.add_argument("--process", help="Optional denoising contact-sheet PNG")
    p.add_argument("--raw-weights", action="store_true", help="Use training weights instead of EMA")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.cpu_threads)
    model, tokenizer, schedule, checkpoint = checkpoint_model(args.checkpoint, args.device, not args.raw_weights)
    unsupported = {caption: tokenizer.unknown_characters(caption) for caption in args.prompt}
    unsupported = {k:v for k,v in unsupported.items() if v}
    if unsupported:
        print(json.dumps({"warning":"Some characters were absent from training and become <unk>", "characters":unsupported}, ensure_ascii=False), flush=True)
    tokens = tokenizer.encode(args.prompt, args.device)
    null_tokens = tokenizer.encode([""]*len(args.prompt), args.device)
    result = sample_ddim(model, schedule, tokens, null_tokens, steps=args.steps, guidance=args.guidance,
        seed=args.seed, return_process=bool(args.process), amp=not args.no_amp)
    images, process = result if args.process else (result, None)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, tensor in enumerate(images):
        path = target if len(images) == 1 else target.with_name(f"{target.stem}-{i:03d}{target.suffix}")
        to_image(tensor).save(path)
        outputs.append(str(path.resolve()))
    if process:
        size, header = model.cfg.image_size*3, 24
        canvas = Image.new("RGB", (size*len(process), (size+header)*len(images)), "#fff9ec")
        draw = ImageDraw.Draw(canvas)
        for j, (label, batch) in enumerate(process):
            for i, tensor in enumerate(batch):
                y = i*(size+header)
                canvas.paste(to_image(tensor).resize((size, size), Image.Resampling.NEAREST), (j*size, y+header))
                # 过程图使用 ASCII 标签，避免无桌面的 Linux 缺少中文字体。
                draw.text((j*size+5, y+5), "initial noise" if j == 0 else label, fill="#183cae")
        process_path = Path(args.process)
        process_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(process_path)
    metadata = {"checkpoint":str(Path(args.checkpoint).resolve()), "training_step":checkpoint["step"],
        "prompts":args.prompt, "seed":args.seed, "ddim_steps":args.steps, "guidance":args.guidance,
        "ema":not args.raw_weights, "outputs":outputs, "resolution":model.cfg.image_size,
        "unknown_characters":unsupported}
    target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
