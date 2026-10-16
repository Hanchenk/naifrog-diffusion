#!/usr/bin/env python3
"""生成全部 24 种提示词组合；同一 seed 内复用完全相同的初始噪声张量。"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys


VIEWS = {"front": "正面", "left": "左前方", "right": "右前方"}
FRAMINGS = {"close": "近景", "full": "全身"}
BACKGROUNDS = {"blue": "蓝色", "pink": "粉色", "green": "绿色", "white": "白色"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-code", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, action="append", help="Repeat for additional independently seeded 24-image groups; default 42")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-threads", type=int, default=4, help="PyTorch CPU threads; avoids excess scheduling overhead")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    sys.path.insert(0, str(args.model_code.resolve()))
    import torch
    torch.set_num_threads(args.cpu_threads)
    from PIL import Image
    from model import checkpoint_model, sample_ddim
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)
    model, tokenizer, schedule, checkpoint = checkpoint_model(str(args.checkpoint), args.device, True)
    records = []
    initial_noise_hashes = {}
    for seed in args.seed or [42]:
        generator = torch.Generator(device=args.device).manual_seed(seed)
        initial_noise = torch.randn((1, 3, model.cfg.image_size, model.cfg.image_size),
            generator=generator, device=args.device)
        noise_hash = hashlib.sha256(initial_noise.detach().cpu().numpy().tobytes()).hexdigest()
        initial_noise_hashes[str(seed)] = noise_hash
        for view, framing, background in itertools.product(VIEWS, FRAMINGS, BACKGROUNDS):
            caption = f"奶蛙捧腹大笑，{VIEWS[view]}视角，{FRAMINGS[framing]}，{BACKGROUNDS[background]}背景"
            unknown = tokenizer.unknown_characters(caption)
            if unknown:
                raise ValueError(f"Evaluation caption has untrained characters: {unknown}")
            # 复用同一个实际噪声张量，不能只给不同随机图标注相同 seed；采样器会 clone。
            # 若直接生成多提示词随机 batch，每张图通常会拿到不同的初始噪声。
            with torch.inference_mode():
                result = sample_ddim(model, schedule, tokenizer.encode([caption], args.device),
                    tokenizer.encode([""], args.device), steps=args.steps, guidance=args.guidance,
                    seed=seed, initial_noise=initial_noise, amp=not args.no_amp)
            array = result[0].detach().float().clamp(-1, 1).add(1).mul(127.5).byte().cpu().permute(1, 2, 0).numpy()
            filename = f"images/seed-{seed}_{view}_{framing}_{background}.png"
            Image.fromarray(array).save(args.out / filename)
            records.append({"image": filename, "file_name": filename, "caption": caption,
                "view": view, "framing": framing, "background": background, "seed": seed,
                "checkpoint_step": checkpoint["step"], "ddim_steps": args.steps,
                "guidance": args.guidance, "ema": True, "initial_noise_sha256": noise_hash})
            print(f"saved {filename}", flush=True)
    (args.out / "generated.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    (args.out / "generation_info.json").write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest.hexdigest(), "checkpoint_step": checkpoint["step"],
        "seeds": args.seed or [42], "images": len(records), "same_initial_noise_within_seed": True,
        "initial_noise_sha256_by_seed": initial_noise_hashes,
        "ddim_steps": args.steps, "guidance": args.guidance, "ema": True,
        "cpu_threads": args.cpu_threads}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
