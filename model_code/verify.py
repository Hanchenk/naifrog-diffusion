#!/usr/bin/env python3
"""数值检查：固定噪声学习、padding mask、DDIM 公式与权重保存加载。

检查不依赖奶蛙数据集，用小网络验证数学和代码路径。
通过这些检查不等于真实生成图有足够视觉质量或能正确遵循所有提示词。
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import torch
from model import (ModelConfig, CharacterTokenizer, ConditionalUNet,
                   DiffusionSchedule, sample_ddim, checkpoint_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    device = torch.device(args.device)
    schedule = DiffusionSchedule(device=device)
    clean = torch.rand(2, 3, 64, 64, device=device)*1.6-.8
    noise = torch.randn_like(clean)
    t = torch.tensor([100, 800], device=device)
    noisy = schedule.add_noise(clean, noise, t)
    a = schedule.alpha_bars[t, None, None, None]
    recovered = (noisy-(1-a).sqrt()*noise)/a.sqrt()
    inverse_error = (recovered-clean).abs().max().item()
    assert inverse_error < 2e-5, inverse_error

    cfg = ModelConfig(base_channels=16, text_width=32, text_layers=1, max_length=20, dropout=0)
    tokenizer = CharacterTokenizer.build(["奶蛙蓝色", "奶蛙粉色"], cfg.max_length)
    tokens = tokenizer.encode(["奶蛙蓝色", "奶蛙粉色"], device)
    null = tokenizer.encode(["", ""], device)
    model = ConditionalUNet(cfg, len(tokenizer.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    first, final = None, None
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy, t, tokens)
        loss = (pred-noise).square().mean()
        loss.backward()
        optimizer.step()
        first = loss.item() if first is None else first
        final = loss.item()
    assert final < first*.9, (first, final)
    text_grad = sum(p.grad.abs().sum().item() for p in model.text.parameters() if p.grad is not None)
    assert text_grad > 0, "Text conditioning did not receive gradients"
    model.eval()
    with torch.no_grad():
        reference = model(noisy, t, tokens)
        pad = model.text.embedding.weight[0].clone()
        model.text.embedding.weight[0].fill_(123.0)
        changed = model(noisy, t, tokens)
        model.text.embedding.weight[0].copy_(pad)
        mask_error = (changed-reference).abs().max().item()
        assert mask_error < 1e-4, mask_error

    # 数学 oracle 已知干净目标 x0，能够精确给出该输入中的真实噪声。
    # 因此 DDIM 应当恢复同一干净目标，不应依赖初始随机种子。
    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), device=device))
            self.cfg = cfg
        def encode_text(self, tok):
            return tok, tok.ne(0)
        def forward(self, x, ts, tokens=None, encoded=None):
            alpha = schedule.alpha_bars[ts, None, None, None]
            return (x-alpha.sqrt()*.25)/(1-alpha).sqrt()

    oracle = Oracle()
    oracle_images = sample_ddim(oracle, schedule, tokens, null, steps=9, guidance=2.5, amp=False)
    oracle_error = (oracle_images-.25).abs().max().item()
    assert oracle_error < .002, oracle_error

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/"test.pt"
        torch.save({"model_config":asdict(cfg), "vocab":tokenizer.vocab,
                    "model":model.state_dict(), "ema":model.state_dict(), "step":16}, path)
        restored, restored_tokenizer, restored_schedule, checkpoint = checkpoint_model(path, device)
        with torch.no_grad():
            checkpoint_error = (restored(noisy, t, tokens)-reference).abs().max().item()
        assert checkpoint_error == 0, checkpoint_error
        one = sample_ddim(restored, restored_schedule, tokens, null, steps=3, guidance=0, seed=9, amp=False)
        two = sample_ddim(restored, restored_schedule, tokens, null, steps=3, guidance=0, seed=9, amp=False)
        assert torch.equal(one, two), "Fixed-seed sampling is not repeatable"
        shared_noise = torch.randn(1, 3, 64, 64, device=device).repeat(2, 1, 1, 1)
        unconditional = sample_ddim(restored, restored_schedule, tokens, null, steps=3,
                        guidance=0, initial_noise=shared_noise, amp=False)
        cfg_error = (unconditional[0]-unconditional[1]).abs().max().item()
        assert cfg_error < 1e-4, cfg_error

    print(json.dumps({"ok":True, "device":str(device), "forward_inverse_max_error":inverse_error,
        "fixed_batch_initial_mse":first, "fixed_batch_final_mse":final, "text_gradient_l1":text_grad,
        "padding_mask_max_error":mask_error, "oracle_ddim_max_error":oracle_error,
        "checkpoint_roundtrip_max_error":checkpoint_error,
        "cfg_zero_prompt_independence_max_error":cfg_error,
        "fixed_seed_repeatability":True}, indent=2))


if __name__ == "__main__":
    main()
