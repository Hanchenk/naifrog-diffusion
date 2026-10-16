"""小型中文字符条件 DDPM：所有参数从随机权重开始学习。

训练目标是二维 RGB 渲染图，不输出三维几何。文本 token 通过交叉注意力
影响噪声预测。这里没有预训练模型、预训练分词器或外部模型服务。
"""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    image_size: int = 64
    base_channels: int = 48
    text_width: int = 128
    text_layers: int = 2
    max_length: int = 64
    diffusion_steps: int = 1000
    dropout: float = 0.05


# 字符级词表从训练 captions 构建：不下载语言模型，也不预设语义类别。
class CharacterTokenizer:
    SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, vocab, max_length=64):
        self.vocab = list(vocab)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.max_length = max_length

    @classmethod
    def build(cls, captions, max_length=64):
        return cls(cls.SPECIAL + sorted(set("".join(captions))), max_length)

    def encode(self, captions, device=None):
        out = torch.zeros((len(captions), self.max_length), dtype=torch.long)
        for i, caption in enumerate(captions):
            ids = [1] + [self.stoi.get(c, 3) for c in caption[:self.max_length-2]] + [2]
            out[i, :len(ids)] = torch.tensor(ids)
        return out.to(device) if device is not None else out

    def unknown_characters(self, caption):
        return sorted(set(c for c in caption if c not in self.stoi))


# 用正余弦把整数时间步映射到连续向量，让同一网络区分噪声强弱。
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / max(half-1, 1))
    phases = timesteps.float()[:, None] * freqs[None]
    return torch.cat([phases.cos(), phases.sin()], dim=-1)


def norm(channels):
    return nn.GroupNorm(min(16, channels), channels)


# 残差卷积块通过通道偏置注入时间向量；skip 保留原始特征路径。
class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, time_dim, dropout):
        super().__init__()
        self.norm1 = norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(time_dim, cout)
        self.norm2 = norm(cout)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


# 图像位置作为 query，文本字符作为 key/value；每个图像位置可读取相关文字。
class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = norm(channels)
        self.q = nn.Linear(channels, channels, bias=False)
        self.kv = nn.Linear(context_dim, 2 * channels, bias=False)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, context, token_mask):
        b, c, h, w = x.shape
        q = self.q(self.norm(x).flatten(2).transpose(1, 2))
        k, v = self.kv(context).chunk(2, dim=-1)
        q = q.reshape(b, h*w, self.heads, c//self.heads).transpose(1, 2)
        k = k.reshape(b, -1, self.heads, c//self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, c//self.heads).transpose(1, 2)
        # 注意：PyTorch SDPA 的布尔 mask 中，True 表示允许参与注意力的 key。
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=token_mask[:, None, None, :])
        attended = attended.transpose(1, 2).reshape(b, h*w, c)
        return x + self.out(attended).transpose(1, 2).reshape(b, c, h, w)


# 小型 Transformer 从头初始化，与图像网络一起训练；字符 PAD 嵌入初值为零。
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.text_width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, cfg.max_length, cfg.text_width) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.text_width, 4, cfg.text_width*3,
                    dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.text_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.text_width)

    def forward(self, tokens):
        mask = tokens.ne(0)
        x = self.embedding(tokens) + self.position[:, :tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask)
        return self.norm(x), mask


# 输入/输出均为 [B,3,64,64]；输出预测噪声，不是直接预测 RGB。
# 逐级缩小空间分辨率，在 16px/8px 层交叉注意力读入文字，再上采样恢复细节。
class ConditionalUNet(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        b = cfg.base_channels
        self.time_dim = b * 4
        self.text = TextEncoder(vocab_size, cfg)
        self.time = nn.Sequential(nn.Linear(self.time_dim, self.time_dim*2), nn.SiLU(), nn.Linear(self.time_dim*2, self.time_dim))
        self.input = nn.Conv2d(3, b, 3, padding=1)
        channels = [b, b*2, b*3, b*4]
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()
        previous = b
        for i, c in enumerate(channels):
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(previous, c, self.time_dim, cfg.dropout),
                ResidualBlock(c, c, self.time_dim, cfg.dropout)]))
            self.down_attn.append(CrossAttention(c, cfg.text_width) if i >= 2 else nn.Identity())
            if i < 3:
                self.downsample.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            previous = c
        self.mid1 = ResidualBlock(channels[-1], channels[-1], self.time_dim, cfg.dropout)
        self.mid_attn = CrossAttention(channels[-1], cfg.text_width)
        self.mid2 = ResidualBlock(channels[-1], channels[-1], self.time_dim, cfg.dropout)
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, c in reversed(list(enumerate(channels))):
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(previous+c, c, self.time_dim, cfg.dropout),
                ResidualBlock(c, c, self.time_dim, cfg.dropout)]))
            self.up_attn.append(CrossAttention(c, cfg.text_width) if i >= 2 else nn.Identity())
            if i > 0:
                self.upsample.append(nn.Conv2d(c, c, 3, padding=1))
            previous = c
        self.output_norm = norm(b)
        self.output = nn.Conv2d(b, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_text(self, tokens):
        return self.text(tokens)

    def forward(self, x, timesteps, tokens=None, encoded=None):
        context, mask = self.encode_text(tokens) if encoded is None else encoded
        time = self.time(timestep_embedding(timesteps, self.time_dim))
        h = self.input(x)
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            h = blocks[1](blocks[0](h, time), time)
            if isinstance(self.down_attn[i], CrossAttention):
                h = self.down_attn[i](h, context, mask)
            skips.append(h)
            if i < 3:
                h = self.downsample[i](h)
        h = self.mid2(self.mid_attn(self.mid1(h, time), context, mask), time)
        for i, blocks in enumerate(self.up_blocks):
            h = blocks[1](blocks[0](torch.cat([h, skips.pop()], dim=1), time), time)
            if isinstance(self.up_attn[i], CrossAttention):
                h = self.up_attn[i](h, context, mask)
            if i < 3:
                h = self.upsample[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(F.silu(self.output_norm(h)))


class DiffusionSchedule:
    def __init__(self, steps=1000, device="cpu"):
        # 使用 Improved DDPM 的余弦累计 alpha 日程，并将 beta 上限设为 0.999。
        t = torch.linspace(0, steps, steps+1, dtype=torch.float64)
        f = torch.cos(((t/steps + 0.008)/1.008) * math.pi/2).square()
        target = f / f[0]
        betas = (1 - target[1:]/target[:-1]).clamp(0, 0.999)
        self.betas = betas.float().to(device)
        self.alpha_bars = (1-betas).cumprod(0).float().to(device)
        self.steps = steps

    def add_noise(self, clean, noise, t):
        a = self.alpha_bars[t, None, None, None]
        return a.sqrt()*clean + (1-a).sqrt()*noise


@torch.inference_mode()
def sample_ddim(model, schedule, tokens, null_tokens, steps=50, guidance=2.5,
                seed=42, initial_noise=None, return_process=False, amp=True):
    """确定性 DDIM（eta=0）加 epsilon 形式的 CFG，返回 [-1,1] 的 RGB 张量。"""
    device = next(model.parameters()).device
    if not 1 <= steps <= schedule.steps:
        raise ValueError(f"Sampling steps must be 1..{schedule.steps}")
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (len(tokens), 3, model.cfg.image_size, model.cfg.image_size)
    x = torch.randn(shape, device=device, generator=generator) if initial_noise is None else initial_noise.clone().to(device)
    # 同一次前向批量计算无条件和有条件预测，随后按 CFG 权重组合。
    combined_tokens = torch.cat([null_tokens, tokens])
    use_amp = amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
        encoded = model.encode_text(combined_tokens)
    times = torch.linspace(schedule.steps-1, 0, steps, device=device).round().long()
    process = [("初始噪声", x.detach().cpu())] if return_process else []
    marks = set(torch.linspace(0, steps-1, 7).round().long().tolist())
    for i, t in enumerate(times):
        ts = torch.full((len(tokens)*2,), int(t), dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            eps_null, eps_text = model(torch.cat([x, x]), ts, encoded=encoded).chunk(2)
        # s=1 为普通条件预测，s=0 为无条件；这里 guidance 对应论文 1+w。
        eps = eps_null.float() + guidance*(eps_text.float()-eps_null.float())
        a = schedule.alpha_bars[t]
        # 从加噪公式反解 x0；低 SNR 易放大误差，因此用 FP32 并裁剪范围。
        x0 = ((x - (1-a).sqrt()*eps)/a.sqrt()).clamp(-1, 1)
        if i == len(times)-1:
            x = x0
        else:
            a_prev = schedule.alpha_bars[times[i+1]]
            x = a_prev.sqrt()*x0 + (1-a_prev).sqrt()*eps
        if return_process and i in marks:
            process.append((f"t={int(t)}", x.detach().cpu()))
    return (x, process) if return_process else x


# 从权重随附的架构/字符表重建模型，默认使用训练期间累积的 EMA 参数。
def checkpoint_model(checkpoint_path, device="cpu", use_ema=True):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["model_config"])
    tokenizer = CharacterTokenizer(checkpoint["vocab"], cfg.max_length)
    model = ConditionalUNet(cfg, len(tokenizer.vocab))
    model.load_state_dict(checkpoint["ema"] if use_ema else checkpoint["model"])
    model.to(device).eval()
    return model, tokenizer, DiffusionSchedule(cfg.diffusion_steps, device), checkpoint
