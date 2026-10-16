#!/usr/bin/env python3
"""直接绘制原始训练 JSONL，不修改、平滑或合并记录。

训练窗口 MSE 与部分验证集 EMA MSE 使用不同图片、噪声和权重。
分轴呈现，避免让读者误以为两者是逐样本配对的直接比较。
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read_log(path):
    records, skipped = [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("not an object")
            records.append(value)
        except ValueError:
            skipped.append(number)
    return records, skipped


def series(records, key):
    # 保留原始日志顺序和重复 step，包含恢复训练后的重复记录。
    return [(r["step"], r[key]) for r in records
            if isinstance(r.get("step"), (int, float)) and isinstance(r.get(key), (int, float))
            and math.isfinite(r["step"]) and math.isfinite(r[key])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("dpi must be 72..600")
    # 延迟导入绘图库，使 --help 只依赖标准库。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records, skipped = read_log(args.metrics)
    train = series(records, "train_mse")
    val = series(records, "validation_ema_mse")
    lr = series(records, "lr")
    if not train and not val:
        raise ValueError("No finite train_mse or validation_ema_mse records found")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fffaf0", "axes.facecolor": "#fffaf0",
                         "savefig.facecolor": "#fffaf0", "text.color": "#17357c", "axes.labelcolor": "#17357c"})
    rows = 3 if lr else 2
    fig, axes = plt.subplots(rows, 1, figsize=(9, 8.3 if lr else 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, .6] if lr else [1, 1]})
    fig.suptitle("Naifrog: diffusion epsilon-prediction objective", x=.11, y=.97, ha="left", fontsize=14, weight="bold")
    panels = [(axes[0], train, "#183cae", "Training MSE", "Raw logged window means; training weights, random batches/noise"),
              (axes[1], val, "#d83728", "Validation EMA MSE", "Raw logged fixed partial-validation estimates; EMA weights")]
    for ax, values, color, label, title in panels:
        if values:
            steps, errors = zip(*values)
            ax.plot(steps, errors, color=color, linewidth=1.15, marker="o", markersize=2 if len(values) > 100 else 3)
        else:
            ax.text(.5, .5, "No logged observations", transform=ax.transAxes, ha="center")
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#c8c9be", alpha=.6, linewidth=.6)
    if lr:
        steps, values = zip(*lr)
        axes[2].plot(steps, values, color="#536a70", linewidth=1.1)
        axes[2].set_ylabel("Learning rate")
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[2].set_ylim(bottom=0)
        axes[2].grid(axis="y", color="#c8c9be", alpha=.6, linewidth=.6)
    axes[-1].set_xlabel("Training iteration (step)")
    footer = ("No smoothing or resampling. Training and validation estimates are not paired.\n"
              "Denoising MSE does not measure visual quality or semantic understanding. Raw JSONL remains the source of truth.")
    fig.text(.11, .025, footer, fontsize=8, linespacing=1.6, color="#5d6464")
    fig.subplots_adjust(left=.11, right=.97, top=.89, bottom=.13, hspace=.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    duplicate_train = {str(step): count for step, count in Counter(point[0] for point in train).items() if count > 1}
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(args.metrics.resolve()), "source_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
                "output": str(args.out.resolve()), "training_points": len(train), "validation_points": len(val),
                "learning_rate_points": len(lr), "invalid_json_line_numbers": skipped,
                "duplicate_training_steps_preserved": duplicate_train,
                "smoothing": "none", "order": "original log order; duplicate/resumed steps preserved",
                "caveat": "Training and validation use different batches/noise and training versus EMA weights; denoising MSE is not a visual/semantic score."}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
