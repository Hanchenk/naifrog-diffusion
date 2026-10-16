#!/usr/bin/env python3
"""用轻量像素指标评价真实的 64px 生成图，并明确这些指标的适用范围。"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = ("front", "left", "right")
FRAMINGS = ("close", "full")
BACKGROUNDS = ("blue", "pink", "green", "white")
EXPECTED = set(itertools.product(VIEWS, FRAMINGS, BACKGROUNDS))
CANONICAL_COLORS = {"blue": "#9db9e8", "pink": "#efb8cc", "green": "#b3d8bc", "white": "#f3f3ef"}
LIMITATIONS = [
    "This is a same-subject synthetic-render evaluation at 64x64, not an open-domain text-to-image benchmark.",
    "Background matching uses four corner patches and empirical RGB references; it does not assess subject quality or text understanding.",
    "Pixel nearest-neighbor distance is sensitive to background, camera, alignment and lighting. A nonzero distance cannot prove absence of memorization, and a small distance alone cannot prove copying.",
    "Factor panels use the same starting-noise seed for controlled visual comparison. View/framing changes require human inspection; pixel changes do not establish correct semantic control.",
    "No FID, CLIP score, perceptual-quality claim or statistically validated generalization claim is produced.",
]


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        relative = row.get("image", row.get("file_name"))
        if not relative:
            raise ValueError(f"Missing image/file_name in {path}")
        row["_path"] = (path.parent / relative).resolve()
        for key, choices in (("view", VIEWS), ("framing", FRAMINGS), ("background", BACKGROUNDS)):
            if row.get(key) not in choices:
                raise ValueError(f"Missing or invalid {key}: {row}")
    if not rows:
        raise ValueError(f"No image records in {path}")
    return rows


def load_pixels(rows):
    images = []
    for row in rows:
        with Image.open(row["_path"]) as image:
            if image.size != (64, 64):
                raise ValueError(f"Expected 64x64 image, got {image.size}: {row['_path']}")
            images.append(np.array(image.convert("RGB"), dtype=np.uint8))
    return np.stack(images)


def corner_color(image, patch):
    corners = [image[:patch, :patch], image[:patch, -patch:],
               image[-patch:, :patch], image[-patch:, -patch:]]
    # 四角 RGB 取中位数，减小零散角色像素与边缘抗锯齿的影响。
    return np.median(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0)


def key(row):
    return row["view"], row["framing"], row["background"]


def tile(image, lines, size=192):
    canvas = Image.new("RGB", (size, size + 55), "#fff9ec")
    canvas.paste(Image.fromarray(image).resize((size, size), Image.Resampling.NEAREST), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for n, line in enumerate(lines[:4]):
        draw.text((5, size + 4 + 12*n), line, font=font, fill="#183cae")
    return canvas


def contact_sheet(tiles, columns, path, title=""):
    cell_w, cell_h = tiles[0].size
    header = 26 if title else 0
    canvas = Image.new("RGB", (cell_w*columns, cell_h*((len(tiles)+columns-1)//columns)+header), "white")
    if title:
        ImageDraw.Draw(canvas).text((8, 7), title, fill="#183cae")
    for i, image in enumerate(tiles):
        canvas.paste(image, ((i % columns)*cell_w, (i//columns)*cell_h+header))
    canvas.save(path)


def nearest_neighbors(train_pixels, generated_pixels, train_rows, generated_rows):
    # 分块穷举精确像素平方差；没有使用特征 embedding、近似近邻索引
    # 或预训练评价模型。非零像素距离本身不能证明没有记忆训练图片。
    train = train_pixels.reshape(len(train_pixels), -1).astype(np.float32)
    train_hashes = defaultdict(list)
    for i, image in enumerate(train_pixels):
        train_hashes[hashlib.sha256(image.tobytes()).hexdigest()].append(i)
    results = []
    for row, image in zip(generated_rows, generated_pixels):
        vector = image.reshape(-1).astype(np.float32)
        squared = np.empty(len(train), dtype=np.float64)
        for start in range(0, len(train), 128):
            delta = train[start:start+128] - vector
            squared[start:start+128] = np.einsum("ij,ij->i", delta, delta, dtype=np.float64) / train.shape[1]
        best = int(np.argmin(squared))
        compatible = np.array([i for i, r in enumerate(train_rows) if key(r) == key(row)], dtype=np.int64)
        conditioned = int(compatible[np.argmin(squared[compatible])]) if len(compatible) else None
        exact_matches = train_hashes.get(hashlib.sha256(image.tobytes()).hexdigest(), [])
        results.append({"train_index": best, "nearest_image": train_rows[best]["image"] if "image" in train_rows[best] else train_rows[best]["file_name"],
            "nearest_caption": train_rows[best].get("caption"), "pixel_rmse_0_to_1": float(np.sqrt(squared[best])/255.0),
            "same_factors_train_index": conditioned,
            "same_factors_nearest_image": None if conditioned is None else train_rows[conditioned].get("image", train_rows[conditioned].get("file_name")),
            "same_factors_pixel_rmse_0_to_1": None if conditioned is None else float(np.sqrt(squared[conditioned])/255.0),
            "exact_rgb_training_matches": len(exact_matches)})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True, help="Training JSONL; image paths relative to this file")
    p.add_argument("--generated", type=Path, required=True, help="Generated JSONL with image, caption, factors and seed")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corner-patch", type=int, default=6)
    args = p.parse_args()
    if not 1 <= args.corner_patch <= 12:
        p.error("--corner-patch must be 1..12 for 64px images")
    args.out.mkdir(parents=True, exist_ok=True)
    training, generated = read_rows(args.train), read_rows(args.generated)
    train_pixels, gen_pixels = load_pixels(training), load_pixels(generated)
    training_colors = np.stack([corner_color(image, args.corner_patch) for image in train_pixels])
    references = {}
    for background in BACKGROUNDS:
        indices = [i for i, row in enumerate(training) if row["background"] == background]
        if not indices:
            raise ValueError(f"No training background examples for {background}")
        references[background] = np.median(training_colors[indices], axis=0)
    reference_matrix = np.stack([references[b] for b in BACKGROUNDS])
    nearest = nearest_neighbors(train_pixels, gen_pixels, training, generated)
    records = []
    for i, (row, image) in enumerate(zip(generated, gen_pixels)):
        color = corner_color(image, args.corner_patch)
        distances = np.sqrt(np.mean(((reference_matrix-color)/255.)**2, axis=1))
        predicted = BACKGROUNDS[int(np.argmin(distances))]
        record = {k: v for k, v in row.items() if not k.startswith("_")}
        record.update({"corner_median_rgb": color.tolist(), "background_predicted_from_corners": predicted,
            "background_target_matches_nearest_rgb_reference": predicted == row["background"],
            "background_rgb_rmse_0_to_1": float(distances[BACKGROUNDS.index(row["background"])]),
            "nearest_neighbor": nearest[i]})
        records.append(record)
    groups = defaultdict(list)
    for i, row in enumerate(generated):
        if "seed" not in row:
            raise ValueError("Each generated record needs seed for controlled-factor panels")
        groups[str(row["seed"])].append(i)
    factor_comparisons = []
    for seed, indices in groups.items():
        mapping = {key(generated[i]): i for i in indices}
        if set(mapping) != EXPECTED or len(indices) != 24:
            missing = sorted(EXPECTED-set(mapping))
            raise ValueError(f"Seed {seed} needs exactly 24 unique combinations; found {len(indices)}, missing {missing}")
        ordered = [mapping[k] for k in itertools.product(VIEWS, FRAMINGS, BACKGROUNDS)]
        contact_sheet([tile(gen_pixels[i], [" / ".join(key(generated[i])), f"seed={seed}"]) for i in ordered], 4,
            args.out / f"all-24-seed-{seed}.png", f"All 24 conditions / fixed seed {seed} / original 64px enlarged by nearest-neighbor")
        base = ("front", "full", "blue")
        factors = [("background", [("front", "full", b) for b in BACKGROUNDS]),
                   ("view", [(v, "full", "blue") for v in VIEWS]),
                   ("framing", [("front", f, "blue") for f in FRAMINGS])]
        panel = Image.new("RGB", (192*4, 247*3+26), "white")
        ImageDraw.Draw(panel).text((8, 8), f"Same initial noise: seed {seed}; reference=front/full/blue", fill="#183cae")
        for row_index, (factor, combos) in enumerate(factors):
            for col, combo in enumerate(combos):
                idx = mapping[combo]
                diff = float(np.sqrt(np.mean(((gen_pixels[idx].astype(float)-gen_pixels[mapping[base]].astype(float))/255.)**2)))
                factor_comparisons.append({"seed": seed, "varied_factor": factor, "condition": list(combo), "reference": list(base), "pixel_change_rmse_0_to_1": diff})
                panel.paste(tile(gen_pixels[idx], [factor+": "+"/".join(combo), f"pixel change vs reference: {diff:.4f}"]), (col*192, row_index*247+26))
        panel.save(args.out / f"factor-comparison-seed-{seed}.png")
        for page, start in enumerate(range(0, len(ordered), 6)):
            tiles = []
            for i in ordered[start:start+6]:
                nn = nearest[i]
                tiles.append(tile(gen_pixels[i], ["GENERATED / "+"/".join(key(generated[i])), f"seed={seed}"]))
                ni = nn["train_index"]
                tiles.append(tile(train_pixels[ni], ["GLOBAL PIXEL NEAREST", f"RMSE={nn['pixel_rmse_0_to_1']:.5f}", str(Path(nn["nearest_image"]).name), "/".join(key(training[ni]))]))
                ni = nn["same_factors_train_index"]
                if ni is None:
                    tiles.append(tile(np.full((64,64,3),255,dtype=np.uint8), ["No same-condition training image"]))
                else:
                    tiles.append(tile(train_pixels[ni], ["SAME-CONDITION PIXEL NEAREST", f"RMSE={nn['same_factors_pixel_rmse_0_to_1']:.5f}", str(Path(nn["same_factors_nearest_image"]).name)]))
            contact_sheet(tiles, 3, args.out / f"nearest-neighbors-seed-{seed}-page-{page+1}.png",
                "Pixel nearest-neighbor comparisons; these distances cannot establish absence of memorization")
    rmse = [n["pixel_rmse_0_to_1"] for n in nearest]
    report = {"training_manifest": str(args.train.resolve()), "generated_manifest": str(args.generated.resolve()),
        "training_images": len(training), "generated_images": len(generated), "image_size": [64,64],
        "seeds": sorted(groups), "combinations_per_seed": 24,
        "background_metric": {"method": f"Median RGB over four {args.corner_patch}x{args.corner_patch} corner patches; nearest empirical training-background RGB centroid",
            "canonical_render_color_specification": CANONICAL_COLORS,
            "empirical_training_corner_medians_rgb": {k:v.tolist() for k,v in references.items()},
            "target_match_count": sum(r["background_target_matches_nearest_rgb_reference"] for r in records),
            "target_match_rate": float(np.mean([r["background_target_matches_nearest_rgb_reference"] for r in records])),
            "mean_target_rgb_rmse_0_to_1": float(np.mean([r["background_rgb_rmse_0_to_1"] for r in records]))},
        "pixel_nearest_neighbor_metric": {"distance": "sqrt(mean((generated_RGB-training_RGB)^2))/255 over all 64x64x3 channels",
            "minimum_rmse": min(rmse), "median_rmse": float(np.median(rmse)), "mean_rmse": float(np.mean(rmse)),
            "generated_images_with_exact_rgb_training_match": sum(n["exact_rgb_training_matches"] > 0 for n in nearest),
            "caveat": LIMITATIONS[2]},
        "factor_comparisons": factor_comparisons, "images": records, "limitations": LIMITATIONS}
    info_path = args.generated.parent / "generation_info.json"
    if info_path.exists():
        report["generation_info"] = json.loads(info_path.read_text(encoding="utf-8"))
    (args.out / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output":str(args.out.resolve()), "images":len(generated),
        "background_target_match_rate":report["background_metric"]["target_match_rate"],
        "median_pixel_nearest_neighbor_rmse":report["pixel_nearest_neighbor_metric"]["median_rmse"],
        "limitations":LIMITATIONS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
