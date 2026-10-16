#!/usr/bin/env python3
"""从真实训练预览中裁切两个固定提示词的结果；不重绘或修饰生成图。"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--samples',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
steps=[1000,3000,5000,10000]
canvas=Image.new('RGB',(192*4,216*2+26),'#fff9ec')
draw=ImageDraw.Draw(canvas)
draw.text((6,7),'Real training progression / same fixed prompts and seed',fill='#183cae')
for col,step in enumerate(steps):
    with Image.open(args.samples/f'step-{step:07d}.png') as grid:
        if grid.size!=(768,384):
            raise ValueError(f'Unexpected grid dimensions: {grid.size}')
        for row,y in enumerate([0,192]):
            canvas.paste(grid.crop((0,y,192,y+192)),(col*192,26+row*216+24))
            draw.text((col*192+6,26+row*216+6),f'step {step:,}',fill='#183cae')
args.out.parent.mkdir(parents=True,exist_ok=True)
canvas.save(args.out)
