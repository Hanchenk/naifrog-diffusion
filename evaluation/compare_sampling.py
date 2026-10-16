#!/usr/bin/env python3
"""保持初始噪声张量一致，比较真实 DDIM 步数和 CFG 系数的效果。"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-code',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prompt',default='奶蛙捧腹大笑，正面视角，全身，蓝色背景')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    sys.path.insert(0,str(args.model_code.resolve()))
    import torch
    from PIL import Image,ImageDraw
    from model import checkpoint_model,sample_ddim
    torch.set_num_threads(4)
    model,tokenizer,schedule,checkpoint=checkpoint_model(args.checkpoint,args.device)
    tokens=tokenizer.encode([args.prompt],args.device)
    null=tokenizer.encode([''],args.device)
    noise=torch.randn((1,3,64,64),device=args.device,generator=torch.Generator(device=args.device).manual_seed(args.seed))
    args.out.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(192*3,216*2),'#fff9ec')
    draw=ImageDraw.Draw(canvas)
    records=[]
    for row,steps in enumerate([50,100]):
        for col,guidance in enumerate([1.0,1.5,2.5]):
            tensor=sample_ddim(model,schedule,tokens,null,steps=steps,guidance=guidance,seed=args.seed,initial_noise=noise)[0]
            array=tensor.float().clamp(-1,1).add(1).mul(127.5).byte().cpu().permute(1,2,0).numpy()
            img=Image.fromarray(array)
            name=f'steps-{steps}_guidance-{guidance}.png'
            img.save(args.out/name)
            canvas.paste(img.resize((192,192),Image.Resampling.NEAREST),(col*192,row*216+24))
            draw.text((col*192+5,row*216+6),f'{steps} steps / guidance {guidance}',fill='#183cae')
            records.append({'file':name,'ddim_steps':steps,'guidance':guidance})
    canvas.save(args.out/'sampling_comparison.png')
    (args.out/'sampling_comparison.json').write_text(json.dumps({'checkpoint_step':checkpoint['step'],'prompt':args.prompt,'seed':args.seed,'same_initial_noise':True,'results':records},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
