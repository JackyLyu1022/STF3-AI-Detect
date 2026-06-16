
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import read_video_frames
from src.models.frequency_branch import FrequencyBranch

ROOT = Path('.')
ASSETS = ROOT / 'docs' / 'ppt_assets'
OUT = ASSETS / 'final_demo_case_study_optimized.png'
# also replace old asset for future rebuilds
OUT_COMPAT = ASSETS / 'final_demo_case_study.png'
DATA_ROOT = ROOT / 'data' / 'GenVideo-Val'
PRED = ROOT / 'outputs' / 'medium_stf3' / 'predictions.csv'

W, H = 2600, 1200
BG = (247, 250, 253)
NAVY = (30, 58, 95)
MUTED = (96, 112, 128)
GRID = (221, 230, 242)
BLUE = (45, 100, 165)
RED = (238, 0, 0)
GREEN = (39, 174, 96)
ORANGE = (242, 153, 74)
WHITE = (255, 255, 255)
BLACK = (24, 33, 47)
PALE_RED = (255, 243, 243)
PALE_GREEN = (241, 255, 245)
PALE_BLUE = (234, 242, 255)


def font(size, bold=False):
    candidates = [
        r'C:/Windows/Fonts/msyhbd.ttc' if bold else r'C:/Windows/Fonts/msyh.ttc',
        r'C:/Windows/Fonts/simhei.ttf',
        r'C:/Windows/Fonts/arialbd.ttf' if bold else r'C:/Windows/Fonts/arial.ttf',
    ]
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE = font(40, True)
F_SUB = font(24, False)
F_CARD_TITLE = font(28, True)
F_LABEL = font(22, True)
F_TEXT = font(22, False)
F_SMALL = font(18, False)
F_TINY = font(15, False)


def rounded_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def add_shadow_card(base, box, radius=30, fill=WHITE, outline=GRID, width=2):
    x1, y1, x2, y2 = box
    shadow = Image.new('RGBA', base.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x1+8, y1+10, x2+8, y2+10), radius=radius, fill=(0,0,0,28))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    base.alpha_composite(shadow)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def resize_cover(img, size):
    img = img.convert('RGB')
    w,h = img.size
    tw,th = size
    scale = max(tw/w, th/h)
    nw,nh = int(w*scale), int(h*scale)
    img = img.resize((nw,nh), Image.Resampling.LANCZOS)
    left = (nw-tw)//2
    top = (nh-th)//2
    return img.crop((left, top, left+tw, top+th))


def make_frame_grid(frames, size=(210,210)):
    # 2x2 grid from 4 frames
    arr = frames.detach().cpu().numpy()
    arr = np.transpose(arr, (0,2,3,1))
    cell = (size[0]//2, size[1]//2)
    canvas = Image.new('RGB', size, (245,245,245))
    for i in range(min(4, len(arr))):
        im = Image.fromarray((arr[i]*255).clip(0,255).astype('uint8'))
        im = resize_cover(im, cell)
        canvas.paste(im, ((i%2)*cell[0], (i//2)*cell[1]))
    return canvas


def make_fft_image(frames, size=(210,210)):
    spec = FrequencyBranch.frames_to_spectrum(frames[:1].unsqueeze(0))[0,0,0].detach().cpu().numpy()
    # robust normalize percentile
    lo, hi = np.percentile(spec, 2), np.percentile(spec, 99.5)
    x = np.clip((spec-lo)/(hi-lo+1e-6),0,1)
    # simple magma-like via matplotlib colormap if available
    try:
        import matplotlib.cm as cm
        rgba = (cm.get_cmap('magma')(x)*255).astype('uint8')
        im = Image.fromarray(rgba[:,:,:3])
    except Exception:
        im = Image.fromarray((x*255).astype('uint8')).convert('RGB')
    im = resize_cover(im, size)
    return im


def pick_cases(df):
    cases=[]
    defs=[
        ('Real video kept', 'Correct Real', 0, 0, GREEN),
        ('AI video detected', 'Correct Fake', 1, 1, GREEN),
        ('Missed AI video', 'False Negative', 1, 0, RED),
        ('False alarm on real', 'False Positive', 0, 1, RED),
    ]
    for zh,en,label,pred,color in defs:
        sub=df[(df.label==label)&(df.pred==pred)].copy()
        if len(sub)==0: continue
        if label==0 and pred==0:
            row=sub.sort_values('fake_prob', ascending=True).iloc[0]
        elif label==1 and pred==1:
            row=sub.sort_values('fake_prob', ascending=False).iloc[0]
        elif label==1 and pred==0:
            row=sub.sort_values('fake_prob', ascending=True).iloc[0]
        else:
            row=sub.sort_values('fake_prob', ascending=False).iloc[0]
        cases.append(dict(zh=zh,en=en,color=color,row=row))
    return cases


def draw_prob_bars(d, x, y, w, real_prob, fake_prob):
    bar_h = 28
    # labels
    d.text((x, y), 'Real', font=F_SMALL, fill=NAVY)
    d.text((x+w-72, y), f'{real_prob:.3f}', font=F_SMALL, fill=NAVY)
    yb=y+28
    d.rounded_rectangle((x, yb, x+w, yb+bar_h), radius=9, fill=(231,238,247), outline=None)
    d.rounded_rectangle((x, yb, x+max(6,int(w*real_prob)), yb+bar_h), radius=9, fill=BLUE, outline=None)
    y2=yb+54
    d.text((x, y2), 'AI-generated', font=F_SMALL, fill=RED)
    d.text((x+w-72, y2), f'{fake_prob:.3f}', font=F_SMALL, fill=RED)
    yb2=y2+28
    d.rounded_rectangle((x, yb2, x+w, yb2+bar_h), radius=9, fill=(255,225,225), outline=None)
    d.rounded_rectangle((x, yb2, x+max(6,int(w*fake_prob)), yb2+bar_h), radius=9, fill=RED, outline=None)


def draw_case(base, case, box):
    x1,y1,x2,y2=box
    d=ImageDraw.Draw(base)
    fill = PALE_GREEN if case['color']==GREEN else PALE_RED
    add_shadow_card(base, box, radius=30, fill=WHITE, outline=case['color'], width=3)
    # top ribbon
    d.rounded_rectangle((x1+24,y1+22,x1+220,y1+62), radius=16, fill=case['color'])
    d.text((x1+42,y1+30), case['en'], font=F_SMALL, fill=WHITE)
    d.text((x1+245,y1+27), case['zh'], font=F_CARD_TITLE, fill=case['color'])
    row=case['row']
    true='Fake' if int(row.label)==1 else 'Real'
    pred='Fake' if int(row.pred)==1 else 'Real'
    generator=str(row.generator)
    fake=float(row.fake_prob); real=1-fake
    # Load visual images
    path=DATA_ROOT / str(row.rel_path)
    try:
        frames=read_video_frames(path, num_frames=4, image_size=160, random_sample=False)
        grid=make_frame_grid(frames, (230,230))
        fft=make_fft_image(frames, (230,230))
    except Exception as e:
        grid=Image.new('RGB',(230,230),(230,230,230)); fft=grid.copy()
        gd=ImageDraw.Draw(grid); gd.text((20,100),'read error',font=F_SMALL,fill=RED)
    # labels and image boxes
    ix=x1+34; iy=y1+92
    for label,im,xx in [('Sampled frames',grid,ix),('FFT log-magnitude',fft,ix+285)]:
        d.text((xx,iy-30),label,font=F_LABEL,fill=NAVY)
        d.rounded_rectangle((xx-4,iy-4,xx+234,iy+234),radius=14,fill=(255,255,255),outline=GRID,width=2)
        base.paste(im,(xx,iy))
    # info panel
    px=ix+600; py=iy-12
    d.text((px,py),f'generator: {generator}',font=F_TEXT,fill=NAVY)
    d.text((px,py+38),f'true: {true}    pred: {pred}',font=F_TEXT,fill=case['color'])
    draw_prob_bars(d, px, py+86, 430, real, fake)
    # bottom diagnosis
    diag = {
      'Correct Real':'Real sample is correctly preserved; the detector is not simply biased toward AI.',
      'Correct Fake':'AI-generated sample is detected; multi-view clues support the fake decision.',
      'False Negative':'Missed detection: an AI video is classified as Real; future work should improve fake recall.',
      'False Positive':'False alarm: a real video is classified as AI; improve robustness to compression and jitter.',
    }.get(case['en'],'')
    d.text((x1+36,y2-46),diag,font=F_SMALL,fill=MUTED)


def main():
    df=pd.read_csv(PRED)
    cases=pick_cases(df)
    base=Image.new('RGBA',(W,H),BG+(255,))
    d=ImageDraw.Draw(base)
    # title
    d.text((64,42),'Demo-style Case Study',font=F_TITLE,fill=NAVY)
    d.text((64,92),'Sampled frames | FFT spectrum | probability output | error diagnosis',font=F_SUB,fill=MUTED)
    # legend
    d.rounded_rectangle((1840,48,2030,86),radius=16,fill=PALE_GREEN,outline=GREEN,width=2)
    d.text((1860,56),'Correct',font=F_SMALL,fill=GREEN)
    d.rounded_rectangle((2055,48,2245,86),radius=16,fill=PALE_RED,outline=RED,width=2)
    d.text((2075,56),'Error',font=F_SMALL,fill=RED)
    d.text((64,1138),'Note: probabilities are from medium_stf3 checkpoint; examples are selected from prediction CSV for presentation and diagnosis.',font=F_TINY,fill=MUTED)
    # Cards 2x2
    positions=[(64,150,1268,600),(1332,150,2536,600),(64,645,1268,1095),(1332,645,2536,1095)]
    for case,box in zip(cases,positions):
        draw_case(base,case,box)
    # save
    OUT.parent.mkdir(parents=True,exist_ok=True)
    base.convert('RGB').save(OUT,quality=96)
    base.convert('RGB').save(OUT_COMPAT,quality=96)
    print('[write]', OUT, OUT.stat().st_size)
    print('[replace]', OUT_COMPAT, OUT_COMPAT.stat().st_size)

if __name__=='__main__':
    main()

