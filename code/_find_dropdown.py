# -*- coding: utf-8 -*-
"""Find the frame in which the item-type dropdown is open."""

import glob

import numpy as np
from PIL import Image

files = sorted(glob.glob(r"C:\Users\Administrator\Downloads\_shots\shot_*.png"))
base = np.asarray(Image.open(files[0]).convert("L"), dtype=np.int16)
h, w = base.shape
region = (slice(int(h * 0.40), int(h * 0.90)), slice(int(w * 0.0), int(w * 0.55)))
cand = []
for f in files:
    img = np.asarray(Image.open(f).convert("L"), dtype=np.int16)
    diff = np.abs(img[region] - base[region]).sum()
    cand.append((int(diff), f.split("\\")[-1]))
cand.sort(reverse=True)
for d, name in cand[:8]:
    print(d, name)
