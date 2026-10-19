# -*- coding: utf-8 -*-
"""Batch audit: soften script targets, ref report, official ASCE sample, tex refs/captions."""
import io, os, re, zipfile

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")
CODE = os.path.join(D, "FCNTM完整代码")

print("#" * 30, "1. soften_tones script: files touched & rules")
s = io.open(os.path.join(CODE, "soften_tones_20261016.py"), encoding="utf-8").read()
print(s[:2200])

print()
print("#" * 30, "2. ref_check report (head)")
p = os.path.join(CODE, "outputs", "rerun", "ref_check_20261016.txt")
if os.path.exists(p):
    t = io.open(p, encoding="utf-8").read()
    print("len", len(t))
    print(t[:1500])
print()

print("#" * 30, "3. official ASCE sample from ascelike_new_main.zip")
zp = os.path.join(D, "_asce_template", "ascelike_new_main.zip")
if os.path.exists(zp):
    z = zipfile.ZipFile(zp)
    for n in z.namelist():
        print("   ", n)
