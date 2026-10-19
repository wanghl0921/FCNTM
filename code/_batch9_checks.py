# -*- coding: utf-8 -*-
"""Batch 9: full ASCE preview read, title lengths, footnotes, appendix flow."""
import io, os, re

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")

pv = io.open(os.path.join(D, "_asce_template", "asce_preview.txt"), encoding="utf-8").read()

print("#" * 22, "A. full preview text (split in 2 chunks)")
half = len(pv) // 2
print("LEN:", len(pv))
print("---------- CHUNK 1 ----------")
print(pv[:half])
