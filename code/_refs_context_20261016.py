# -*- coding: utf-8 -*-
"""Batch 7: contexts of cross-refs in main_en.tex; tool availability."""
import io, os, re, subprocess

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")

s = io.open(os.path.join(PAPER, "main_en.tex"), encoding="utf-8").read()

print("### Section 3.3 contexts")
for m in re.finditer(r"Section\s*3\.3", s):
    i = m.start()
    print("  ...", " ".join(s[max(0, i-200):i+160].split())[-330:])
    print()

print("### \\ref{sec:...} contexts")
for m in re.finditer(r"\\ref\{sec:[^}]*\}", s):
    i = m.start()
    print("  ...", " ".join(s[max(0, i-160):i+120].split())[-260:])
    print()

print("### \\eqref contexts")
for m in re.finditer(r"\\eqref\{[^}]*\}", s):
    i = m.start()
    print("  ...", " ".join(s[max(0, i-120):i+100].split())[-210:])
    print()

print("### \\ref{tab:...} contexts (first 6)")
cnt = 0
for m in re.finditer(r"\\ref\{tab:[^}]*\}", s):
    i = m.start()
    print("  ...", " ".join(s[max(0, i-100):i+80].split())[-170:])
    cnt += 1
    if cnt >= 6:
        break

print()
print("### tool availability")
for c in ["soffice", "soffice.exe", "winword", "et", "wps"]:
    r = subprocess.run(["where.exe", c], capture_output=True, text=True)
    print("  where %-12s -> %s" % (c, (r.stdout or r.stderr).strip()[:150]))

cands = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
    r"C:\Users\Administrator\AppData\Local\Kingsoft\WPS Office",
    r"C:\Program Files (x86)\Kingsoft\WPS Office",
    r"C:\Program Files\Kingsoft\WPS Office",
]
for c in cands:
    print("  exists %-70s %s" % (c, os.path.exists(c)))

pandoc = os.path.join(D, "_tools", "pandoc-3.12.1", "pandoc.exe")
r = subprocess.run([pandoc, "--version"], capture_output=True, text=True)
print("  pandoc:", r.stdout.splitlines()[0] if r.stdout else r.stderr[:100])
