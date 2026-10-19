# -*- coding: utf-8 -*-
"""Read delivery zip script fully + results_manifest usage + practical applications check."""
import io, os, re

D = r"C:\Users\Administrator\Downloads"
CODE = os.path.join(D, "FCNTM完整代码")
PAPER = os.path.join(D, "FCNTM完整论文")
EXT = os.path.join(D, "_asce_template", "_extract", "ascelike-new-main")

print("#" * 20, "make_fcntm_zips_20261014.py (full)")
print(io.open(os.path.join(D, "make_fcntm_zips_20261014.py"), encoding="utf-8").read())

print("#" * 20, "results_manifest.py head")
s = io.open(os.path.join(CODE, "results_manifest.py"), encoding="utf-8").read()
print(s[:3500])

print("#" * 20, "check_english_numbers.py head")
s = io.open(os.path.join(CODE, "check_english_numbers.py"), encoding="utf-8").read()
print(s[:2000])

print("#" * 20, "grep Practical in ascelike class/template files")
for f in [os.path.join(PAPER, "ascelike-new.cls"),
          os.path.join(EXT, "template.tex"),
          os.path.join(EXT, "original", "ascexmpl-new.tex")]:
    if os.path.exists(f):
        t = io.open(f, encoding="utf-8", errors="replace").read()
        hits = [m.start() for m in re.finditer(r"[Pp]ractical", t)]
        print("  %s : %d hits" % (os.path.basename(f), len(hits)))
        for h in hits[:6]:
            print("     ", " ".join(t[max(0, h-80):h+120].split())[:200])
