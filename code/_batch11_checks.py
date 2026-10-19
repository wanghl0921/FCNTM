# -*- coding: utf-8 -*-
"""Batch 11: hunt newer reviews, control chars in scripts, cite/ref details, github asset text."""
import io, os, re

D = r"C:\Users\Administrator\Downloads"
CODE = os.path.join(D, "FCNTM完整代码")
PAPER = os.path.join(D, "FCNTM完整论文")

print("#" * 22, "1. recent review-like files (last 4 days)")
import datetime
cut = datetime.datetime(2026, 10, 12).timestamp()
for root, dirs, files in os.walk(D):
    if any(x in root for x in ["_tools", "FCNTM_检查点_20261014", "node_modules", "FCNTM_github\\FCNTM\\code"]):
        dirs[:] = []
        continue
    for f in files:
        p = os.path.join(root, f)
        try:
            mt = os.path.getmtime(p)
        except Exception:
            continue
        if mt < cut:
            continue
        low = f.lower()
        if ("review" in low or "审阅" in f or "复审" in f or "revie" in low):
            print("  ", datetime.datetime.fromtimestamp(mt).strftime("%m-%d %H:%M"), os.path.getsize(p), p.replace(D + os.sep, ""))

print()
print("#" * 22, "2. control chars in python scripts (root + code)")
for base in [D, CODE, os.path.join(D, "_asce_template")]:
    for f in os.listdir(base):
        if not f.endswith(".py"):
            continue
        p = os.path.join(base, f)
        if not os.path.isfile(p):
            continue
        b = open(p, "rb").read()
        bad = {}
        for byte in b:
            if byte < 32 and byte not in (9, 10, 13):
                bad[byte] = bad.get(byte, 0) + 1
        if bad:
            print("   ", p.replace(D + os.sep, ""), bad)

print()
print("#" * 22, "3. cite/ref details in main_en.tex")
s = io.open(os.path.join(PAPER, "main_en.tex"), encoding="utf-8").read()
cits = re.findall(r"\\cite(\[[^\]]*\])?\{([^}]*)\}", s)
withopt = [c for c in cits if c[0]]
print("  cites with optional args:", len(withopt))
print("  multi-key count:", len([c for c in cits if "," in c[1]]))
secs = re.findall(r"Section~\\ref\{sec:[^}]*\}", s)
print("  Section refs:", len(secs))
for m in re.finditer(r".{80}Section~\\ref\{sec:[^}]*\}.{80}", s, re.S):
    print("    ...", " ".join(m.group(0).split()))
print("  eqref:", re.findall(r"\\eqref\{[^}]*\}", s))
print("  tab refs:", len(re.findall(r"\\ref\{tab:", s)))

print()
print("#" * 22, "4. keywords line + tail of main_en.tex")
i = s.find("Keywords")
print("  keywords:", " ".join(s[i-30:i+250].split())[:280])
print("  tail:", repr(s[-120:]))

print()
print("#" * 22, "5. github assets README head + gitignore")
for name in ["README.md", "gitignore.txt", "LICENSE.txt"]:
    p = os.path.join(CODE, "github_assets", name)
    if os.path.exists(p):
        t = io.open(p, encoding="utf-8").read()
        print("-----", name, "len", len(t))
        print(t[:1200])

print()
print("#" * 22, "6. manifest paper section keys")
import json
p = os.path.join(CODE, "outputs", "RESULTS_MANIFEST.json")
doc = json.load(io.open(p, encoding="utf-8"))
print("  groups:", list(doc.keys()))
print("  paper keys:", list(doc.get("paper", {}).keys()))
print("  artifacts groups:", {k: len(v) for k, v in doc.get("artifacts", {}).items()})

print()
print("#" * 22, "7. _pkgverify_20261015 content")
pv = os.path.join(D, "_pkgverify_20261015")
if os.path.exists(pv):
    for root, dirs, files in os.walk(pv):
        for f in files[:40]:
            print("  ", os.path.join(root, f).replace(D + os.sep, ""))
