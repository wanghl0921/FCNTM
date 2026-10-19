# -*- coding: utf-8 -*-
"""Second audit batch: official ASCE example vs our main_asce, corruption source, template journal."""
import io, os, re, zipfile, glob

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")
CODE = os.path.join(D, "FCNTM完整代码")
TMP = os.path.join(D, "_asce_template", "_extract")

# 1. extract official example from zip if not already
ex = os.path.join(TMP, "ascelike-new-main", "example", "outputs", "ascelike-new-example.pdf")
if not os.path.exists(ex):
    os.makedirs(TMP, exist_ok=True)
    z = zipfile.ZipFile(os.path.join(D, "_asce_template", "ascelike_new_main.zip"))
    z.extractall(TMP)
print("example pdf exists:", os.path.exists(ex))

# 2. find 'under its own' in py files (corruption culprit search)
print()
print("### scripts containing 'under its own':")
for f in glob.glob(os.path.join(CODE, "*.py")) + glob.glob(os.path.join(D, "*.py")):
    try:
        s = io.open(f, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    if "under its own" in s or "keeps a small lead" in s:
        print("   ", f)
        for m in re.finditer(r"under its own", s):
            i = m.start()
            print("      ...", repr(s[max(0,i-60):i+120]))

# 3. search for single-backslash-b sequences in py files (real backspace bug)
print()
print("### py files with potential \\b escape bugs (single backslash before 'b' inside quotes):")
for f in glob.glob(os.path.join(CODE, "*.py")) + glob.glob(os.path.join(D, "*.py")) + glob.glob(os.path.join(PAPER, "*.py")):
    try:
        b = open(f, "rb").read()
    except Exception:
        continue
    # byte pattern: backslash 'b' where the backslash is NOT preceded by another backslash
    hits = []
    i = 0
    while True:
        i = b.find(b"\\b", i)
        if i < 0:
            break
        if i == 0 or b[i-1] != 0x5c:
            hits.append(i)
        i += 2
    if hits:
        print("   ", os.path.basename(f), len(hits))
        for h in hits[:6]:
            print("        ", repr(b[max(0,h-40):h+40]))
