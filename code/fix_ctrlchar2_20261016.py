# -*- coding: utf-8 -*-
"""Second pass: fix U+000C (form feed) where '\\fcntm' was mangled to '\\x0c' + 'cntm'."""
import io, os, glob

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

files = sorted(glob.glob(os.path.join(PAPER, "*.tex")))
total = 0
for p in files:
    s = io.open(p, encoding="utf-8").read()
    ctrl = sorted(set(ch for ch in s if ord(ch) < 32 and ch not in "\n\r\t"))
    bad = "".join(ctrl)
    before = {repr(c): s.count(c) for c in ctrl}
    fixed = s
    # fix mangled \fcntm (form feed + 'cntm')
    n1 = fixed.count("\x0ccntm")
    fixed = fixed.replace("\x0ccntm", "\\fcntm")
    # fix mangled \beta (backspace + 'eta') - safety net in case it reappears
    n2 = fixed.count("\x08eta")
    fixed = fixed.replace("\x08eta", "\\beta")
    # remove any other stray control chars entirely (report them)
    rest = sorted(set(ch for ch in fixed if ord(ch) < 32 and ch not in "\n\r\t"))
    if rest:
        for c in rest:
            print("REMAINING", repr(c), "in", os.path.basename(p), "count", fixed.count(c))
        fixed = "".join(ch for ch in fixed if ord(ch) >= 32 or ch in "\n\r\t")
    if fixed != s:
        io.open(p, "w", encoding="utf-8", newline="\n").write(fixed)
        total += n1 + n2
        print("fixed in %s: \\fcntm x%d, \\beta x%d" % (os.path.basename(p), n1, n2))
    else:
        print("no control chars:", os.path.basename(p))

print("TOTAL fixes:", total)
