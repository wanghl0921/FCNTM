# -*- coding: utf-8 -*-
"""Repair control characters that slipped in from a Python string literal."""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"


def main() -> None:
    for name in ("main_en.tex", "main_asce.tex", "supplementary_en.tex"):
        path = PAPER + "\\" + name
        try:
            s = io.open(path, encoding="utf-8").read()
        except FileNotFoundError:
            continue
        before = sum(s.count(c) for c in (chr(8), chr(12)))
        s = s.replace("$" + chr(8) + "eta$", "$" + chr(92) + "beta$")
        s = s.replace(chr(12) + "cntm", chr(92) + "fcntm")
        s = s.replace(chr(8), "")
        io.open(path, "w", encoding="utf-8").write(s)
        after = sum(s.count(c) for c in (chr(8), chr(12)))
        print("%-24s control chars %d -> %d" % (name, before, after))


if __name__ == "__main__":
    main()
