# -*- coding: utf-8 -*-
"""UFGS（美国军方统一设施技术规格书）：可下载性 + 条文结构 + 版权状态。

UFGS 由 USACE / NAVFAC / AFCEC 编制，属美国联邦政府雇员作品，在美国不受版权保护
(17 U.S.C. §105)，因此可以合法再发布——这是它相对香港规范/国标的关键优势。
"""

from __future__ import annotations

import gzip
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://www.wbdg.org/FFC/DOD/UFGS/UFGS%20{}.pdf"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept-Encoding": "gzip"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# 土木/结构/市政相关的高频章节号（MasterFormat）
CANDIDATES = [
    "01 11 00", "01 33 00", "02 41 00", "02 82 00",
    "03 10 00", "03 20 00", "03 30 00", "03 40 00", "03 50 00",
    "04 20 00", "05 12 00", "05 50 00",
    "31 00 00", "31 11 00", "31 23 00", "31 23 16", "31 31 16", "31 62 00",
    "32 11 23", "32 13 13", "32 31 13", "32 92 00",
    "33 11 00", "33 40 00",
    "35 20 00", "35 31 00",
]


def get(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def main() -> int:
    found = []
    for num in CANDIDATES:
        url = BASE.format(num.replace(" ", "%20"))
        code, raw = get(url)
        ok = code == 200 and raw[:4] == b"%PDF"
        print(f"  {num}  HTTP {code}  {len(raw):,} bytes  {'PDF' if ok else ''}")
        if ok:
            found.append((num, url, raw))
    print(f"\n{len(found)}/{len(CANDIDATES)} sections downloadable")
    if not found:
        return 1

    num, url, raw = found[0]
    pdf = ROOT / f"ufgs_{num.replace(' ', '')}.pdf"
    pdf.write_bytes(raw)
    txt = pdf.with_suffix(".txt")
    subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), str(txt)],
                   check=False)
    text = txt.read_text(encoding="utf-8", errors="replace")
    print(f"\n=== sample: UFGS {num} ({len(text):,} chars) ===")
    print(re.sub(r"[ \t]+", " ", text[:900]))
    for kw in ("public domain", "copyright", "©"):
        for m in re.finditer(kw, text, re.I):
            print(f"\n  [{kw}] ...{re.sub(chr(10), ' ', text[max(0, m.start()-160):m.start()+160])}...")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
