# -*- coding: utf-8 -*-
"""香港屋宇署 (Buildings Department) 作业守则：列出 PDF 并抽一份条文看看结构。

BD 的 Code of Practice 是公开可下载 PDF（中英双语），结构天然是
"条文标题 + 条文正文"，可以做成规范类两字段语料。
"""

from __future__ import annotations

import re
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
PAGES = [
    "https://www.bd.gov.hk/en/resources/codes-and-references/codes-and-design-manuals/index.html",
    "https://www.bd.gov.hk/en/resources/codes-and-references/index.html",
]


def candidate_links(html: str, page: str) -> list[tuple[str, str]]:
    """Any <a> whose href or text mentions codes / manuals / pdf."""
    out = []
    for href, text in re.findall(r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>',
                                 html, flags=re.S | re.I):
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
        hay = (href + " " + title).lower()
        if any(k in hay for k in ("code", "manual", "guide", "design_",
                                  ".pdf", "standard")):
            out.append((urllib.parse.urljoin(page, href), title))
    return out


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40, context=CTX) as resp:
        return resp.read()


def main() -> int:
    links: dict[str, str] = {}
    for page in PAGES:
        try:
            html = fetch(page).decode("utf-8", "replace")
        except Exception as exc:                       # noqa: BLE001
            print(f"  page failed {page}: {exc}")
            continue
        cands = candidate_links(html, page)
        if cands:
            print(f"  {len(cands)} candidate link(s) on {page.rsplit('/', 2)[-2]}:")
            for url, title in cands[:12]:
                print(f"    - {title[:64]:<66} {url[:110]}")
        for href, text in re.findall(r'<a[^>]+href="([^"]+\.pdf)"[^>]*>(.*?)</a>',
                                     html, flags=re.S | re.I):
            url = urllib.parse.urljoin(page, href)
            title = re.sub(r"<[^>]+>", " ", text)
            title = re.sub(r"\s+", " ", title).strip()
            links.setdefault(url, title or "(no title)")
        print(f"  {page} -> {len(links)} pdf link(s) so far")
    print(f"\nfound {len(links)} PDF(s); first 25:")
    for url, title in list(links.items())[:25]:
        print(f"  {title[:70]:<72} {url.rsplit('/', 1)[-1][:60]}")
    if not links:
        return 1

    # pick a concrete-structure code if present, else the first link
    pick = next((u for u, t in links.items()
                 if "concrete" in (t + u).lower()), list(links)[0])
    print(f"\ndownloading {pick}")
    pdf = ROOT / "hkbd_sample.pdf"
    pdf.write_bytes(fetch(pick))
    print(f"  saved {pdf} ({pdf.stat().st_size:,} bytes)")

    txt = ROOT / "hkbd_sample.txt"
    try:
        subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout",
                        str(pdf), str(txt)], check=True)
    except Exception as exc:                            # noqa: BLE001
        print(f"  pdftotext failed: {exc}")
        return 1
    raw = txt.read_text(encoding="utf-8", errors="replace")
    print(f"  extracted {len(raw):,} characters")
    body = re.sub(r"[ \t]+", " ", raw)
    print("\n----- first 1,200 characters -----")
    print(body[:1200])

    # 有没有中文版？对中文期刊投稿更有用
    cn_candidates = [
        "https://www.bd.gov.hk/doc/tc/resources/codes-and-references/code-and-design-manuals/CoP_SUC2013c.pdf",
        "https://www.bd.gov.hk/doc/tc/resources/codes-and-references/code-and-design-manuals/FoundationCode2017_c.pdf",
        "https://www.bd.gov.hk/doc/en/resources/codes-and-references/code-and-design-manuals/CoP_SUC2013c.pdf",
    ]
    print("\n----- Chinese-version probe -----")
    for url in cn_candidates:
        try:
            blob = fetch(url)
            print(f"  OK  {len(blob):,} bytes  {url.rsplit('/', 1)[-1]}")
            Path(ROOT / "hkbd_cn_sample.pdf").write_bytes(blob)
            subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout",
                            str(ROOT / "hkbd_cn_sample.pdf"),
                            str(ROOT / "hkbd_cn_sample.txt")], check=False)
            cn = (ROOT / "hkbd_cn_sample.txt").read_text(
                encoding="utf-8", errors="replace")
            print("  snippet:", re.sub(r"\s+", " ", cn)[:400])
            break
        except Exception as exc:                       # noqa: BLE001
            print(f"  --  {type(exc).__name__}: {exc}  {url.rsplit('/', 1)[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
