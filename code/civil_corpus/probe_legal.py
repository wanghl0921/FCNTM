# -*- coding: utf-8 -*-
"""检查规范文本与政府网站的使用条款：能不能拿来做语料、能不能再发布。

两部分：
  1) 已下载的规范 PDF 文本里有没有版权/授权声明（本地检索，不需要网络）；
  2) 抓港府/屋宇署的版权与使用条款页面，打印含关键词的段落。
"""

from __future__ import annotations

import gzip
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

KEYWORDS = ["copyright", "reproduce", "reproduction", "permission",
            "all rights reserved", "©", "licence", "license",
            "版權", "知識產權", "知識産權", "許可", "不得", "合理使用"]

LOCAL_TEXTS = ["hkbd_sample.txt", "hkbd_cn_sample.txt"]

PAGES = [
    ("HKSAR 政府版权", "https://www.gov.hk/en/info/copyright.htm"),
    ("HKSAR 版权(about)", "https://www.gov.hk/en/about/copyright.htm"),
    ("知识产权署 版权", "https://www.ipd.gov.hk/en/copyright/index.html"),
    ("屋宇署 使用条款", "https://www.bd.gov.hk/en/terms-of-use.html"),
    ("屋宇署 版权", "https://www.bd.gov.hk/en/copyright.html"),
]

DISCOVER_FROM = [
    "https://www.bd.gov.hk/en/index.html",
    "https://www.gov.hk/en/index.html",
]


def fetch(url: str) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return resp.status, resp.headers.get("Content-Type", ""), raw


def to_text(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text,
                  flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def hits(text: str, limit: int = 6) -> list[str]:
    out = []
    for para in re.split(r"(?<=[.;。；])\s+|\n", text):
        para = para.strip()
        if 40 <= len(para) <= 600 and any(k.lower() in para.lower()
                                          for k in KEYWORDS):
            out.append(para)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    print("=" * 70)
    print("1) 规范 PDF 自身的版权声明")
    for name in LOCAL_TEXTS:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found = hits(text)
        print(f"\n--- {name} ({len(text):,} chars) ---")
        print("  no copyright/permission sentence found" if not found
              else "\n".join("  * " + h[:300] for h in found))

    print("\n" + "=" * 70)
    print("2) 从首页发现版权/条款页面")
    discovered: list[tuple[str, str]] = []
    for page in DISCOVER_FROM:
        try:
            _, _, raw = fetch(page)
            html = raw.decode("utf-8", "replace")
            for href, text in re.findall(
                    r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>', html,
                    flags=re.S | re.I):
                title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
                hay = (href + " " + title).lower()
                if any(k in hay for k in ("copyright", "terms", "disclaimer",
                                          "notices", "privacy", "版權", "條款")):
                    url = href if href.startswith("http") else page.rsplit("/", 1)[0] + "/" + href.lstrip("/")
                    if url not in [u for u, _ in discovered]:
                        discovered.append((url, title))
        except Exception as exc:                       # noqa: BLE001
            print(f"    discover failed {page}: {exc}")
    for url, title in discovered[:12]:
        print(f"    - {title[:50]:<52} {url[:110]}")

    print("\n" + "=" * 70)
    print("3) 港府/屋宇署 网站条款")
    targets = PAGES + [(t or u, u) for u, t in discovered[:6]]
    seen = set()
    for name, url in targets:
        if url in seen:
            continue
        seen.add(url)
        print(f"\n=== {name}\n    {url}")
        try:
            status, ctype, raw = fetch(url)
            text = to_text(raw)
            print(f"    HTTP {status} {ctype} {len(raw):,} bytes")
            found = hits(text, limit=5)
            print("    (no keyword paragraph)" if not found
                  else "\n".join("    * " + h[:400] for h in found))
        except urllib.error.HTTPError as exc:
            print(f"    HTTP {exc.code} {exc.reason}")
        except Exception as exc:                       # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
