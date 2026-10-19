# -*- coding: utf-8 -*-
"""Reconnaissance: which public civil-engineering *document* sources are reachable?

Tested in one pass so a single network approval covers everything:
  1. 国家标准全文公开系统 (openstd.samr.gov.cn)
  2. 全国公共资源交易平台 (deal.ggzy.gov.cn) 招标公告
  3. 中国招标投标公共服务平台 (cebpubservice.com)
  4. 住建部 (mohurd.gov.cn) 标准公告
  5. UFGS 工程规范索引 (wbdg.org, 美国军方工程规范, 公开)
  6. 香港屋宇署 Code of Practice (bd.gov.hk)

For each it prints: HTTP status, content type, byte length, and a text snippet.
"""

from __future__ import annotations

import gzip
import io
import re
import ssl
import sys
import urllib.error
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

TARGETS = [
    ("国标全文公开 搜索页",
     "https://openstd.samr.gov.cn/bzgk/gb/std_list?p.p1=0&p.p2=%E6%B7%B7%E5%87%9D%E5%9C%9F"),
    ("国标全文公开 详情页(需hcno)", "https://openstd.samr.gov.cn/bzgk/gb/std_list"),
    ("全国公共资源交易平台 交易公告接口",
     "http://deal.ggzy.gov.cn/ds/deal/dealList_find.jsp"),
    ("中国招标投标公共服务平台", "https://www.cebpubservice.com/"),
    ("住建部 政策法规/标准", "https://www.mohurd.gov.cn/gongkai/zhengce/index.html"),
    ("UFGS 规范索引 (WBDG)", "https://www.wbdg.org/ffc/dod/unified-facilities-guide-specifications-ufgs"),
    ("香港屋宇署 作业守则", "https://www.bd.gov.hk/en/resources/codes-and-references/index.html"),
]


def fetch(url: str, data: bytes | None = None, ctype: str | None = None):
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": UA,
        "Accept": "*/*",
        "Accept-Encoding": "gzip",
        **({"Content-Type": ctype} if ctype else {}),
    })
    with urllib.request.urlopen(req, timeout=30, context=CTX) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return resp.status, resp.headers.get("Content-Type", ""), raw


def snippet(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text,
                  flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:320]


def main() -> int:
    for name, url in TARGETS:
        print(f"\n=== {name}\n    {url}")
        try:
            status, ctype, raw = fetch(url)
            print(f"    HTTP {status}  {ctype}  {len(raw):,} bytes")
            print(f"    {snippet(raw)}")
        except urllib.error.HTTPError as exc:
            body = exc.read()[:200]
            print(f"    HTTP {exc.code} {exc.reason}  ({body[:120]!r})")
        except Exception as exc:                       # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
