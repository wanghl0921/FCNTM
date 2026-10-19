# -*- coding: utf-8 -*-
"""OpenAlex 的许可声明 + 本语料里"开放许可"记录占比。

要点：标题通常不受版权保护；摘要属于可受版权保护的表达，权利一般在作者/出版社。
OpenAlex 以 CC0 发布其数据集，但稳妥做法是只保留 OpenAlex 标注为开放许可
（cc0 / cc-by / cc-by-sa / public-domain 等）的记录。
"""

from __future__ import annotations

import gzip
import json
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research; mailto:fcnm-research@example.com)",
      "Accept-Encoding": "gzip"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
MAILTO = "fcnm-research@example.com"
API = "https://api.openalex.org"
SUB = "2205"          # Civil and Structural Engineering；再加 Building and Construction
YEARS = "2000-2023"
BASE_FILTER = (f"has_abstract:true,language:en,publication_year:{YEARS}"
               ",type:article|review|conference-paper")

DOC_PAGES = [
    ("OpenAlex 使用条款", "https://docs.openalex.org/how-to-use-the-api/api-overview"),
    ("OpenAlex FAQ", "https://help.openalex.org/hc/en-us/articles/24397657423127-What-license-is-OpenAlex-available-under"),
    ("OpenAlex about", "https://openalex.org/about"),
]
KEYWORDS = ["cc0", "licen", "copyright", "public domain", "reuse", "attribut"]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40, context=CTX) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw


def api(path: str, **params) -> dict:
    params.setdefault("mailto", MAILTO)
    url = f"{API}{path}?" + urllib.parse.urlencode(params, doseq=True)
    return json.loads(get(url))


def main() -> int:
    print("=" * 70)
    print("1) OpenAlex 的许可说明（网页原文摘录）")
    for name, url in DOC_PAGES:
        print(f"\n--- {name}  {url}")
        try:
            raw = get(url)
        except Exception as exc:                       # noqa: BLE001
            print(f"    FAIL {type(exc).__name__}: {exc}")
            continue
        text = None
        for enc in ("utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text,
                      flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        shown = 0
        for m in re.finditer(r"[^.]{0,220}(cc0|public domain|licen[cs]e)[^.]{0,220}\.",
                             text, re.I):
            print("    * " + m.group(0).strip()[:340])
            shown += 1
            if shown >= 4:
                break
        if not shown:
            print("    (no licence sentence found on this page)")

    print("\n" + "=" * 70)
    print("2) 本语料的许可分布（OpenAlex primary_location.license）")
    for sub in ("2205",):
        try:
            data = api("/works", filter=f"primary_topic.subfield.id:{sub},"
                                        f"{BASE_FILTER}",
                       group_by="primary_location.license")
        except Exception as exc:                       # noqa: BLE001
            print(f"    group_by failed: {exc}")
            continue
        groups = data.get("group_by", [])
        total = sum(g["count"] for g in groups)
        print(f"    subfield {sub}: {total:,} works")
        for g in groups[:14]:
            print(f"      {str(g['key']):<28} {g['count']:>8,}")

    print("\n" + "=" * 70)
    print("3) 只保留开放许可会剩多少（逐许可计数）")
    for lic in ("cc-by", "cc0", "cc-by-sa", "public-domain", "cc-by-nc"):
        try:
            d = api("/works", filter=f"primary_topic.subfield.id:2205,"
                                     f"{BASE_FILTER},primary_location.license:{lic}",
                    **{"per-page": 1})
            print(f"    {lic:<16} {d['meta']['count']:>8,}")
        except Exception as exc:                       # noqa: BLE001
            print(f"    {lic:<16} failed: {exc}")

    print("\n" + "=" * 70)
    print("4) 两个子领域合并 + 许可过滤后的可用规模")
    sub_ids = ["2205"]
    try:
        d = api("/subfields", search="building and construction", **{"per-page": 3})
        for c in d["results"][:1]:
            sub_ids.append(c["id"].rsplit("/", 1)[-1])
            print(f"    + subfield {c['display_name']} "
                  f"({c['id'].rsplit('/', 1)[-1]})")
    except Exception as exc:                           # noqa: BLE001
        print(f"    subfield lookup failed: {exc}")
    sid = "|".join(sub_ids)
    ok_lic = "cc-by|cc-by-sa|public-domain"
    for label, extra in (
            ("全部（不限许可）", ""),
            ("只要 cc-by / cc-by-sa / public-domain", f",primary_location.license:{ok_lic}"),
            ("再加 cc-by-nc（允许再发布但限非商业）",
             ",primary_location.license:cc-by|cc-by-sa|public-domain|cc-by-nc")):
        try:
            d = api("/works",
                    filter=f"primary_topic.subfield.id:{sid},{BASE_FILTER}{extra}",
                    **{"per-page": 1})
            print(f"    {label:<44} {d['meta']['count']:>9,}")
        except Exception as exc:                       # noqa: BLE001
            print(f"    {label:<44} failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
