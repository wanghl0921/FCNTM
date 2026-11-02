# -*- coding: utf-8 -*-
"""Tiny search helper: query DuckDuckGo HTML and print result titles/links."""

import html
import re
import sys
import urllib.parse
import urllib.request


def search(q: str, n: int = 10):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 Chrome/120 Safari/537.36"})
    text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    out = []
    for block in text.split('class="result results_links')[:n]:
        m = re.search(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                      block, re.S)
        if not m:
            m = re.search(r'<a[^>]*href="([^"]+)"[^>]*class="result__a"[^>]*>'
                          r'(.*?)</a>', block, re.S)
        if not m:
            continue
        href, title = m.group(1), m.group(2)
        uddg = re.search(r"uddg=([^&]+)", href)
        if uddg:
            href = urllib.parse.unquote(uddg.group(1))
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        snippet = html.unescape(re.sub("<[^>]+>", "", sn.group(1))) if sn else ""
        out.append((html.unescape(re.sub("<[^>]+>", "", title)).strip(), href,
                    " ".join(snippet.split())[:260]))
    return out


if __name__ == "__main__":
    for q in sys.argv[1:]:
        print("=" * 10, q)
        try:
            for t, h, s in search(q):
                print("-", t)
                print("  ", h[:150])
                if s:
                    print("  ", s)
        except Exception as exc:  # noqa: BLE001
            print("ERR", exc)
