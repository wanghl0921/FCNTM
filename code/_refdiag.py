# -*- coding: utf-8 -*-
import json
import re
import urllib.request

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

for pdf in ("main_en.pdf", "main_asce.pdf"):
    t = "".join(p.get_text() for p in pymupdf.open(PAPER + "\\" + pdf))
    print("==", pdf)
    for pat in ("71246-8", "26123-2", "85896-4", "Zhou", "Vangelova"):
        for m in re.finditer(pat, t):
            print("   ", repr(t[max(0, m.start() - 60):m.start() + 60]))
            break


def cr(doi):
    url = ("https://api.crossref.org/works/"
           + urllib.request.quote(doi) + "?mailto=manqp@126.com")
    req = urllib.request.Request(url, headers={"User-Agent": "refdiag/1.0"})
    with urllib.request.urlopen(req, timeout=40) as fh:
        return json.loads(fh.read().decode())["message"]


for doi in ("10.1016/j.autcon.2024.105890", "10.1016/j.autcon.2026.107220"):
    m = cr(doi)
    print("==", doi)
    print("   title:", (m.get("title") or [""])[0][:90])
    print("   authors:", [(a.get("given"), a.get("family"))
                          for a in m.get("author", [])])
    print("   volume:", m.get("volume"), "| article:", m.get("article-number"),
          "| page:", m.get("page"))
    for k in ("published", "published-print", "published-online",
              "created", "issued"):
        print("   %-16s" % k, (m.get(k) or {}).get("date-parts"))
