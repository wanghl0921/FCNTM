# -*- coding: utf-8 -*-
"""Deeper checks on the generated manuscript docx files."""
import re
import sys
import zipfile

for path in sys.argv[1:]:
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8")
    txt = re.sub(r"<w:p [^>]*>|<w:p>", "\n", xml)
    txt = re.sub(r"<[^>]+>", "", txt)
    print("=" * 70)
    print(path)
    tabs = sorted(set(re.findall(r"Table \d+:", txt)),
                  key=lambda s: int(re.findall(r"\d+", s)[0]))
    figs = sorted(set(re.findall(r"Figure \d+:", txt)),
                  key=lambda s: int(re.findall(r"\d+", s)[0]))
    print("table captions:", tabs)
    print("figure captions:", figs)
    print("question marks in refs:", txt.count("??"))
    heads = re.findall(r"\n(\d+(?:\.\d+)?)\s*\n?([A-Z][A-Za-z ,\-]{3,60})\n",
                       txt)
    print("numbered headings:", heads[:12])
    for probe in ("References", "Practical Applications",
                  "Missing-Abstract", "Keywords"):
        print("  %-24s %s" % (probe, probe in txt))
    i = txt.find("References")
    print("  refs head:", repr(txt[i:i + 260]))
    m = re.search(r"\n\[1\] (\S{0,20})", txt)
    print("  first numbered ref:", m.group(0).strip() if m else None)
