# -*- coding: utf-8 -*-
"""Inspect a .docx produced by pandoc: equations, tables, images, key text."""
import sys
import zipfile

path = sys.argv[1]
probes = sys.argv[2:]
z = zipfile.ZipFile(path)
names = z.namelist()
xml = z.read("word/document.xml").decode("utf-8")
print("== %s ==" % path)
print("parts:", len(names), "| media:", [n for n in names if "/media/" in n])
print("oMath (display+inline) :", xml.count("<m:oMath>"))
print("oMathPara              :", xml.count("<m:oMathPara>"))
print("superscript runs       :", xml.count("w:vertAlign"))
print("tables                 :", xml.count("<w:tbl>"))
print("drawings (images)      :", xml.count("<w:drawing>"))
print("italic runs            :", xml.count('w:val="italic"'))
for needle in ("Title", "Heading1", "Heading2", "Abstract", "References"):
    print("style %-10s     :" % needle, ('w:val="%s"' % needle) in xml)
for p in probes:
    print("probe %-28r :" % p, xml.count(p))
    if xml.count(p) and len(p) > 3:
        i = xml.find(p)
        print("   ctx:", repr(xml[max(0, i - 160):i + 160]))
