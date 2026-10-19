# -*- coding: utf-8 -*-
import re
import zipfile

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

doc = pymupdf.open(PAPER + r"\main_asce.pdf")
t = "".join(p.get_text() for p in doc)
print("ASCE pdf: PRACTICAL APPLICATIONS:",
      "PRACTICAL APPLICATIONS" in t,
      "| DATA AVAILABILITY:", "DATA AVAILABILITY" in t)
print("ASCE pdf cite sample:", bool(re.search(r"\(Blei et al\.?\s*2003\)", t)))
print("ASCE pdf pages:", doc.page_count)

z = zipfile.ZipFile(PAPER + r"\FC-NTM_manuscript.docx")
xml = z.read("word/document.xml").decode("utf-8")
sup = re.search(
    r'<w:r>(?:(?!</w:r>).)*w:val="superscript"(?:(?!</w:r>).)*</w:r>', xml)
print("en docx first superscript run:", sup.group(0)[:200] if sup else None)

plain = re.sub(r"<[^>]+>", "", xml)
print("en docx multi-cite sample:", "(8,9,10,11)" in plain
      or "[8,9,10,11]" in plain)

a = zipfile.ZipFile(PAPER + r"\FC-NTM_manuscript_ASCE.docx")
axml = re.sub(r"<[^>]+>", "", a.read("word/document.xml").decode("utf-8"))
j = axml.find("(Blei et al. 2003)")
print("ASCE docx in-text cite at", j, ":",
      repr(axml[max(0, j - 70):j + 40]) if j >= 0 else None)
