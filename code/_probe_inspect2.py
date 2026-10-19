# -*- coding: utf-8 -*-
import re
import zipfile

z = zipfile.ZipFile(r"C:\Users\Administrator\Downloads\FCNTM完整代码\_docx_probe3\probe.docx")
xml = z.read("word/document.xml").decode("utf-8")
i = xml.find('w:val="Author"')
j = xml.find("</w:p>", i)
print("=== author paragraph ===")
print(xml[max(0, i - 60):j][:1600])
print()
print("=== equation paragraphs ===")
for m in re.finditer(r"<m:oMathPara>.*?</m:oMathPara>", xml, re.S):
    print(repr(m.group(0)[:260]))
