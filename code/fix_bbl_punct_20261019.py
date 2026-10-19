# -*- coding: utf-8 -*-
"""Remove the duplicated period after titles that end in '?' or '!'.

The ascelike bst appends a period inside the closing quote of a title
(``...when?.''``); an interrogative title must keep a single mark.
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
path = PAPER + r"\main_asce.bbl"
s = io.open(path, encoding="utf-8").read()
before = s.count("?.'") + s.count("!.'")
s = s.replace("?.''", "?''").replace("!.'", "!'")
# an in-press article needs the comma before the note
s = s.replace("}\\ in press", "}, in press")
io.open(path, "w", encoding="utf-8").write(s)
print("main_asce.bbl: fixed", before, "title punctuation(s)")
