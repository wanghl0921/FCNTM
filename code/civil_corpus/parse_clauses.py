# -*- coding: utf-8 -*-
"""把作业守则的 PDF 文本切分成"条文标题 + 条文正文"，看能否做两字段语料。"""

from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 形如 "6.3.2 Design of connections" 的条文起始行
CLAUSE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+(\S.{0,90})$")


def paragraphs(text: str) -> list[str]:
    text = text.replace("\f", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if b]


def clauses(text: str, min_words: int = 25) -> list[dict]:
    out = []
    for block in paragraphs(text):
        lines = block.split("\n")
        head = lines[0].strip()
        m = CLAUSE.match(head)
        if not m:
            continue
        body = " ".join(l.strip() for l in lines[1:]).strip()
        out.append({"num": m.group(1), "title": m.group(2).strip(),
                    "body": body})
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "hkbd_sample.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    items = [i for i in clauses(text) if len(i["body"].split()) >= 20]
    words = [len(i["body"].split()) for i in items]
    print(f"file: {path.name}  ({len(text):,} chars)")
    print(f"clauses with >=20 body words: {len(items)}")
    if words:
        print(f"body words : mean {st.mean(words):.0f}  median {st.median(words):.0f}"
              f"  min {min(words)}  max {max(words)}")
        print(f"title words: mean "
              f"{st.mean([len(i['title'].split()) for i in items]):.1f}")
        print(f"body/title : mean "
              f"{st.mean([len(i['body'].split()) / max(len(i['title'].split()), 1) for i in items]):.1f}")
    print("\n----- samples -----")
    for item in items[40:46]:
        print(f"\n[{item['num']}] {item['title']}")
        print(f"    {item['body'][:380]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
