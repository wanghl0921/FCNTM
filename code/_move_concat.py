# -*- coding: utf-8 -*-
"""One-off: move the FCNTMConcat class from the middle of models.py to its end."""

from pathlib import Path

p = Path("models.py")
lines = p.read_text(encoding="utf-8").split("\n")
start = None
for i, t in enumerate(lines):
    if t.startswith("class FCNTMConcat"):
        assert start is None, "multiple class starts"
        start = i
assert start is not None
end = None
for j in range(start + 1, len(lines)):
    if lines[j] == "    def train_step(":
        end = j
        break
assert end is not None
block = lines[start:end]
while block and block[-1].strip() == "":
    block.pop()
rest = lines[:start] + lines[end:]
while start > 1 and rest[start - 1].strip() == "" and rest[start - 2].strip() == "":
    del rest[start - 1]
    start -= 1
out = rest + ["", ""] + block + [""]
p.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
print("moved block of", len(block), "lines to end of file")
