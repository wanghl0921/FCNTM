# -*- coding: utf-8 -*-
"""One-off: finish moving the misplaced FCNTMConcat methods.

The first move left part of the class inside ProdLDAModel.  This script
cuts the stray methods out of ProdLDAModel and appends them to the
FCNTMConcat class at the end of the file.
"""

from pathlib import Path

p = Path("models.py")
lines = p.read_text(encoding="utf-8").split("\n")

prod = next(i for i, t in enumerate(lines) if t.startswith("class ProdLDAModel"))
# ProdLDA's own get_beta (first one after the class start)
gb = next(i for i in range(prod, len(lines))
          if lines[i] == "    def get_beta(self) -> np.ndarray:")
start = None
for i in range(gb + 2, len(lines)):
    if lines[i] == "    def train_step(" and \
            lines[i + 2].strip() == "x_norm: torch.Tensor,":
        start = i
        break
assert start is not None
# the stray block runs up to (not including) the blank line before this method
end = start
while lines[end - 1].strip() == "":
    end -= 1
block = lines[gb + 2:end]
assert block and block[0].strip() == "" and \
    block[1] == "    def train_step(", block[:3]

rest = lines[:gb + 2] + ["", ""] + lines[end:]

# append the block to FCNTMConcat (last class of the file)
tail = rest
assert any(t.startswith("class FCNTMConcat") for t in tail)
out = tail + [""] + [ln for ln in block if True]
text = "\n".join(out).rstrip("\n") + "\n"
p.write_text(text, encoding="utf-8")
print("moved", len(block), "lines; first:", block[0].strip()[:40],
      "| last:", block[-1].strip()[:40])
