# -*- coding: utf-8 -*-
"""2026-10-12 review: replace the last old table numbers in code comments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(name) for f in fails):
        out = text.replace("\n", "\r\n") if crlf else text
        path.write_bytes(out.encode("utf-8"))
        print(f"  patched {name}")


patch("check_paper_numbers.py", [
    ("docstring",
     "1. Every table value is **parsed out of main.tex** (tables 1--9, the\n"
     "   clean-test table and the field/redundancy tables).",
     "1. Every table value is **parsed out of main.tex** (every numbered table,\n"
     "   the clean-test table and the field/redundancy tables).", 1),
])

patch("finish_agnews_occlusion.py", [
    ("table ref",
     "     re-scores them and restores the numbers the paper quotes for Table 5",
     "     re-scores them and restores the numbers the paper quotes for the\n"
     "     whole-field occlusion table (tab:occlusion)", 1),
])

patch("run_public_missing_field.py", [
    ("table ref",
     "                # documents (the definition used in Table 3 of the paper):",
     "                # documents (the field-view definition of the paper,\n"
     "                # tab:fieldview):", 1),
])

patch("run_multiseed_occlusion.py", [
    ("table ref",
     "``--dump-state`` (it also writes the whole-field occlusion row used in\n"
     "Table 5), merging into the existing state file through ``--reuse-state``.",
     "``--dump-state`` (it also writes the whole-field occlusion row used in\n"
     "``tab:occlusion``), merging into the existing state file through\n"
     "``--reuse-state``.", 1),
])

import sys
if fails:
    print("unmatched anchors:")
    for f in fails:
        print("  ", f)
sys.exit(1 if fails else 0)
