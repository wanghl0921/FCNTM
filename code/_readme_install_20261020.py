# -*- coding: utf-8 -*-
"""Add the PyTorch CPU index and the embedding-download note to the README."""

import io
import sys

OLD_BLOCK = "\n".join([
    "```bash",
    "cd code",
    "pip install -r requirements.txt            # Python 3.13, CPU-only torch",
    "python tests_review_fixes.py               # 33 numeric regression checks",
])

NEW_BLOCK = "\n".join([
    "```bash",
    "cd code",
    "python -m pip install -r requirements.txt  # CPU torch; requirements.txt",
    "                                           # carries the PyTorch CPU index",
    "python tests_review_fixes.py               # 33 numeric regression checks",
])

OLD_NOTE = "\n".join([
    "place the checkpoint",
    "files from the Release assets under `code/outputs/` and",
    "`code/outputs/rerun/`; the sentence-embedding model used by the contextual",
    "baselines is downloaded automatically on first use.",
])

NEW_NOTE = "\n".join([
    "place the checkpoint",
    "files from the Release assets under `code/outputs/` and",
    "`code/outputs/rerun/`. The sentence-embedding model used by the contextual",
    "baselines (`sentence-transformers/all-MiniLM-L6-v2`, revision pinned in",
    "`requirements.txt`) is downloaded automatically on first use (about 90 MB)",
    "and then cached by `huggingface_hub`; set `FCNTM_HF_OFFLINE=1` to force the",
    "cached copy on a machine without network access.",
])


def apply(path: str) -> None:
    s = io.open(path, encoding="utf-8").read()
    for old, new in ((OLD_BLOCK, NEW_BLOCK), (OLD_NOTE, NEW_NOTE)):
        if s.count(old) != 1:
            raise SystemExit("%s: expected one match for %r..."
                             % (path, old[:40]))
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(s)
    print("updated", path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        apply(p)
