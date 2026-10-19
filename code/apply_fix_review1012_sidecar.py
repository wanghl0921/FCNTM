# -*- coding: utf-8 -*-
"""2026-10-12: the field-view sidecar must describe the file, not the call.

A regeneration can merge several invocations into one CSV (LDA/NMF/BTM in one,
ProdLDA in another); the sidecar used to list only the last invocation's
corpora/models, which under-describes a 3-corpus file.  The sidecar is now
derived from the merged frame itself.
"""
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
        path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        print(f"  patched {name}")


patch("baseline_fieldview.py", [
    ("helper",
     '''def main() -> int:''',
     '''def file_identity(frame, k: int, seeds) -> dict:
    """Identity of the *file*: every corpus protocol and fit it contains.

    A regeneration may merge several invocations into one CSV (LDA/NMF/BTM in
    one run, ProdLDA in another), so the sidecar has to describe the merged
    table rather than the invocation that happened to write it last.
    """
    corpora = sorted(str(c) for c in frame["dataset"].unique())
    eval_ids = {}
    for corpus in corpora:
        ids = sorted(str(i) for i in
                     frame.loc[frame["dataset"] == corpus, "eval_id"].unique())
        eval_ids[corpus] = ids[0] if len(ids) == 1 else ids
    return RI.identity({
        "script": "baseline_fieldview.main", "k": int(k),
        "seeds": [int(s) for s in seeds], "corpora": corpora,
        "models": sorted(str(m) for m in frame["model"].unique()),
        "eval_ids": eval_ids,
        "fit_ids": sorted({str(f) for f in frame["fit_id"].unique()}),
        "code": RI.code_version()})


def main() -> int:''',
     1),
    ("write sidecar",
     '''    RI.write(out, RI.identity({
        "script": "baseline_fieldview.main", "k": args.k, "seeds": seeds,
        "corpora": sorted(eval_ids), "models": models, "eval_ids": eval_ids,
        "fit_ids": sorted({str(f) for f in frame["fit_id"].unique()}),
        "code": RI.code_version()}))''',
     '''    RI.write(out, file_identity(frame, args.k, seeds))''',
     1),
])

import sys
if fails:
    print("unmatched anchors:")
    for f in fails:
        print("  ", f)
sys.exit(1 if fails else 0)
