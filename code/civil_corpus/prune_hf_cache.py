# -*- coding: utf-8 -*-
"""删除 HF 缓存里对 PyTorch 推理无用的权重格式（tf / flax / onnx / openvino）。

保留：config.json、tokenizer*、vocab.txt、sentence_bert_config.json、
      1_Pooling/*、modules.json、model.safetensors、pytorch_model.bin。
被删的只是同一模型的其它框架副本（可随时重新下载）。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

HUB = Path(r"C:\Users\Administrator\.cache\huggingface\hub")
ONLY_MODEL = "models--sentence-transformers--all-MiniLM-L6-v2"

DROP_RE = re.compile(r"(^|[\\/])(tf_model|flax_model|rust_model|openvino)|"
                     r"\.(h5|msgpack|onnx|ot|xml)$|"
                     r"(^|[\\/])onnx[\\/]", re.I)


def main() -> int:
    root = HUB / ONLY_MODEL / "snapshots"
    if not root.exists():
        print("no snapshot dir; nothing to do")
        return 0
    keep_blobs: set[str] = set()
    drop_entries: list[tuple[Path, Path, int]] = []      # link, blob, size
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            link = Path(dirpath) / name
            rel = str(link.relative_to(root))
            if not link.is_symlink():
                keep_blobs.add(str(link))
                continue
            target = Path(os.path.realpath(link))
            if DROP_RE.search(rel):
                size = target.stat().st_size if target.exists() else 0
                drop_entries.append((link, target, size))
            else:
                keep_blobs.add(str(target))
    # never delete a blob that some kept link still points at
    drop_entries = [d for d in drop_entries if str(d[1]) not in keep_blobs]
    total = sum(d[2] for d in drop_entries)
    print(f"files to drop: {len(drop_entries)}  ({total / 1024 / 1024:,.1f} MB)")
    for link, _blob, size in drop_entries[:12]:
        print(f"   {size / 1024 / 1024:7.1f} MB  {link.relative_to(root)}")
    if len(drop_entries) > 12:
        print(f"   ... and {len(drop_entries) - 12} more")
    if "--apply" not in sys.argv:
        print("\n(dry run; pass --apply to delete)")
        return 0
    freed = 0
    for link, blob, size in drop_entries:
        try:
            link.unlink()
            freed += size
            if blob.exists() and str(blob) not in keep_blobs:
                blob.unlink()
        except OSError as exc:
            print(f"   skip {link.name}: {exc}")
    print(f"\nfreed {freed / 1024 / 1024:,.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
