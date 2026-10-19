# -*- coding: utf-8 -*-
"""Assemble the open-source GitHub package (2026-10-16).

Builds ``Downloads/FCNTM_github_20261016.zip`` with a clean repository
layout::

    FCNTM/
      README.md          (written here)
      .gitignore
      LICENSE            (placeholder: licence not chosen yet)
      code/              (the reproduction code, data and result artifacts)
      docs/figures/      (rendered figures for the README)

The manuscript and the submission files are deliberately *not* part of the
repository; the README links the key figures instead.

The 11 model checkpoints (*.pt) and the large details dumps are excluded
from the repository (GitHub blocks files over 100 MB and warns over 50 MB);
the existing ``FCNTM_检查点_20261014.zip`` is meant to be attached to a
GitHub *Release* next to the repository.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

DL = Path(r"C:\Users\Administrator\Downloads")
CODE = DL / "FCNTM完整代码"
PAPER = DL / "FCNTM完整论文"
BUILD = DL / "FCNTM_github"
ZIP = DL / "FCNTM_github_20261016.zip"

BIG_DETAILS = {
    "fcntm_civil_seed53plus_details.json",
    "fcntm_github_seed53plus_details.json",
    "fcntm_agnews_seed53plus_details.json",
    "fcntm_github_final_details.json",
    "fcntm_agnews_final_details.json",
    "fcntm_civil_final_details.json",
}

PAPER_KEEP = {
    "main_en.tex", "main_en.pdf", "supplementary_en.tex",
    "supplementary_en.pdf", "main_asce.tex", "main_asce.pdf",
    "main_asce.bib", "ascelike-new.cls", "ascelike-new.bst",
    "make_paper_figures_en.py",
}


def keep_code(path: Path) -> bool:
    name = path.name
    if "__pycache__" in path.parts:
        return False
    if path.parts and path.parts[0].startswith(("_docx_build", "_docx_probe")):
        return False
    if name.startswith(("_asce_build", "_en_build", "_zh_build")):
        return False
    if name.endswith((".pyc", ".pt", ".zip")):
        return False
    if ".resume." in name:
        return False
    if name in BIG_DETAILS:
        return False
    return True


DOC_FIGURES = (
    "fig_architecture_en.png",
    "fig_seed_paired_en.png",
    "fig_main_comparison_en.png",
    "fig_occlusion_en.png",
    "fig_occlusion_cross_corpus_en.png",
    "fig_complementarity_en.png",
)


def copy_tree(src: Path, dst: Path, keep) -> None:
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        if not keep(rel):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def main() -> None:
    if BUILD.exists():
        resolved = BUILD.resolve()
        assert str(resolved).startswith(str(DL.resolve())), resolved
        shutil.rmtree(resolved)
    root = BUILD / "FCNTM"
    (root / "code").mkdir(parents=True)
    (root / "docs" / "figures").mkdir(parents=True)
    copy_tree(CODE, root / "code", keep_code)
    for name in DOC_FIGURES:
        shutil.copy2(PAPER / name, root / "docs" / "figures" / name)
    assets = Path(__file__).resolve().parent / "github_assets"
    for name, target in (("README.md", root / "README.md"),
                         ("gitignore.txt", root / ".gitignore"),
                         ("LICENSE.txt", root / "LICENSE")):
        shutil.copy2(assets / name, target)
    files = [p for p in root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    largest = max((p.stat().st_size for p in files), default=0)
    print(f"repo tree: {total/1e6:.1f} MB over {len(files)} files, "
          f"largest file {largest/1e6:.1f} MB")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(BUILD).as_posix())
    print(f"wrote {ZIP.name}: {ZIP.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
