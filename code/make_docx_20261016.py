# -*- coding: utf-8 -*-
"""Build Word (.docx) versions of the FC-NTM manuscript (2026-10-16).

Three variants are produced from the LaTeX sources:

  FC-NTM_manuscript.docx       English working version: numbered sections,
                               superscript [n] citations, numbered reference
                               list in ASCE entry format
  FC-NTM_manuscript_ASCE.docx  ASCE layout: unnumbered word headings,
                               author-year citations, alphabetical ASCE
                               reference list
  FC-NTM_manuscript_zh.docx    Chinese version: superscript [n] citations

Equations are converted to OMML (Office Math) by pandoc, i.e. they are
native equations that WPS/Word can edit; no equation is inserted as an
image.  Figures are embedded as PNG pictures from the paper folder.

Run:  python -X utf8 -B make_docx_20261016.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

DL = Path(r"C:\Users\Administrator\Downloads")
PAPER = DL / "FCNTM完整论文"
BUILD = DL / "FCNTM完整代码" / "_docx_build"
PANDOC = DL / "_tools" / "pandoc-3.12.1" / "pandoc.exe"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write(p: Path, t: str) -> None:
    p.write_text(t, encoding="utf-8")


def match_brace(s: str, i: int) -> int:
    """s[i] == '{'; return index just after the matching '}'."""
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


def parse_aux(path: Path) -> dict:
    if not path.exists():
        return {}
    s = read(path)
    return {k: v for k, v, _ in
            re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", s)}


def parse_thebibliography(tex: str):
    """Return [(key, entry_text), ...] in file order."""
    start = tex.index(r"\begin{thebibliography}")
    end = tex.index(r"\end{thebibliography}")
    block = tex[start:end]
    parts = block.split("\\bibitem{")[1:]
    out = []
    for part in parts:
        key, _, rest = part.partition("}")
        out.append((key.strip(), rest.strip()))
    return out


BIBITEM_RE = re.compile(
    r"\\bibitem\[\\protect\\citeauthoryear\{[^}]*\}"
    r"\{((?:[^{}]|\{[^{}]*\})*)\}\{([^}]*)\}\]\{([^}]*)\}")


def parse_bbl(path: Path):
    """Return (order, key->short-author, key->year, key->entry-text)."""
    s = read(path)
    order, short, year, text = [], {}, {}, {}
    items = list(BIBITEM_RE.finditer(s))
    for n, m in enumerate(items):
        a, y, key = m.group(1), m.group(2), m.group(3)
        stop = items[n + 1].start() if n + 1 < len(items) else s.index(
            "\\end{thebibliography}")
        body = s[m.end():stop]
        order.append(key)
        short[key] = a.replace("~", " ").replace("\\@", "").strip()
        year[key] = y
        text[key] = clean_bbl(body)
    return order, short, year, text


def clean_bbl(t: str) -> str:
    t = t.replace("\\newblock", " ")
    t = t.replace("{\\em ", "\\emph{")
    t = t.replace("\\@", "")
    t = t.replace("$<$", "<").replace("$>$", ">")
    t = t.replace("~", " ")
    t = t.replace("\\ ", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def number_equations(t: str, count: list) -> str:
    def repl(m):
        count[0] += 1
        return "\\begin{equation}%s\\qquad(%d)\\end{equation}" % (
            m.group(1), count[0])
    return re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", repl, t,
                  flags=re.S)


def number_captions(t: str, env: str, word: str) -> str:
    out, pos, n = [], 0, 0
    pat = re.compile(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (env, env), re.S)
    for m in pat.finditer(t):
        n += 1
        body = m.group(1)
        cm = re.search(r"\\caption\{", body)
        if cm:
            close = match_brace(body, cm.end() - 1)
            body = (body[:cm.end()] + "%s %d: " % (word, n)
                    + body[cm.end():close] + body[close:])
        out.append(t[pos:m.start()])
        out.append("\\begin{%s}%s\\end{%s}" % (env, body, env))
        pos = m.end()
    out.append(t[pos:])
    return "".join(out)


def resolve_refs(t: str, aux: dict) -> str:
    def val(key):
        return aux.get(key, "??")
    t = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + val(m.group(1)) + ")", t)
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: val(m.group(1)), t)
    return t


def fix_figure_paths(t: str) -> str:
    t = t.replace("_en.pdf", "_en.png")
    t = re.sub(r"(fig_[a-z_]+)\.pdf", r"\1.png", t)
    return t


def flatten_shortstack(t: str) -> str:
    """pandoc drops tables containing \\shortstack; flatten it to one line."""
    while True:
        i = t.find("\\shortstack")
        if i < 0:
            return t
        j = t.index("{", i)
        end = match_brace(t, j)
        parts = [p.strip() for p in re.split(r"\\\\", t[j + 1:end - 1])]
        flat = (parts[0] + ": " + "; ".join(parts[1:])
                if len(parts) > 2 else "; ".join(parts))
        t = t[:i] + flat + t[end:]


def strip_preamble_class(t: str, cls_line: str) -> str:
    return t.replace(cls_line, "\\documentclass{article}")


def run_pandoc(pre: Path, out: Path, resource: list, number_sections: bool):
    cmd = [str(PANDOC), str(pre), "-f", "latex", "-t", "docx",
           "--resource-path=" + ";".join(str(p) for p in resource),
           "-o", str(out)]
    if number_sections:
        cmd.append("--number-sections")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise RuntimeError("pandoc failed: " + (r.stderr or "")[-2000:])
    return (r.stderr or "").strip()


def inspect_docx(path: Path) -> None:
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8")
    media = [n for n in z.namelist() if "/media/" in n]
    txt = re.sub(r"<[^>]+>", "", xml)
    print("   %s: %.1f MB | oMath=%d | images=%d | tables=%d | superscripts=%d"
          % (path.name, path.stat().st_size / 1e6,
             xml.count("<m:oMath>"), len(media), xml.count("<w:tbl>"),
             xml.count("w:vertAlign")))
    print("   leftovers: cite=%d ref=%d eqref=%d fcntm=%d backslash=%d"
          % (txt.count("\\cite"), txt.count("\\ref"), txt.count("\\eqref"),
             txt.count("\\fcntm"), txt.count("\\")))


# --------------------------------------------------------------------------
# variant builders
# --------------------------------------------------------------------------
def build_en() -> None:
    print("[1/3] English numbered version (superscript citations)")
    tex = read(PAPER / "main_en.tex")
    tex = tex.replace("\\newcommand{\\fcntm}{FC-NTM}", "")
    entries = parse_thebibliography(tex)
    num = {k: i + 1 for i, (k, _) in enumerate(entries)}
    aux = parse_aux(PAPER / "main_en.aux")
    _, _, _, bbl_text = parse_bbl(PAPER / "main_asce.bbl")

    body = tex[:tex.index(r"\begin{thebibliography}")]
    body = strip_preamble_class(body, "\\documentclass[11pt]{article}")
    body = fix_figure_paths(body)
    body = flatten_shortstack(body)
    body = number_captions(body, "table", "Table")
    body = number_captions(body, "figure", "Figure")
    eqn = [0]
    body = number_equations(body, eqn)
    body = resolve_refs(body, aux)
    body = body.replace("\\fcntm{}", "FC-NTM").replace("\\fcntm", "FC-NTM")

    def repl_cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        return "\\textsuperscript{[%s]}" % ",".join(
            str(num.get(k, "?")) for k in keys)

    body = re.sub(r"\\cite\{([^}]*)\}", repl_cite, body)

    refs = [r"\section*{References}", ""]
    for i, (key, orig) in enumerate(entries, 1):
        txt = bbl_text.get(key) or clean_bbl(orig)
        refs.append("[%d] %s" % (i, txt))
        refs.append("")
    out = body + "\n".join(refs) + "\n\\end{document}\n"
    pre = BUILD / "main_en_pre.tex"
    write(pre, out)
    dst = PAPER / "FC-NTM_manuscript.docx"
    msg = run_pandoc(pre, dst, [PAPER, BUILD], number_sections=True)
    if msg:
        print("   pandoc:", msg[:300])
    inspect_docx(dst)
    print("   cites=%d equations=%d refs=%d"
          % (len(re.findall(r"\\textsuperscript\{\[", out)), eqn[0],
             len(entries)))


def build_asce() -> None:
    print("[2/3] ASCE version (author-year citations)")
    tex = read(PAPER / "main_asce.tex")
    tex = tex.replace("\\newcommand{\\fcntm}{FC-NTM}", "")
    _, short, year, bbl_text = parse_bbl(PAPER / "main_asce.bbl")
    order = list(bbl_text.keys())
    aux = parse_aux(PAPER / "main_asce.aux")

    body = tex[:tex.index(r"\bibliography{main_asce}")]
    body = strip_preamble_class(
        body, "\\documentclass[Journal,letterpaper]{ascelike-new}")
    body = body.replace("\\NameTag{Wang, \\today}\n", "")
    # rebuild the author block into plain LaTeX that pandoc understands
    m = re.search(r"\\title\{(.*?)\}\n\n(\\author\[1\]\{.*?\})\n\n"
                  r"(\\affil\[1\]\{(.*?)\})\n", body, re.S)
    if m:
        title, affil = m.group(1), m.group(4)
        auth = ("Hongli Wang\\textsuperscript{1} and "
                "Qingpeng Man\\textsuperscript{1,*}\\\\[4pt]\n"
                "\\textsuperscript{1}" + affil)
        body = body[:m.start(2)] + "\\author{" + auth + "}" + body[m.end(2):]
        body = body.replace(m.group(3), "")
    kw = re.search(r"\\KeyWords\{(.*?)\}\n", body, re.S)
    if kw:
        body = body.replace(
            kw.group(0),
            "\\noindent\\textbf{Keywords:} " + " ".join(kw.group(1).split())
            + "\n")
    body = fix_figure_paths(body)
    body = flatten_shortstack(body)
    body = number_captions(body, "table", "Table")
    body = number_captions(body, "figure", "Figure")
    eqn = [0]
    body = number_equations(body, eqn)
    body = resolve_refs(body, aux)
    body = body.replace("\\fcntm{}", "FC-NTM").replace("\\fcntm", "FC-NTM")

    def repl_cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        items = ["%s %s" % (short.get(k, "?"), year.get(k, "?"))
                 for k in keys]
        return "(" + "; ".join(items) + ")"

    body = re.sub(r"\\cite\{([^}]*)\}", repl_cite, body)

    refs = [r"\section*{References}", ""]
    for key in order:
        refs.append(bbl_text[key])
        refs.append("")
    out = body + "\n".join(refs) + "\n\\end{document}\n"
    pre = BUILD / "main_asce_pre.tex"
    write(pre, out)
    dst = PAPER / "FC-NTM_manuscript_ASCE.docx"
    msg = run_pandoc(pre, dst, [PAPER, BUILD], number_sections=False)
    if msg:
        print("   pandoc:", msg[:300])
    inspect_docx(dst)
    print("   equations=%d refs=%d" % (eqn[0], len(order)))


def render_zh_tikz() -> bool:
    """Render the Chinese architecture figure (TikZ) to PNG."""
    tex = read(PAPER / "main.tex")
    i = tex.index(r"\begin{tikzpicture}")
    j = tex.index(r"\end{tikzpicture}") + len(r"\end{tikzpicture}")
    tikz = tex[i:j]
    stand = BUILD / "_zh_fig.tex"
    write(stand, "\n".join([
        r"\documentclass[border=3pt]{standalone}",
        r"\usepackage{ctex}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc}",
        r"\begin{document}", tikz, r"\end{document}", ""]))
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "_zh_fig.tex"],
                       cwd=BUILD, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    pdf = BUILD / "_zh_fig.pdf"
    if not pdf.exists():
        print("   tikz render failed:", (r.stdout or "")[-500:])
        return False
    import pymupdf
    doc = pymupdf.open(pdf)
    pix = doc[0].get_pixmap(dpi=300)
    pix.save(PAPER / "_zh_arch_fig.png")
    doc.close()
    return True


def build_zh() -> None:
    print("[3/3] Chinese version (superscript citations)")
    tex = read(PAPER / "main.tex")
    tex = tex.replace("\\newcommand{\\fcntm}{FC-NTM}", "")
    entries = parse_thebibliography(tex)
    num = {k: i + 1 for i, (k, _) in enumerate(entries)}
    aux = parse_aux(PAPER / "main.aux")

    body = tex[:tex.index(r"\begin{thebibliography}")]
    body = strip_preamble_class(body, "\\documentclass[11pt]{ctexart}")
    body = fix_figure_paths(body)
    body = flatten_shortstack(body)
    if render_zh_tikz():
        i = body.index(r"\begin{tikzpicture}")
        j = body.index(r"\end{tikzpicture}") + len(r"\end{tikzpicture}")
        body = (body[:i] + "\\includegraphics[width=0.98\\textwidth]"
                "{_zh_arch_fig.png}" + body[j:])
    body = number_captions(body, "table", "表")
    body = number_captions(body, "figure", "图")
    eqn = [0]
    body = number_equations(body, eqn)
    body = resolve_refs(body, aux)
    body = body.replace("\\fcntm{}", "FC-NTM").replace("\\fcntm", "FC-NTM")
    body = body.replace("\\textsuperscript{*}", "*")

    def repl_cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        return "\\textsuperscript{[%s]}" % ",".join(
            str(num.get(k, "?")) for k in keys)

    body = re.sub(r"\\cite\{([^}]*)\}", repl_cite, body)

    refs = ["\\section*{参考文献}", ""]
    for i, (key, orig) in enumerate(entries, 1):
        refs.append("[%d] %s" % (i, clean_bbl(orig)))
        refs.append("")
    out = body + "\n".join(refs) + "\n\\end{document}\n"
    pre = BUILD / "main_zh_pre.tex"
    write(pre, out)
    dst = PAPER / "FC-NTM_manuscript_zh.docx"
    msg = run_pandoc(pre, dst, [PAPER, BUILD], number_sections=True)
    if msg:
        print("   pandoc:", msg[:300])
    inspect_docx(dst)
    print("   equations=%d refs=%d" % (eqn[0], len(entries)))


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    build_en()
    build_asce()
    build_zh()


if __name__ == "__main__":
    main()
