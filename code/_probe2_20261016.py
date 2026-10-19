# -*- coding: utf-8 -*-
"""Probe 2:
  A) title candidate lengths
  B) ASCE scratch build with class-native citations (no natbib, no SectionNumbers)
  C) pandoc mini test: title/author/maketitle/abstract/tag/thebibliography
"""
import io, os, re, shutil, subprocess, zipfile

D = r"C:\Users\Administrator\Downloads"
CODE = os.path.join(D, "FCNTM完整代码")
PAPER = os.path.join(D, "FCNTM完整论文")
PANDOC = os.path.join(D, "_tools", "pandoc-3.12.1", "pandoc.exe")
SCRATCH = os.path.join(CODE, "_asce_scratch")
TESTBI = os.path.join(CODE, "_docx_test2")

# ---------- A) title lengths ----------
print("#" * 20, "A. title candidates")
cands = [
    "Field-Cooperative Neural Topic Modeling of Civil Engineering Literature: Missing-Abstract Read-out",
    "Field-Cooperative Neural Topic Modeling of Civil Engineering Literature: Missing-Abstract Inference",
    "Field-Cooperative Neural Topic Modeling in Civil Engineering: Read-out and Missing-Abstract Inference",
    "Field-Cooperative Neural Topic Modeling for Civil Engineering: Read-out and Missing-Abstract Inference",
    "Field-Cooperative Neural Topic Modeling of Civil Engineering Literature with Missing-Abstract Read-out",
]
for c in cands:
    print("  %3d  %s" % (len(c), c))

# ---------- B) ASCE scratch build ----------
print()
print("#" * 20, "B. ASCE scratch build (class-native cites, no SectionNumbers)")
if os.path.exists(SCRATCH):
    shutil.rmtree(SCRATCH)
os.makedirs(SCRATCH)
for f in ["ascelike-new.cls", "ascelike-new.bst", "main_asce.bib",
          "main_asce.bbl",
          "fig_main_comparison_en.pdf", "fig_complementarity_en.pdf",
          "fig_occlusion_en.pdf", "fig_occlusion_cross_corpus_en.pdf"]:
    srcf = os.path.join(PAPER, f)
    if os.path.exists(srcf):
        shutil.copy2(srcf, SCRATCH)

s = io.open(os.path.join(PAPER, "main_asce.tex"), encoding="utf-8").read()
s2 = s.replace(r"\documentclass[Journal,letterpaper,SectionNumbers]{ascelike-new}",
               r"\documentclass[Journal,letterpaper]{ascelike-new}")
s2 = s2.replace(r"\usepackage{natbib}", "% natbib removed for probe")
s2 = re.sub(r"\\citep(\[[^\]]*\])?\{", r"\\cite{", s2)
s2 = s2.replace(r"Section~\ref{sec:readout}", "the ``Unified topic-word read-out'' subsection")
s2 = s2.replace(r"Section~\ref{sec:occl}", "the ``Field inference under partial occlusion'' subsection")
s2 = s2.replace(r"Section~\ref{sec:protocol}", "the ``Baselines, training protocols and metrics'' subsection")
io.open(os.path.join(SCRATCH, "probe_asce.tex"), "w", encoding="utf-8", newline="\n").write(s2)

def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    return r

r1 = run(["pdflatex", "-interaction=nonstopmode", "probe_asce.tex"], SCRATCH)
print("  pdflatex1 rc:", r1.returncode)
rb = run(["bibtex", "probe_asce"], SCRATCH)
print("  bibtex rc:", rb.returncode, "| out tail:", " ".join((rb.stdout or "").split())[-200:])
r2 = run(["pdflatex", "-interaction=nonstopmode", "probe_asce.tex"], SCRATCH)
r3 = run(["pdflatex", "-interaction=nonstopmode", "probe_asce.tex"], SCRATCH)
print("  pdflatex2 rc:", r2.returncode, " pdflatex3 rc:", r3.returncode)

logp = os.path.join(SCRATCH, "probe_asce.log")
if os.path.exists(logp):
    log = io.open(logp, encoding="utf-8", errors="replace").read()
    errs = re.findall(r"^! .*$", log, re.M)
    print("  errors:", errs[:6])
    undef = re.findall(r"Citation `([^']*)' on page", log)
    print("  undefined citations:", sorted(set(undef))[:10])

pdf = os.path.join(SCRATCH, "probe_asce.pdf")
if os.path.exists(pdf):
    import pymupdf
    doc = pymupdf.open(pdf)
    print("  pages:", doc.page_count)
    for pno in range(1, min(7, doc.page_count)):
        t = doc[pno].get_text()
        m = re.search(r"\([A-Z][A-Za-z]+[^)]{0,70}\d{4}[a-z]?\)", t)
        if m:
            print("  cite example p%d: %r" % (pno + 1, m.group(0)))
            break
    found = False
    for pno in range(1, min(9, doc.page_count)):
        t = doc[pno].get_text()
        for hm in re.finditer(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{5,60})$", t, re.M):
            print("  numbered heading found p%d: %r" % (pno + 1, hm.group(0).strip()))
            found = True
            break
        if found:
            break
    if not found:
        print("  no numbered headings in first pages (good for ASCE convention)")
    doc.close()

# ---------- C) pandoc mini test 2 ----------
print()
print("#" * 20, "C. pandoc mini test 2 (title/author/abstract/tag/bibliography)")
os.makedirs(TESTBI, exist_ok=True)
mini_lines = [
    r"\documentclass{article}",
    r"\usepackage{amsmath}",
    r"\title{\bfseries Probe Title Line One:\\Probe Subtitle Line Two}",
    r"\author{Hongli Wang\textsuperscript{1} and Qingpeng Man\textsuperscript{1,*}\\[6pt]",
    r"\small\textsuperscript{1}School of Civil Engineering, Harbin Institute of Technology, Harbin, China\\",
    r"\small\textsuperscript{*}Corresponding author. E-mails: 2010254508@qq.com (H.~Wang), manqp@126.com (Q.~Man)}",
    r"\date{}",
    r"\begin{document}",
    r"\maketitle",
    r"\begin{abstract}",
    r"This is the probe abstract text.",
    r"\end{abstract}",
    r"\noindent\textbf{Keywords:} alpha; beta",
    r"\section{Introduction}",
    r"Text with superscript \textsuperscript{[1,2]} and cite \cite{blei2003lda} plus \cite{yin2014gsdmm,srivastava2017avitm}.",
    r"\begin{equation}",
    r"a = b + c \tag{3}",
    r"\end{equation}",
    r"Reference to Equation~2 and Table~4 here.",
    r"\begin{thebibliography}{99}",
    r"\bibitem{blei2003lda} Blei D M, Ng A Y, Jordan M I. Latent Dirichlet allocation[J]. Journal of Machine Learning Research, 2003, 3: 993--1022.",
    r"\bibitem{yin2014gsdmm} Yin J, Wang J. A Dirichlet multinomial mixture model-based approach for short text clustering[C]//Proceedings of the 20th ACM SIGKDD, 2014: 233--242.",
    r"\end{thebibliography}",
    r"\end{document}",
    "",
]
io.open(os.path.join(TESTBI, "mini2.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(mini_lines))
outp = os.path.join(TESTBI, "mini2.docx")
if os.path.exists(outp):
    os.remove(outp)
r = subprocess.run([PANDOC, "mini2.tex", "-o", "mini2.docx", "--resource-path", PAPER],
                   cwd=TESTBI, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("  rc:", r.returncode, "| stderr:", (r.stderr or "")[:600])
if os.path.exists(outp):
    z = zipfile.ZipFile(outp)
    xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    print("  oMath:", xml.count("<m:oMath"), "| superscript:", xml.count('w:val="superscript"'))
    texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml)
    joined = " | ".join(t for t in texts if t.strip())
    print("  TEXT:", joined[:1400])
    print("  media:", [n for n in z.namelist() if "media" in n])
    z.close()
