# -*- coding: utf-8 -*-
"""Probe: mini pandoc latex->docx behaviour for booktabs tables, multirow, equations, superscript, cite."""
import io, os, re, subprocess, zipfile

D = r"C:\Users\Administrator\Downloads"
CODE = os.path.join(D, "FCNTM完整代码")
TESTDIR = os.path.join(CODE, "_docx_test")
PANDOC = os.path.join(D, "_tools", "pandoc-3.12.1", "pandoc.exe")

os.makedirs(TESTDIR, exist_ok=True)

mini = r"""\documentclass{article}
\usepackage{amsmath,amssymb}
\begin{document}
Text with superscript cite \textsuperscript{[1,2]} end.

\begin{table}[t]
\centering
\caption{Probe caption}
\label{tab:probe}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llcc@{}}
\toprule
Model & Read-out & $C_V$ & NPMI \\
\midrule
\multirow{2}{*}{FC-NTM} & $\beta$ & 0.721$\pm$0.016 & 0.266 \\
 & Class TF--IDF & 0.700 & 0.235 \\
\bottomrule
\end{tabular}
\end{table}

\begin{equation}
q_f(z_f\mid x_f)=\mathcal N\big(\mu_f(x_f),\operatorname{diag}(\sigma_f^2(x_f))\big)
\label{eq:probe}
\end{equation}

Flattened: \begin{tabular}{llcc} A & B & C & D \\ 1 & 2 & 3 & 4 \end{tabular}

Header-span table: \begin{tabular}{lcccc} \multirow{2}{*}{$\lambda_f$} & \multicolumn{2}{c}{$\beta$ read-out} & \multicolumn{2}{c}{Class TF--IDF} \\ & $C_V$ & NPMI & $C_V$ & NPMI \\ 0 & 0.721 & 0.266 & 0.700 & 0.235 \\ \end{tabular}
\end{document}
"""

src = os.path.join(TESTDIR, "probe.tex")
io.open(src, "w", encoding="utf-8", newline="\n").write(mini)

out = os.path.join(TESTDIR, "probe.docx")
if os.path.exists(out):
    os.remove(out)
r = subprocess.run([PANDOC, "probe.tex", "-o", "probe.docx"], cwd=TESTDIR,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("pandoc rc:", r.returncode)
print("stderr:", (r.stderr or "")[:2000])
print("stdout:", (r.stdout or "")[:500])

if os.path.exists(out):
    z = zipfile.ZipFile(out)
    names = z.namelist()
    print("entries:", names)
    xml = z.read("word/document.xml").decode("utf-8", errors="replace")
    print("oMath count:", xml.count("<m:oMath"))
    print("superscript count:", xml.count('w:val="superscript"'))
    for token in ["multirow", "shortstack", "multicolumn", "cite", "ref", "\\", "$"]:
        n = xml.count(token)
        print("raw token %r count: %d" % (token, n))
    # show some text runs
    texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml)
    joined = " | ".join(t for t in texts if t.strip())
    print("TEXT:", joined[:1500])
    z.close()
else:
    print("NO DOCX PRODUCED")

print()
print("#" * 20, "results_manifest verify internals")
s = io.open(os.path.join(CODE, "results_manifest.py"), encoding="utf-8").read()
i = s.find("def verify")
print(s[i:i+2600] if i >= 0 else "verify fn not found")
print()
print("#" * 20, "make_github keep_paper internals")
s = io.open(os.path.join(CODE, "make_github_package_20261016.py"), encoding="utf-8").read()
i = s.find("def keep_paper")
print(s[i:i+2200] if i >= 0 else "keep_paper not found")
