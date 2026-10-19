# -*- coding: utf-8 -*-
"""Build the ASCE submission version of the English paper.

Uses the official ASCE LaTeX class ``ascelike-new`` (the class shipped with
the ASCE Overleaf template) and converts the working draft ``main_en.tex``
into ``main_asce.tex`` plus a BibTeX file ``main_asce.bib``:

  * document class ``[Journal,letterpaper]`` (double-spaced ASCE journal
    manuscript, unnumbered word headings per the ASCE convention),
  * ASCE-style abstract (single paragraph, no mathematics) and
    --\\KeyWords{--},
  * author-year citations via the class's own commands and
    ascelike-new.bst (no natbib, matching the official ASCE example),
  * textual cross-references to the read-out, occlusion and protocol
    subsections (sections themselves are unnumbered),
  * Data Availability Statement and Supplemental Materials sections,
  * the 50 free-text \\bibitem entries parsed into BibTeX records.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(r"C:\Users\Administrator\Downloads\FCNTM完整论文")
SRC = PAPER / "main_en.tex"

ABSTRACT = r"""Title--abstract records are a core form of civil engineering literature, and part of most
collections is available with titles only. This paper studies how the two fields of such a record
should be modelled together, and what can still be inferred when the abstract is missing. We
present FC-NTM, a field-cooperative neural topic model that builds one conditional variational
stream per field, shares a single topic decoder across the streams, and handles partial and
complete field absence with cross-field reconstruction, a consistency term and field-level word
masking; a single trained model yields a native beta read-out and a class TF-IDF read-out under
one evaluation vocabulary. On civil engineering title-abstract records from OpenAlex (6000/2000),
FC-NTM attains the highest coherence among the evaluated methods and read-out configurations, and
a ten-seed paired comparison against BTM favours it in all ten seeds (plus 0.049 class TF-IDF
read-out and plus 0.061 native beta read-out, a raw-parameter read-out whose batch-normalised
counterpart is smaller). Title-only routing reaches 0.66 accuracy against OpenAlex primary-topic labels on the subset
covered by the fifteen selected topics (42.9\% of the test split); these labels are assigned
automatically and serve as an external reference rather than expert-annotated ground truth, and
the task is a routing aid complementing, rather than replacing, supervised classification. A same-budget concatenated-field control is close on
every corpus, so the reported gains pertain to the complete modelling configuration, and the
occlusion experiments characterise inference stability under missing-field conditions when read
alongside absolute prediction loss and topic agreement. The results establish a corpus-dependent
empirical case for field-cooperative topic modelling and identify the conditions that require
further validation. Cross-domain evaluations and read-out protocols are reported in the paper and
supplement, and per-seed results, fixed data splits and hyper-parameters are released with the
code."""

KEYWORDS = ("topic model; field cooperation; neural variational inference; "
            "fielded documents with short summary fields; class TF--IDF "
            "read-out; complementarity measures")

# ASCE reference style abbreviates journal names; the bst does not carry an
# abbreviation table, so the bib file must hold the short forms.
JOURNAL_ABBREV = {
    "Journal of Machine Learning Research": "J. Mach. Learn. Res.",
    "Automation in Construction": "Autom. Constr.",
    "Journal of Computing in Civil Engineering": "J. Comput. Civ. Eng.",
    "ACM Transactions on Information Systems": "ACM Trans. Inf. Syst.",
    "Artificial Intelligence Review": "Artif. Intell. Rev.",
    "Information Sciences": "Inf. Sci.",
    "Journal of Statistical Software": "J. Stat. Softw.",
    "Knowledge-Based Systems": "Knowl.-Based Syst.",
    "The Computer Journal": "Comput. J.",
    "Transactions of the Association for Computational Linguistics":
        "Trans. Assoc. Comput. Linguist.",
}

# The bst lower-cases titles into sentence case; protect proper nouns and
# acronyms by wrapping them in braces.
TITLE_PROTECT = (
    "Dirichlet", "Bayesian", "Gibbs", "Markov", "Gaussian", "Lipschitz",
    "BERT", "LDA", "NMF", "ETM", "CTM", "WordNet", "GitHub", "Top2Vec",
    "TF-IDF", "NVDM", "AVITM", "GSDMM", "BTM", "CRNTM", "ProdLDA",
    "CombinedTM", "OpenAlex", "NeurIPS", "BERTopic",
)


def protect_title(text: str) -> str:
    for term in TITLE_PROTECT:
        text = re.sub(r"(?<![A-Za-z{}])" + re.escape(term) + r"(?![A-Za-z{}])",
                      lambda m: "{" + m.group(0) + "}", text)
    return text

PREAMBLE = r"""\documentclass[Journal,letterpaper]{ascelike-new}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage[figurename=Fig.,labelfont=bf,labelsep=period]{caption}
\usepackage{url}
\usepackage[colorlinks=true,citecolor=red,linkcolor=black]{hyperref}

\newcommand{\fcntm}{FC-NTM}

\NameTag{Wang, \today}
"""

FRONT = r"""\begin{document}

\title{Field-Cooperative Neural Topic Modeling of Civil Engineering Literature:
Missing-Abstract Inference}

\author[1]{Hongli Wang}
\author[1]{Qingpeng Man}

\affil[1]{School of Civil Engineering, Harbin Institute of Technology, Harbin, China.
E-mails: 2010254508@qq.com (H.~Wang); manqp@126.com (Q.~Man, corresponding author)}

\maketitle

\begin{abstract}
%ASCE_ABSTRACT%
\end{abstract}

\KeyWords{%KEYWORDS%}
"""

EXTRA_SECTIONS = r"""\section{Data Availability Statement}

The code, the fixed corpus splits, the trained model checkpoints and the per-seed experimental
results that support this study accompany the submission and are released in a public repository
(the permanent repository URL is inserted at submission). The experiments can be reproduced with
the released configurations and evaluation scripts. Access to the third-party source corpora
remains subject to the terms and licences of the original providers.

\section{Practical Applications}

Engineering libraries and design offices accumulate far more title--abstract records than any team
can read one by one, and a large share of those records circulate with titles only. The method
presented here, FC-NTM, treats the title and the abstract of a record as two cooperating views of
the same subject: it learns one topic space for both fields and can still assign topics when the
abstract is missing. On the civil engineering corpus, FC-NTM achieved the highest topic coherence
among the evaluated methods. Its advantage over BTM was supported by ten paired random-seed
comparisons, while the comparisons with the other baselines followed the evaluation protocols
reported in the paper. A companion routing task shows
that titles alone can point to the right subject category for about two-thirds of the records
that carry an automatic category label, which supports browsing and screening rather than
replacing expert cataloguing. Because one trained model serves both a title-only and a
full-record reading, an organisation can index new records immediately and refine the index
later, when the abstract arrives. Collections that keep a short summary field beside a longer
descriptive field are the natural next setting in which to test the same procedure.

\section{Supplemental Materials}

The supplementary material contains the auxiliary experiments and controls, the data and metric
notes, and the case-study tables cited in the text.

"""

# The ASCE manuscript keeps the tables that carry the main claims; the
# ancillary ablations, sweeps and secondary comparisons move to the
# supplementary material (which numbers them S8--S13).
MOVED_TO_SUPPLEMENT = (
    "tab:conditional", "tab:fieldview-baselines", "tab:fusion",
    "tab:core-ablation", "tab:weight", "tab:occlusion",
)

SUPPLEMENT_POINTER = r"""The component ablations (the three-step fusion ablation and the
core-component ablation), the sensitivity sweep over the fusion-reconstruction weight, the
whole-field occlusion diagnostic, the field-redundancy measures and the field-restricted read-out
comparison are reported in the supplementary material (Tables S8--S13).

"""


def drop_tables(body: str) -> str:
    blocks = list(re.finditer(r"\\begin\{table\}.*?\\end\{table\}\n\n?",
                              body, re.S))
    drop = []
    for label in MOVED_TO_SUPPLEMENT:
        hits = [b for b in blocks if ("\\label{%s}" % label) in b.group(0)]
        if len(hits) != 1:
            print("  warning: %s matches %d table blocks" % (label, len(hits)))
            continue
        drop.append(hits[0])
    for b in sorted(drop, key=lambda m: m.start(), reverse=True):
        body = body[:b.start()] + body[b.end():]
    return body


PARTICLES = ("de", "van", "von", "der", "den", "la", "le", "di", "da",
             "del", "della", "dos", "das", "ten", "ter", "st", "bin", "ibn",
             "al")


def fmt_authors(text: str) -> str:
    parts = [p.strip() for p in text.strip().rstrip(".").split(",")]
    out = []
    for p in parts:
        if p.lower() in ("et al", "et al."):
            out.append("others")
            continue
        toks = p.split()
        if not toks:
            continue
        # leading particles belong to the surname, whether lower case
        # ("van der Maaten L J P") or capitalised ("De Wolf C"):
        # -> "van der Maaten, L. J. P." / "De Wolf, C."
        k = 0
        while k < len(toks) - 2 and (
                toks[k][0].islower()
                or toks[k].strip(".").lower() in PARTICLES):
            k += 1
        surname = " ".join(toks[:k + 1])
        initials = " ".join(i if i.endswith(".") else i + "." for i in toks[k + 1:])
        out.append(f"{surname}, {initials}" if initials else surname)
    return " and ".join(out)


def escape_bib(text: str) -> str:
    return text.replace("&", r"\&")


def parse_entry(key: str, text: str) -> str:
    text = " ".join(text.split())
    doi = None
    m = re.search(r"DOI:\s*(\S+?)\.?$", text)
    if m:
        # a DOI is a literal string: LaTeX-escaped underscores are unescaped
        doi = m.group(1).rstrip(".")
        if doi.startswith("\\url{") and doi.endswith("}"):
            doi = doi[5:-1]
        doi = doi.replace("\\_", "_")
        text = text[:m.start()].strip()
    # the year is the first ", YYYY" token after the venue; taking the last
    # four digits would pick up page numbers such as "2031--2040"
    year = None
    m = re.search(r",\s*(19\d\d|20[0-2]\d)\b", text)
    if m:
        year = m.group(1)
    else:
        years = re.findall(r"\b(\d{4})\b", text)
        if years:
            year = years[-1]
    tag = None
    m = re.search(r"\[([A-Za-z/]+)\]", text)
    if m:
        tag = m.group(1)
    title = None
    authors = None
    rest = ""
    head, sep, tail = text.partition(f"[{tag}]") if tag else (text, "", "")
    if sep:
        if ". " in head:
            authors, title = head.split(". ", 1)
        else:
            authors, title = "", head
        rest = tail
    else:
        authors = ""
        title = head
    fields = []
    if authors:
        fields.append(f"  author = {{{fmt_authors(authors)}}}")
    if title:
        fields.append(f"  title = {{{escape_bib(protect_title(title.strip().rstrip('.')))}}}")
    if year:
        fields.append(f"  year = {{{year}}}")
    kind = "misc"
    if tag in ("J", "J/OL"):
        venue, _, tail2 = rest.partition(f", {year}")
        venue = venue.lstrip(". ").strip()
        if venue.lower().startswith("arxiv"):
            kind = "article"
            fields.append(f"  journal = {{arXiv preprint {venue}}}")
        else:
            kind = "article"
            venue = JOURNAL_ABBREV.get(venue, venue)
            fields.append(f"  journal = {{{escape_bib(venue)}}}")
            m = re.match(r"\s*,?\s*([\w ]+?)\s*\((\d+)\)\s*:\s*(.+?)\.?\s*$",
                         tail2)
            m2 = re.match(r"\s*,?\s*(.+?)\s*:\s*(.+?)\.?\s*$", tail2)
            if re.search(r"in press", tail2, re.I):
                fields.append("  note = {in press}")
            elif m:
                fields.append(f"  volume = {{{m.group(1)}}}")
                fields.append(f"  number = {{{m.group(2)}}}")
                fields.append(f"  pages = {{{m.group(3)}}}")
            elif m2:
                vol = m2.group(1).strip()
                pages = m2.group(2).strip()
                if vol.replace(" ", "").isdigit():
                    fields.append(f"  volume = {{{vol}}}")
                else:
                    fields.append(f"  volume = {{{vol}}}")
                fields.append(f"  pages = {{{pages}}}")
            elif tail2.strip(" ,."):
                fields.append(f"  pages = {{{tail2.strip(' ,.')}}}")
    elif tag == "C":
        if rest.startswith("//"):
            rest = rest[2:]
        if rest.startswith("//"):
            rest = rest[2:]
        venue, _, tail2 = rest.partition(f", {year}")
        venue = venue.lstrip(". ").strip()
        pages = tail2.lstrip(": ").strip().rstrip(".")
        if venue:
            kind = "inproceedings"
            fields.append(f"  booktitle = {{{escape_bib(venue)}}}")
        if pages and any(ch.isdigit() for ch in pages):
            fields.append(f"  pages = {{{pages}}}")
    elif tag == "M":
        if rest.startswith("//"):
            rest = rest[2:]
        base = rest.partition(f", {year}")[0].lstrip(". ").strip()
        booktitle = base.split(". ")[0].strip()
        if booktitle:
            kind = "incollection"
            fields.append(f"  booktitle = {{{escape_bib(booktitle)}}}")
        if "Springer" in base:
            fields.append("  publisher = {Springer}")
        tail2 = rest.partition(f", {year}")[2] if year in rest else ""
        pages = tail2.lstrip(": ").strip().rstrip(".")
        if pages and any(ch.isdigit() for ch in pages):
            fields.append(f"  pages = {{{pages}}}")
    else:
        arx = re.search(r"arXiv:([0-9]{4}\.[0-9]{4,5})", text)
        if arx:
            kind = "article"
            fields.append(f"  journal = {{arXiv preprint arXiv:{arx.group(1)}}}")
        elif rest.lstrip(". "):
            # electronic resources (datasets, repositories): the class bst
            # renders organisation, URL and note for @misc entries, but not
            # the howpublished field, so the parts are split out here
            m = re.search(r"\\url\{([^}]*)\}", rest)
            if m:
                url = m.group(1).strip()
                head = rest[:m.start()].strip(" .,;")
                org, _, extra = head.partition(",")
                if org.strip():
                    fields.append(f"  organization = {{{escape_bib(org.strip())}}}")
                extra = extra.strip(" .,;")
                if extra:
                    fields.append(f"  note = {{{escape_bib(extra)}}}")
                fields.append(f"  url = {{{url}}}")
            else:
                fields.append(f"  howpublished = {{{escape_bib(rest.lstrip('. '))}}}")
    if doi:
        fields.append(f"  doi = {{{doi}}}")
    body = ",\n".join(fields)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main() -> None:
    tex = SRC.read_text(encoding="utf-8")
    body = tex[tex.index(r"\section{Introduction}"):]
    bib_start = body.index(r"\begin{thebibliography}")
    references = body[bib_start:]
    body = body[:bib_start]
    entries = []
    for part in references.split("\\bibitem{")[1:]:
        key, _, rest = part.partition("}")
        text = rest.split("\\end{thebibliography}")[0].strip()
        entries.append((key.strip(), text))
    bib_lines = []
    for key, text in entries:
        text = text.split("\n\\bibitem")[0].strip()
        bib_lines.append(parse_entry(key, text))
    bib_text = "\n".join(bib_lines)
    (PAPER / "main_asce.bib").write_text(bib_text, encoding="utf-8")
    body = body.replace(r"\section{Conclusion}", r"\section{Summary and Conclusions}")
    body = body.replace(r"Section~\ref{sec:readout}",
                        "the ``Unified topic-word read-out'' subsection")
    body = body.replace(r"Section~\ref{sec:occl}",
                        "the ``Field inference under partial occlusion'' subsection")
    body = body.replace(r"Section~\ref{sec:protocol}",
                        "the ``Baselines, training protocols and metrics'' subsection")
    body = drop_tables(body)
    # a table that moved to the supplement is cited by its S-number
    for number, label in enumerate(MOVED_TO_SUPPLEMENT, start=8):
        body = body.replace("Table~\\ref{%s}" % label, "Table~S%d" % number)
        body = body.replace("\\ref{%s}" % label, "Table~S%d" % number)
    body = body.replace(r"\section{Discussion and Limitations}",
                        SUPPLEMENT_POINTER + r"\section{Discussion and Limitations}")
    out = (PREAMBLE + "\n" + FRONT
           .replace("%ASCE_ABSTRACT%", ABSTRACT)
           .replace("%KEYWORDS%", KEYWORDS)
           + "\n" + body
           + EXTRA_SECTIONS
           + "% bibliography style is set by ascelike-new.cls\n"
           + "\\bibliography{main_asce}\n"
           + "\n\\end{document}\n")
    (PAPER / "main_asce.tex").write_text(out, encoding="utf-8")
    print("wrote main_asce.tex and main_asce.bib with", len(entries),
          "references")


if __name__ == "__main__":
    main()
