# -*- coding: utf-8 -*-
"""Rewrite the checklist and the comments text for the ASCE external-hosting policy."""

import io

DST = r"C:\Users\Administrator\Downloads\FCNTM_submission_20261019"

README = """# JCCE (ASCE) submission package - 19 Oct 2026

**Manuscript:** Field-Cooperative Neural Topic Modeling of Civil Engineering Literature:
Missing-Abstract Inference
**Authors:** Hongli Wang and Qingpeng Man (corresponding author)

The manuscript is 30 double-spaced pages with three figures and two tables; the remaining
tables and figures are in the supplementary PDF, which is hosted externally (see below).

## ASCE policy on supplementary material (from January 5, 2026)

ASCE no longer hosts or accepts supplementary files: "Authors may include a link to these
materials in the body of their manuscript, hosted elsewhere." The manuscript therefore carries
the link in its Supplemental Materials section, and the Editorial Manager question asks for a
URL or DOI instead of a file. **Do not upload supplementary.pdf as a figure file.**

Supplementary URL:
https://github.com/wanghl0921/FCNTM/releases/download/v1.0.1/supplementary.pdf

## Files to upload

| File | Item type | Description |
|---|---|---|
| `manuscript.pdf` | Manuscript | Manuscript |
| `cover_letter.pdf` (or `.docx`) | Cover Letter | Cover Letter |
| `figures/Fig1_architecture.pdf` | Figure | Fig. 1 |
| `figures/Fig2_seed_paired.pdf` | Figure | Fig. 2 |
| `figures/Fig3_occlusion.pdf` | Figure | Fig. 3 |

Do **not** upload `optional_supplementary/supplementary.pdf`: it is already available at the
URL above, and the system has no supplementary item type.

## Questions to answer

- **Supplemental materials:** Yes - paste the URL above into the "DOI or URL" field.
- **Data availability:** the GitHub repository https://github.com/wanghl0921/FCNTM
  (release v1.0.0: checkpoints and per-seed details).
- **Comments to the Editor:** paste `optional_supplementary/comments_to_editor.txt`.

## Before submitting

- [ ] `manuscript.pdf` is 30 pages, has line numbers, and its Supplemental Materials section
      shows the URL.
- [ ] Three figures uploaded separately.
- [ ] Cover letter attached.
- [ ] Supplement URL entered in the system question.
- [ ] AI-use disclosure completed.
- [ ] Funding entered (National Key R&D Program of China, 2024YFC3809900, sub-project
      2024YFC3809904-03) and matching the Acknowledgments section.
"""

COMMENTS = """The supplementary material (Tables S1-S20, Figures S1-S3, the ablation details and
the training details of the baselines) is available at
https://github.com/wanghl0921/FCNTM/releases/download/v1.0.1/supplementary.pdf, in line with the
ASCE policy that supplementary files are hosted externally; the same link appears in the
Supplemental Materials section of the manuscript.

The manuscript PDF is 30 double-spaced pages including references, tables and figures (the ASCE
limit). The code, the fixed corpus splits, the trained model checkpoints and the per-seed results
are available at https://github.com/wanghl0921/FCNTM (release v1.0.0).
"""

io.open(DST + r"\README_upload_checklist.md", "w", encoding="utf-8").write(README)
io.open(DST + r"\optional_supplementary\comments_to_editor.txt", "w",
        encoding="utf-8").write(COMMENTS)
print("checklist and comments rewritten")
