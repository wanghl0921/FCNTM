# -*- coding: utf-8 -*-
"""Rewrite the submission checklist for the trimmed 3-figure manuscript."""

import io

PATH = (r"C:\Users\Administrator\Downloads\FCNTM_submission_20261019"
        r"\README_upload_checklist.md")

TEXT = """# JCCE (ASCE) submission package - 19 Oct 2026

**Manuscript:** Field-Cooperative Neural Topic Modeling of Civil Engineering Literature:
Missing-Abstract Inference
**Authors:** Hongli Wang and Qingpeng Man (corresponding author)
**Code and data:** https://github.com/wanghl0921/FCNTM (release v1.0.0)

The manuscript is 30 double-spaced pages (ASCE limit) and carries three figures and two
tables; the remaining tables and figures are in the supplementary PDF.

## Files and the item type to choose in Editorial Manager

| File | Item type | Description to enter |
|---|---|---|
| `manuscript.pdf` | Manuscript | Manuscript |
| `cover_letter.pdf` (or `.docx`) | Cover Letter | Cover Letter |
| `figures/Fig1_architecture.pdf` | Figure | Fig. 1 |
| `figures/Fig2_seed_paired.pdf` | Figure | Fig. 2 |
| `figures/Fig3_occlusion.pdf` | Figure | Fig. 3 |
| `optional_supplementary/supplementary.pdf` | Figure (see note) | Supplementary Material (PDF) |

**Note on the supplement.** The JCCE Editorial Manager site has no "Supplemental Material"
item type; the list is Manuscript, Cover Letter, Table, Figure, Figure Captions List,
Graphical Abstract, Permissions, Response to Editors/Reviewers Comments, Track Changes
Version, Change of Corresponding Author Form and Author Byline Change Form. Upload the
supplement as **Figure** with the description above and paste the text of
`optional_supplementary/comments_to_editor.txt` into the **Comments** step. The manuscript
cites the supplement in 15 places (Tables S8-S20, Figures S1-S3, ablation and training
details), so attaching it avoids broken references.

## Upload order

1. `manuscript.pdf` (Manuscript)
2. `Fig. 1` ... `Fig. 3` (Figure)
3. `supplementary.pdf` (Figure, description "Supplementary Material (PDF)")
4. Cover letter (item type Cover Letter)

Use **Update File Order** if the system reorders the list.

## What the system will ask

- **Article type:** choose the technical-paper option the journal offers.
- **Manuscript file:** PDF at initial submission (ASCE produces the review PDF in Editorial
  Manager; Word/source files are required at revision, not now).
- **Suggested reviewers:** optional; keep three names and e-mail addresses ready.
- **Comments to the Editor:** paste the contribution statement and
  `optional_supplementary/comments_to_editor.txt`.
- **PDF approval:** after "Proceed", check the built PDF; the supplement should appear in the
  attachment list with its description.

## Before clicking Submit

- [ ] `manuscript.pdf` is 30 pages and carries line numbers (ASCE requirement).
- [ ] The three figure files are uploaded separately.
- [ ] `supplementary.pdf` uploaded with the description "Supplementary Material (PDF)".
- [ ] Cover letter attached.
- [ ] Repository and release links in the Data Availability Statement resolve.
- [ ] The LaTeX/Word sources in `source/` are kept for the revision stage.

## Practical notes

- First decisions at JCCE typically take about 6-12 weeks; reports of several months are
  common, so a polite status query after ~12 weeks is reasonable.
- Reviewers often split between a methodological and an engineering viewpoint.
- At revision, check that every response file appears in the built PDF: a reported pitfall is
  that response attachments fail to reach reviewers.
- All figures use line style and marker as well as colour, so they remain legible in
  black-and-white print.
"""

io.open(PATH, "w", encoding="utf-8").write(TEXT)
print("README rewritten:", len(TEXT), "chars")
