# -*- coding: utf-8 -*-
"""Fix the bibliography issues reported on 2026-10-19.

1. DOIs with underscores are set in \\url{} so that the underscore survives
   text extraction and copy/paste (the roman-font \\_ renders as a rule that
   is invisible to PDF text extraction).
2. [44] first-author initials corrected and the fourth author restored,
   following the publisher record (Peng Zhou, Yifan Qi, Qian Yang,
   Yuan Chang).
3. [21] GitHub-issues dataset pinned to the revision used for the
   experiments (Hugging Face commit f7bdbac, last updated 2026-06-06) with
   an access date.
4. [22] OpenAlex entry now links the arXiv record of the paper; the platform
   link moves into the corpus description in the body.
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

DOIS = {
    "10.1007/978-3-319-71246-8\\_15":
        "\\url{10.1007/978-3-319-71246-8_15}",
    "10.1007/978-3-319-26123-2\\_36":
        "\\url{10.1007/978-3-319-26123-2_36}",
    "10.1007/978-3-030-85896-4\\_19":
        "\\url{10.1007/978-3-030-85896-4_19}",
}

OLD_DATASET = (
    "Hugging Face, 2026(snapshot 2026-06). "
    "\\url{https://huggingface.co/datasets/open-index/open-github-issues}.")
NEW_DATASET = (
    "Hugging Face, 2026. Revision f7bdbac081d5c16af1bbd61b3d593dec5caf3b1e "
    "(last updated 2026-06-06), accessed 2026-10-19. "
    "\\url{https://huggingface.co/datasets/open-index/open-github-issues/tree/"
    "f7bdbac081d5c16af1bbd61b3d593dec5caf3b1e}.")

OLD_OPENALEX = "\\url{https://openalex.org}."
NEW_OPENALEX = "\\url{https://arxiv.org/abs/2205.01833}."

OLD_BODY = ("from OpenAlex \\cite{openalex2022} in \\emph{Civil and Structural")
NEW_BODY = ("from OpenAlex \\cite{openalex2022} (\\url{https://openalex.org}) "
            "in \\emph{Civil and Structural")

OLD_AUTHORS = "bibitem{civiltopic2024} Zhou P C, Qi Y, Yang Q, et al."
NEW_AUTHORS = "bibitem{civiltopic2024} Zhou P, Qi Y, Yang Q, Chang Y."


def main() -> None:
    path = PAPER + r"\main_en.tex"
    text = io.open(path, encoding="utf-8").read()
    replacements = []
    for old, new in DOIS.items():
        replacements.append(("DOI: " + old + ".", "DOI: " + new + "."))
    replacements += [
        (OLD_DATASET, NEW_DATASET),
        (OLD_OPENALEX, NEW_OPENALEX),
        (OLD_BODY, NEW_BODY),
        (OLD_AUTHORS, NEW_AUTHORS),
    ]
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise SystemExit("expected exactly one match, found %d: %r"
                             % (count, old[:70]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(text)
    print("main_en.tex updated:", len(replacements), "replacements")


if __name__ == "__main__":
    main()
