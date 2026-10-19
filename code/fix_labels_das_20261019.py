# -*- coding: utf-8 -*-
"""Latest review (2026-10-19): label wording and the data-availability text.

* "external automatic labels" / "independent OpenAlex labels" are replaced by
  explicit wording: automatically assigned OpenAlex primary-topic labels on
  the covered subset, an external reference rather than expert ground truth;
* the data-availability statement lists the released materials and the
  third-party data condition.
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
SCRIPT = r"C:\Users\Administrator\Downloads\FCNTM完整代码" \
         r"\make_asce_version_20261015.py"

EN = [
    ("Routing\ntitles to OpenAlex primary topics reaches 0.66 accuracy on 15 external "
     "automatic labels (42.9\\%\nof the test split) with a supervised TF--IDF classifier "
     "as the upper reference; the task is designed as a routing aid complementing, "
     "rather than replacing, supervised classification.",
     "Title-only routing reaches 0.66 accuracy against OpenAlex primary-topic labels on "
     "the subset covered by the 15 selected topics (42.9\\% of the test split); these "
     "labels are assigned automatically and serve as an external reference rather than "
     "expert-annotated ground truth, and the task is a routing aid complementing, rather "
     "than replacing, supervised classification."),
    ("to civil subfields against independent OpenAlex labels.",
     "to civil subfields against automatically assigned OpenAlex primary-topic labels."),
    ("and title-only routing reaches\n0.66 accuracy against independent OpenAlex labels "
     "(42.9\\% coverage).",
     "and title-only routing reaches\n0.66 accuracy against automatically assigned "
     "OpenAlex primary-topic labels on the 42.9\\% of the test\nsplit covered by the 15 "
     "selected topics."),
]

ASCE = [
    ("Routing titles to OpenAlex primary topics reaches 0.66 accuracy on\n"
     "fifteen external automatic labels covering 42.9\\% of the test split, with a "
     "supervised TF-IDF\nclassifier as the upper reference; the task is designed as a "
     "routing aid complementing, rather\nthan replacing, supervised classification.",
     "Title-only routing reaches 0.66 accuracy against OpenAlex primary-topic labels on "
     "the subset\ncovered by the fifteen selected topics (42.9\\% of the test split); "
     "these labels are assigned\nautomatically and serve as an external reference rather "
     "than expert-annotated ground truth, and\nthe task is a routing aid complementing, "
     "rather than replacing, supervised classification."),
    ("Some or all data, models, or code that support the findings of this study are "
     "available from the\ncorresponding author upon reasonable request. The archived "
     "corpus splits, model checkpoints and\nper-seed results accompany this submission, "
     "and the reproduction code is released in a public\nrepository.",
     "The code, the fixed corpus splits, the trained model checkpoints and the per-seed "
     "experimental\nresults that support this study accompany the submission and are "
     "released in a public repository\n(the permanent repository URL is inserted at "
     "submission). The experiments can be reproduced with\nthe released configurations "
     "and evaluation scripts. Access to the third-party source corpora\nremains subject "
     "to the terms and licences of the original providers."),
]


def apply(path: str, pairs) -> None:
    text = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        count = text.count(old)
        if count != 1:
            raise SystemExit("%s: expected one match, found %d: %r"
                             % (path, count, old[:80]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(text)
    print("updated", path.split("\\")[-1], "with", len(pairs), "edits")


def main() -> None:
    apply(PAPER + r"\main_en.tex", EN)
    apply(SCRIPT, ASCE)


if __name__ == "__main__":
    main()
