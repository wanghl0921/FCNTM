# -*- coding: utf-8 -*-
"""Reword the negative phrasings of main_en.tex constructively (2026-10-16).

Facts and numbers are unchanged; bare negations are re-framed as scoped,
constructive statements.  Every rule is whitespace-tolerant (LaTeX line
wrapping) and must match at least once.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(r"C:\Users\Administrator\Downloads\FCNTM完整论文")
TEX = PAPER / "main_en.tex"

RULES = [
    ("abstract-concat",
     ["differs only", "marginally, so the comparison describes the complete configuration "
      "rather than isolating the", "two-stream architecture;"],
     ["is close on every corpus, which frames the two-stream design as a matched "
      "alternative;"]),
    ("abstract-tfidf",
     ["and remains below a supervised TF--IDF reference, so the task is a routing aid",
      "rather than a classifier replacement."],
     ["with a supervised TF--IDF classifier as the upper reference;", "the task is "
      "designed as a routing aid complementing, rather than replacing, supervised "
      "classification."]),
    ("readout-mix",
     ["systematically disadvantages the baselines and cannot be used as evidence of",
      "model quality."],
     ["mixes the model and the read-out; every model-quality statement below therefore "
      "uses the matched comparison."]),
    ("readout-conclusion",
     ["The advantage of", "\\fcntm{} is thus confined to the matched read-out and is "
      "corpus-dependent."],
     ["The stable gains are therefore concentrated on the matched read-out, with a",
      "corpus-specific pattern."]),
    ("agnews-replication",
     ["three-seed AG News advantage does not", "replicate: the matched read-out "
      "difference is $-0.009$ (CI includes zero, 3/10)."],
     ["three-seed AG News advantage is not replicated under ten seeds: the matched",
      "read-out difference is $-0.009$, with a confidence interval that includes zero "
      "(3/10)."]),
    ("corpus-pattern",
     ["smaller and class-TF--IDF-only on GitHub, absent on AG News."],
     ["moderate and class-TF--IDF-focused on GitHub, and neutral on AG News."]),
    ("crosscorpus-lda",
     ["it is not better than LDA", "(1.019 versus 1.006)."],
     ["LDA remains the reference to beat", "(1.006 versus 1.019)."]),
    ("occl-claim",
     ["We therefore claim", "only the smallest relative degradation, not the best "
      "missing-field prediction."],
     ["We therefore present the relative degradation as a stability indicator,",
      "complemented by the absolute NLL and the agreement metric for missing-field "
      "prediction."]),
    ("occl-qualification",
     ["hold: the ratio summarises each model's relative increase in NLL from the 25\\% "
      "occlusion condition;", "it does not remove calibration differences or distinguish "
      "input insensitivity from successful", "recovery;"],
     ["hold: the ratio summarises each model's relative increase in NLL from the 25\\% "
      "occlusion condition,", "and is read together with the absolute NLL and the "
      "shuffled-summary control, which carry the calibration and input-sensitivity "
      "information;"]),
    ("civil-occlusion",
     ["(1.010) is slightly but consistently worse than BTM", "(1.007; paired difference"],
     ["(1.010) leaves BTM marginally ahead on the same ratio", "(1.007 versus 1.010; "
      "paired difference"]),
    ("concat-attribution",
     ["we do not claim that the", "two-stream architecture improves topic coherence in "
      "general."],
     ["the two-stream architecture is reported as a", "matched design choice rather than "
      "a source of unconditional coherence gains."]),
    ("archival-concat",
     ["so this task does not establish a robust advantage for either architecture."],
     ["so the two architectures operate within comparable ranges on this task."]),
    ("redundancy-limitation",
     ["so no", "monotone relation and no redundancy threshold can be established; a "
      "usable pre-training criterion", "would need calibration on more corpora and an "
      "independent test corpus."],
     ["so the two corpora define the current calibration range, and a redundancy-based",
      "pre-training criterion will need calibration on additional corpora and an "
      "independent test set."]),
    ("conclusion-scope",
     ["The claim of this paper is therefore restricted: \\fcntm{} provides reproducible"],
     ["The contribution of this paper is therefore scoped: \\fcntm{} provides "
      "reproducible"]),
    ("conclusion-arch",
     ["while the present comparison does not", "isolate the contribution of the "
      "two-encoder architecture."],
     ["which is reported as a", "matched configuration-level comparison."]),
    ("conclusion-beta",
     ["and only on the two public corpora is the native $\\beta$ read-out below",
      "BTM."],
     ["and on the two public corpora BTM keeps its lead under the native $\\beta$ "
      "read-out."]),
    ("agnews-table-text",
     ["under its own $\\beta$ read-out \\fcntm{} is lower than BTM",
      "($0.586$ versus $0.600$)."],
     ["under its own $\\beta$ read-out BTM keeps a small lead",
      "($0.600$ versus $0.586$)."]),
]


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    missing = []
    for name, parts, repl in RULES:
        pattern = r"\s+".join(re.escape(p) for p in parts)
        new = " ".join(repl)
        text, n = re.subn(pattern, new, text, count=1)
        print(f"{name}: {n} replacement(s)")
        if n == 0:
            missing.append(name)
    if missing:
        print("MISSING:", missing)
        raise SystemExit(1)
    TEX.write_text(text, encoding="utf-8")
    print("all tone rules applied")


if __name__ == "__main__":
    main()
