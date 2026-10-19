# -*- coding: utf-8 -*-
"""Second negative-phrasing pass (2026-10-19, latest review).

* discussion: replace the repeated "gain is limited" sentence with the
  engineering-use framing;
* limitations: four separate paragraphs, each stating the limitation, its
  effect on the conclusions and the next validation step;
* conclusion: the routing result is quoted once (it was duplicated);
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

OLD_USE = (
    "For engineering document management the protocol has direct uses: the field-restricted "
    "read-out\nmatches the way titles and abstracts are actually written, the occlusion "
    "diagnostic corresponds\nto estimating topics when abstracts are missing, and the "
    "title-only routing experiment shows that\nsuch topics can be mapped to civil subfields "
    "with moderate accuracy against independent OpenAlex\nlabels. Even though the generic "
    "coherence gain over the concatenated control is limited, these\nread-outs are usable "
    "engineering outputs.")

NEW_USE = (
    "For engineering document management the field-restricted read-out matches the way titles "
    "and\nabstracts are actually written, the occlusion diagnostic corresponds to estimating "
    "topics when\nabstracts are missing, and the title-only routing experiment shows that such "
    "topics can be mapped\nto civil subfields against independent OpenAlex labels. The "
    "field-restricted read-outs, missing-field\ninference and title-only routing thus provide "
    "complementary outputs for engineering document\nmanagement, and the comparison with the "
    "concatenated-field control clarifies the extent to which\nthese outputs depend on the "
    "complete \\fcntm{} configuration.")

OLD_LIM = (
    "Four limitations delimit the present evidence. \\emph{Data extrapolation}: all three "
    "corpora are\nEnglish and the field split follows existing title/body structure; automatic "
    "field-boundary detection\nis untested, and the GitHub background field has a long tail "
    "(up to 20532 tokens) that the\nsentence-embedding baselines truncate at 128 tokens, for "
    "which the Supplementary Material reports a\nwindow-pooling control. \\emph{Experimental "
    "design}: the redundancy measures are computed on two\ncorpora with almost the same "
    "held-out predictability ($\\rho^2_{\\mathrm{cca}}$ 0.306 versus 0.245)\nwhile the "
    "ten-seed gain appears on one of them, so the two corpora define the current calibration\n"
    "range and a redundancy-based pre-training criterion will need calibration on additional "
    "corpora and an\nindependent test set; no independent confirmation corpus is used; and all "
    "numbers come from one\nprotocol ($K=30$, 700-epoch budget, unified read-out and evaluation "
    "vocabulary), with \\fcntm{}\nand BTM extended to ten seeds on all three corpora (and "
    "LDA's occlusion curves for the new seeds)\nwhile the remaining baselines and the main "
    "tables stay at three seeds. \\emph{Method applicability}:\nfield-restricted read-out is "
    "available to any model with a topic--word matrix and is not unique to a\ntwo-stream "
    "encoder; the non-parametric baselines choose their own topic numbers (Top2Vec degrades\nto "
    "2--4 topics on AG News) and are shown for reference only; the masking rate, the "
    "whole-field\ndropout probability and the fusion weight are empirical values whose "
    "interaction with corpus size\nand field ratio is not characterised; and the in-house "
    "variational baselines and \\fcntm{} use 20\\% of the\ntraining documents for early "
    "stopping (the official CombinedTM keeps its own held-out split), while\nthe class TF--IDF "
    "read-out of \\fcntm{} aggregates the full training set, so the number of documents\nused "
    "for optimisation and for read-out differs. \\emph{Mechanism interpretation}: the "
    "concatenated-field\ncontrol and the title-only routing task are run for three seeds per "
    "cell and against one label\nsource (OpenAlex primary topics), so their differences should "
    "be read at the scale of the\nseed-to-seed variation, and the occlusion ratio and the "
    "redundancy measures describe, rather than\npredict, missing-field behaviour and the field "
    "relation.")

NEW_LIM = (
    "\\emph{Data extrapolation.} All three corpora are English and the field split follows "
    "existing\ntitle/body structure; automatic field-boundary detection is untested, and the "
    "GitHub background field\nhas a long tail that the sentence-embedding baselines truncate at "
    "128 tokens (the Supplementary Material\nreports a window-pooling control). The conclusions "
    "therefore describe title/body-style records in\nEnglish. Extending them to other "
    "languages, other field conventions or automatically detected\nboundaries requires "
    "re-running the same protocol on those corpora.\n\n"
    "\\emph{Experimental design.} The redundancy measures are computed on two corpora with "
    "almost the\nsame held-out predictability ($\\rho^2_{\\mathrm{cca}}$ 0.306 versus 0.245) "
    "while the ten-seed gain\nappears on one of them, so the two corpora define the current "
    "calibration range; all numbers come\nfrom one protocol ($K=30$, 700-epoch budget, unified "
    "read-out and evaluation vocabulary), with\n\\fcntm{} and BTM extended to ten seeds on all "
    "three corpora while the remaining baselines and the\nmain tables stay at three seeds. The "
    "precision of the reported comparisons is therefore bounded by\nthese seed counts, and a "
    "redundancy-based pre-training criterion or any claim outside these settings\nrequires "
    "calibration on additional corpora, more seeds for the remaining controls and an "
    "independent\nconfirmation corpus.\n\n"
    "\\emph{Method applicability.} Field-restricted read-out is available to any model with a\n"
    "topic--word matrix and is not unique to a two-stream encoder; the non-parametric baselines "
    "choose\ntheir own topic numbers and are shown for reference only; and the masking rate, the "
    "whole-field dropout\nprobability and the fusion weight are empirical values whose "
    "interaction with corpus size and field\nratio is not characterised. In addition, the "
    "in-house variational baselines and \\fcntm{} use 20\\% of the\ntraining documents for "
    "early stopping (the official CombinedTM keeps its own held-out split), while\nthe class "
    "TF--IDF read-out of \\fcntm{} aggregates the full training set, so the number of "
    "documents\nused for optimisation and for read-out differs. The configuration reported here "
    "is therefore a\ncalibrated setting rather than a set of universal defaults, and transferring "
    "it should re-tune these\nvalues on the target corpus.\n\n"
    "\\emph{Mechanism interpretation.} The concatenated-field control and the title-only "
    "routing task\nare run for three seeds per cell and against one label source (OpenAlex "
    "primary topics), so their\ndifferences should be read at the scale of the seed-to-seed "
    "variation; the occlusion ratio and the\nredundancy measures describe, rather than predict, "
    "missing-field behaviour and the field relation.\nSeparating ``recovering content from the "
    "summary'' from ``insensitivity to the input'' therefore\nrequires a protocol with an "
    "independent source of the missing content.")

OLD_DUP = (
    "the civil corpus and on GitHub; title-only routing reaches 0.66 accuracy against "
    "independent OpenAlex\nlabels (42.9\\% coverage). The boundaries")

NEW_DUP = "the civil corpus and on GitHub. The boundaries"

REPLACEMENTS = [(OLD_USE, NEW_USE), (OLD_LIM, NEW_LIM), (OLD_DUP, NEW_DUP)]


def main() -> None:
    path = PAPER + r"\main_en.tex"
    text = io.open(path, encoding="utf-8").read()
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit("expected one match, found %d: %r"
                             % (count, old[:90]))
        text = text.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(text)
    print("main_en.tex updated:", len(REPLACEMENTS), "passages")


if __name__ == "__main__":
    main()
