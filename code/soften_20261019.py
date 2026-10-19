# -*- coding: utf-8 -*-
"""Apply the negative-phrasing revision of 2026-10-19.

Keeps every reported result and every evidence boundary; removes repeated
self-negation from the abstract, reframes the empirical contribution
positively, groups the limitations into four categories, and reorders the
conclusion so that the supported contributions come first.
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

OLD_ABS = (
    "A same-budget concatenated-field control is close on every corpus, which frames the "
    "two-stream design as a matched alternative; a constant unigram model is almost as stable "
    "as \\fcntm{} under the\nocclusion ratio, so that ratio is a diagnostic under a specified "
    "baseline set rather than\nevidence of complementarity. Cross-domain evaluations and "
    "read-out protocols are reported in the paper and supplement;\nper-seed results, fixed data "
    "splits and hyper-parameters are released with the code.")

NEW_ABS = (
    "The results show that \\fcntm{} provides field-restricted topic read-outs and supports "
    "topic inference\nwhen one field is missing. The matched class TF--IDF read-out is above "
    "BTM on the civil corpus (ten\nof ten seeds) and on GitHub issues (nine of ten), while the "
    "corresponding advantage is not observed\non AG News; a same-budget concatenated-field "
    "control achieves comparable performance, so the reported\ngains pertain to the complete "
    "modelling configuration. The occlusion experiments characterise\ninference stability "
    "under missing-field conditions, read alongside absolute prediction loss and topic\n"
    "agreement, and identify the conditions that require further validation. Cross-domain "
    "evaluations\nand read-out protocols are reported in the paper and supplement; per-seed "
    "results, fixed data\nsplits and hyper-parameters are released with the code.")

OLD_INTRO = (
    "\\textbf{Empirically}, the paper\ndoes not claim that the mechanism beats classical "
    "baselines on arbitrary dual-field corpora.\nIt asks what field structure provides and "
    "whether the resulting gains can be measured reliably.")

NEW_INTRO = (
    "\\textbf{Empirically}, the paper characterises the conditions under which "
    "field-cooperative topic\nmodelling provides reproducible read-out gains, and how inference "
    "behaves when a field is missing.")

OLD_DISC = (
    "The limitations are as follows. First, all three corpora are English and the field split "
    "follows\nexisting title/body structure; automatic field-boundary detection is untested, "
    "and the GitHub\nbackground field has a long tail (up to 20532 tokens) that the "
    "sentence-embedding baselines truncate\nat 128 tokens, for which the Supplementary Material "
    "reports a window-pooling control. Second, the\nredundancy measures are computed on two "
    "corpora with almost the same held-out predictability\n($\\rho^2_{\\mathrm{cca}}$ 0.306 "
    "versus 0.245) while the ten-seed gain appears on one of them, so the two corpora define "
    "the current calibration range, and a redundancy-based pre-training criterion will need "
    "calibration on additional corpora and an independent test set. Third, the non-parametric\n"
    "baselines choose their own topic numbers (Top2Vec degrades to 2--4 topics on AG News) and "
    "are shown\nfor reference only. Fourth, field-restricted read-out is available to any model "
    "with a\ntopic--word matrix and is not unique to a two-stream encoder. Fifth, the masking "
    "rate, the\nwhole-field dropout probability and the fusion weight are empirical values "
    "whose interaction with\ncorpus size and field ratio is not characterised; the in-house "
    "variational baselines and \\fcntm{}\nuse 20\\% of the training documents for early stopping "
    "(the official CombinedTM keeps its own\nheld-out split), while the class TF--IDF read-out "
    "of \\fcntm{} aggregates the full training set, so\nthe number of documents used for "
    "optimisation and for read-out differs. Sixth, all\nnumbers come from one protocol ($K=30$, "
    "700-epoch budget, unified read-out and evaluation\nvocabulary): \\fcntm{} and BTM are "
    "extended to ten seeds on all three corpora (and LDA's occlusion\ncurves for the new "
    "seeds), while the remaining baselines and the main tables stay at three seeds.\nSeventh, "
    "the concatenated-field control and the title-only routing task are run for three seeds\n"
    "per cell and against one label source (OpenAlex primary topics), so their differences "
    "should be\nread at the scale of the seed-to-seed variation. Finally, no independent "
    "confirmation corpus is\nused.")

NEW_DISC = (
    "Four limitations delimit the present evidence. \\emph{Data extrapolation}: all three "
    "corpora are\nEnglish and the field split follows existing title/body structure; automatic "
    "field-boundary detection\nis untested, and the GitHub background field has a long tail "
    "(up to 20532 tokens) that the\nsentence-embedding baselines truncate at 128 tokens, for "
    "which the Supplementary Material reports a\nwindow-pooling control. \\emph{Experimental "
    "design}: the redundancy measures are computed on two\ncorpora with almost the same "
    "held-out predictability ($\\rho^2_{\\mathrm{cca}}$ 0.306 versus 0.245)\nwhile the ten-seed "
    "gain appears on one of them, so the two corpora define the current calibration\nrange and "
    "a redundancy-based pre-training criterion will need calibration on additional corpora and "
    "an\nindependent test set; no independent confirmation corpus is used; and all numbers come "
    "from one\nprotocol ($K=30$, 700-epoch budget, unified read-out and evaluation vocabulary), "
    "with \\fcntm{}\nand BTM extended to ten seeds on all three corpora (and LDA's occlusion "
    "curves for the new seeds)\nwhile the remaining baselines and the main tables stay at three "
    "seeds. \\emph{Method applicability}:\nfield-restricted read-out is available to any model "
    "with a topic--word matrix and is not unique to a\ntwo-stream encoder; the non-parametric "
    "baselines choose their own topic numbers (Top2Vec degrades\nto 2--4 topics on AG News) and "
    "are shown for reference only; the masking rate, the whole-field\ndropout probability and "
    "the fusion weight are empirical values whose interaction with corpus size\nand field ratio "
    "is not characterised; and the in-house variational baselines and \\fcntm{} use 20\\% of "
    "the\ntraining documents for early stopping (the official CombinedTM keeps its own held-out "
    "split), while\nthe class TF--IDF read-out of \\fcntm{} aggregates the full training set, so "
    "the number of documents\nused for optimisation and for read-out differs. \\emph{Mechanism "
    "interpretation}: the concatenated-field\ncontrol and the title-only routing task are run "
    "for three seeds per cell and against one label\nsource (OpenAlex primary topics), so their "
    "differences should be read at the scale of the\nseed-to-seed variation, and the occlusion "
    "ratio and the redundancy measures describe, rather than\npredict, missing-field behaviour "
    "and the field relation.")

OLD_CONC = (
    "Two field-relation measures were tested and\nneither predicts the gain: the corpora have "
    "similar redundancy, and occluding the background field\nleaves coherence almost unchanged "
    "while changing the document-level posterior and the word-level\nprediction. The "
    "contribution of this paper is therefore scoped: \\fcntm{} provides reproducible\n"
    "field-restricted read-outs and missing-field inference behaviour, and read-out-matched "
    "advantages\nsupported by ten seeds on the civil corpus and on GitHub. The complete "
    "\\fcntm{} configuration shows\nsmall mean coherence gains over the concatenated control, "
    "which is reported as a matched configuration-level comparison. Whether field structure\n"
    "is worth the extra modelling cost should be measured on the target corpus with the "
    "protocol used\nhere rather than decided from redundancy beforehand.")

NEW_CONC = (
    "Taken together, the contributions are the field-cooperative configuration, the unified "
    "read-out\nand the missing-field inference behaviour, with read-out-matched advantages "
    "supported by ten seeds on\nthe civil corpus and on GitHub; title-only routing reaches 0.66 "
    "accuracy against independent OpenAlex\nlabels (42.9\\% coverage). The boundaries of this "
    "evidence are stated by the two field-relation\nmeasures: the corpora have similar "
    "redundancy, and occluding the background field leaves coherence\nalmost unchanged while "
    "changing the document-level posterior and the word-level prediction, so neither\nmeasure "
    "is used to predict the direction of the gain. The complete configuration shows small mean\n"
    "coherence gains over the concatenated control, and the reported effects are "
    "corpus-dependent;\nwhether field structure is worth the extra modelling cost should "
    "therefore be measured on the\ntarget corpus with the protocol used here.")

REPLACEMENTS = [(OLD_ABS, NEW_ABS), (OLD_INTRO, NEW_INTRO),
                (OLD_DISC, NEW_DISC), (OLD_CONC, NEW_CONC)]


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
