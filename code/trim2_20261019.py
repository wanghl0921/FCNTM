# -*- coding: utf-8 -*-
"""Second compression pass: discussion, results prose and application notes."""

import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"


def relaxed(old: str):
    return re.compile(r"\s+".join(re.escape(w) for w in old.split()))


PAIRS = [
    # discussion, first paragraph
    ("""The value of dual-field topic modelling has to be stated in parts. What is reproducible is the
behaviour at the \\emph{representation} level: one trained model yields field-restricted topic words,
and it still infers topic proportions when the background field is missing (on GitHub about 66\\% of
the documents change their most probable topic and the per-token NLL rises by 0.24). What is
corpus-dependent is the advantage on \\emph{generic} coherence metrics: under the matched read-out,
the ten-seed comparison supports \\fcntm{}-cTFIDF being $+0.028$ $C_V$ above BTM on GitHub
(95\\% CI excludes zero, 9/10 seeds) and $+0.049$ on the civil corpus (10/10) but $-0.009$ on
AG News (3/10); on the civil corpus the native $\\beta$ read-out is also above BTM ($+0.061$,
10/10). This differs from the simple
hypothesis that complementary corpora benefit and redundant ones suffer, for two reasons: coherence
metrics are sensitive to implementation details such as the decoding path, the training budget and
Monte-Carlo noise, so measuring under one protocol is more reliable than extrapolating from
intuition; and the held-out redundancy measures show that GitHub is in fact slightly the more
predictable of the two corpora. Redundancy is therefore reported as a description of the field
relation, not as a predictor of the direction of the gain.""",
     """The reproducible part of dual-field topic modelling is the representation: one trained model
yields field-restricted topic words and still infers topic proportions when the background field is
missing (on GitHub about 66\\% of the documents change their most probable topic and the per-token
NLL rises by 0.24). The advantage on generic coherence metrics is corpus-dependent: the ten-seed
comparison supports $+0.028$ $C_V$ over BTM on GitHub (9/10 seeds) and $+0.049$ on the civil corpus
(10/10), but $-0.009$ on AG News (3/10), and on the civil corpus the native $\\beta$ read-out is also
above BTM ($+0.061$, 10/10). Coherence is sensitive to the decoding path, the training budget and
Monte-Carlo noise, and the held-out redundancy measures show that GitHub is slightly the more
predictable of the two corpora, so redundancy is reported as a description of the field relation
rather than as a predictor of the gain."""),
    # discussion, engineering-use paragraph
    ("""For engineering document management the field-restricted read-out matches the way titles and
abstracts are actually written, the occlusion diagnostic corresponds to estimating topics when
abstracts are missing, and the title-only routing experiment shows that such topics can be mapped
to civil subfields against automatically assigned OpenAlex primary-topic labels. The field-restricted read-outs,
missing-field inference and title-only routing thus provide complementary outputs for engineering
document management, and the comparison with the concatenated-field control clarifies the extent to
which these outputs depend on the complete \\fcntm{} configuration.""",
     """For engineering document management the field-restricted read-out matches how titles and
abstracts are written, the occlusion diagnostic corresponds to estimating topics when abstracts are
missing, and title-only routing maps such topics to civil subfields. These read-outs and the
concatenated-field control together show what the complete \\fcntm{} configuration contributes."""),
    # read-out-matched paragraph
    ("""The class TF--IDF read-out is strong but its effect depends on the model and the corpus: on GitHub
it lifts CTM-lite, ProdLDA and LDA by $0.091$, $0.087$ and $0.068$ but leaves ETM ($-0.006$) and
NMF ($-0.011$) unchanged or worse, while on AG News it helps every model except BTM by
$+0.005$--$+0.045$. Comparing \\fcntm{}-cTFIDF against baselines read out with their own $\\beta$
matrix therefore mixes the model and the read-out; every model-quality statement below therefore uses the matched comparison. Under the matched read-out the strongest baselines on GitHub are BTM ($0.700$),
CTM-lite ($0.675$), ETM ($0.669$) and ProdLDA ($0.666$), and \\fcntm{}-cTFIDF leads them by
$0.027$--$0.061$ $C_V$ and $0.039$--$0.059$ NPMI, larger than its seed-to-seed standard deviation
($0.022$). On AG News the matched advantage is $+0.019$ $C_V$ over BTM but only two of three seeds
are positive and the NPMI advantage rests on one seed, while under the native $\\beta$ read-out
\\fcntm{} is below BTM ($0.586$ versus $0.600$). On the civil corpus the strongest
matched-read-out baselines are NMF ($0.607$) and ETM ($0.603$), ahead of BTM ($0.590$);
\\fcntm{}-cTFIDF leads them by $0.029$ and $0.033$ $C_V$, but the ten-seed paired comparison is
against BTM, so the multi-seed civil evidence is specific to that comparison. The stable gains are therefore concentrated on the matched read-out, with a corpus-specific pattern. Two caveats apply: the class TF--IDF transform does not
automatically favour BTM (its $C_V$ drops from $0.712$ to $0.700$ on GitHub and from $0.600$ to
$0.593$ on AG News), so its direction must be measured rather than assumed; and the architecture
hyper-parameters of \\fcntm{} were selected on development data while the baselines use their
defaults, a tuning-budget difference that the read-out comparison cannot remove.""",
     """The class TF--IDF read-out depends on the model and the corpus: on GitHub it lifts CTM-lite,
ProdLDA and LDA by $0.091$, $0.087$ and $0.068$ but leaves ETM and NMF unchanged or worse, while on
AG News it helps every model except BTM. Comparisons therefore use the matched read-out. Under it,
\\fcntm{}-cTFIDF leads the strongest GitHub baselines (BTM $0.700$, CTM-lite $0.675$, ETM $0.669$,
ProdLDA $0.666$) by $0.027$--$0.061$ $C_V$ and $0.039$--$0.059$ NPMI, larger than its seed-to-seed
standard deviation ($0.022$); on AG News the advantage over BTM is $+0.019$ $C_V$ but rests on two
of three seeds, and under the native $\\beta$ read-out \\fcntm{} is below BTM ($0.586$ versus
$0.600$); on the civil corpus NMF ($0.607$) and ETM ($0.603$) are ahead of BTM ($0.590$), and
\\fcntm{}-cTFIDF leads them by $0.029$ and $0.033$ $C_V$, although the ten-seed comparison is
against BTM. Two caveats remain: the TF--IDF transform does not automatically favour BTM
($0.712\\to0.700$ on GitHub, $0.600\\to0.593$ on AG News), and the architecture hyper-parameters
of \\fcntm{} were selected on development data while the baselines use defaults."""),
    # concatenated control, closing paragraph
    ("""The control is competitive everywhere, so the two-stream structure is not a necessary condition
for topic quality. After the validation-mode correction the mean difference favours \\fcntm{} in
every corpus--read-out cell ($+0.010$ to $+0.029$ $C_V$), but the paired majorities are mixed
(1/3 to 3/3) and the gaps remain at the scale of the seed-to-seed variation with three seeds per
cell, so the two-stream architecture is reported as a matched design choice rather than a source of unconditional coherence gains.
The control also supports
field-restricted read-out and title-only inference, so those read-outs are not exclusive to the
two-stream design either.""",
     """The control is competitive everywhere, so the two-stream structure is not necessary for topic
quality. The mean difference favours \\fcntm{} in every corpus--read-out cell ($+0.010$ to
$+0.029$ $C_V$), but the paired majorities are mixed (1/3 to 3/3) and the gaps are at the scale of
three-seed variation, so the two-stream architecture is a matched design choice rather than a
source of unconditional gains; the control also supports field-restricted read-out and title-only
inference."""),
]


def main() -> None:
    path = PAPER + r"\main_en.tex"
    text = io.open(path, encoding="utf-8").read()
    for old, new in PAIRS:
        text, n = relaxed(old).subn(new, text, count=1)
        if n != 1:
            raise SystemExit("no match for: %r" % old[:70])
    io.open(path, "w", encoding="utf-8").write(text)
    print("compressed", len(PAIRS), "paragraphs")


if __name__ == "__main__":
    main()
