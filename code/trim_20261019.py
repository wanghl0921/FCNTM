# -*- coding: utf-8 -*-
"""Compress the ASCE manuscript towards the 30-page limit.

Rewrites the two densest literature paragraphs and the baseline paragraph;
every citation is preserved, only the prose is tightened. The removed
hyper-parameter detail is appended to the supplement separately.
"""

import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

RELATED = r"""\section{Related Work}

\subsection{Neural topic models}

Neural topic models map a bag of words to a latent Gaussian posterior and decode topics from it:
NVDM applied variational auto-encoders to text \cite{miao2016nvdm}, ProdLDA introduced a linear
decoder with a readable topic--word matrix \cite{srivastava2017avitm}, CTM/CombinedTM concatenate
sentence embeddings with counts \cite{bianchi2021ctm}, and ETM defines topics in an embedding space
\cite{dieng2020etm}. For short texts, BTM models word pairs \cite{yan2013btm}, BERTopic and Top2Vec
build representations from class TF--IDF statistics and joint embeddings
\cite{grootendorst2022bertopic,angelov2020top2vec}, GraphBTM and BCK-TM add word-pair structure
\cite{zhu2018graphbtm,zhang2020bcktm}, and BigARTM provides an additive regularisation framework
\cite{vorontsov2015bigartm}. Our decoder follows the ProdLDA structure with topic orthogonality,
dropout \cite{srivastava2014dropout} and validation-based early stopping \cite{prechelt1998early};
training loops and learning rates differ between models, so the objectives are not identical
across algorithms.

\subsection{Multi-view and structured-document topic models}

Multi-view topic models share information across views: bilingual, multi-source and disentangled
models \cite{kurle2019multisource,zhang2017robust,pergola2021disentangled,liu2022dualview},
structured neural topic models for reviews \cite{esmaeili2019structured}, metadata-conditioned
models \cite{card2018metadata,roberts2019stm}, and dual-view short-text work (DV-BTM, CRNTM and
DVI-DTM) \cite{liu2022dualview,crntm2020,dvidtm2026}, alongside related word-pair models
\cite{relationalbtm2018,sparsebtm2021}. Multi-source variational inference with per-source encoders
and posterior-divergence terms \cite{kurle2019multisource} and JMVAE, which infers a missing
modality from the remaining one \cite{suzuki2017jmvae}, already contain separate encoders,
posterior alignment and missing-view inference; SubNTM models sub-document topics
\cite{nguyen2026subntm}. Our setting is fixed by the record schema: two textual fields share one
vocabulary and one linear decoder, and the study asks what field-restricted read-outs and
missing-background inference measure rather than claiming these building blocks as new.

\subsection{Text mining and topic modelling of engineering documents}

Engineering informatics has an extensive text-mining literature on safety reports
\cite{civilsafety2015,civilaccident2018}, regulatory and contractual documents
\cite{compliance2013,ontology2015,contract2022,risknlp2023}, and domain surveys
\cite{civilnlp2021,civilnlp2023}. Topic models include DUET \cite{duet2019}, neural topic modelling
of building applications \cite{civiltopic2024} and passport taxonomies \cite{passport2026}. These
studies use topics as inputs to classification or mapping; we study the field structure inside one
engineering record and ask when field-restricted topics and missing-field inference are
informative.

"""

BASELINES = r"""\subsection{Baselines, training protocols and metrics}
\label{sec:protocol}

Baselines are LDA \cite{blei2003lda}, BTM \cite{yan2013btm}, NMF with a KL objective
\cite{lee2001nmf}, ProdLDA \cite{srivastava2017avitm}, ETM \cite{dieng2020etm}, CTM-lite, the
official CombinedTM \cite{bianchi2021ctm}, BERTopic \cite{grootendorst2022bertopic} and Top2Vec
\cite{angelov2020top2vec}; contextual models use Sentence-BERT embeddings
\cite{reimers2019sbert}. The implementations fall into four families with different training
procedures, which we do not describe as interchangeable loops: Gibbs sampling (BTM, LDA), matrix
factorisation (NMF), neural variational models (ProdLDA, ETM, CTM-lite and \fcntm{} share the Adam
loop and the 700-epoch budget but differ in learning rate; the official CombinedTM keeps its own
loop), and embedding clustering (BERTopic, Top2Vec), which choose their own topic numbers and are
shown for reference only. Full hyper-parameters are listed in the Supplementary Material. Quality
is measured by $C_V$ coherence \cite{roder2015coherence}, NPMI \cite{bouma2009npmi,lau2014tea}, UCI
and UMass, IRBO topic diversity \cite{webber2010rbo} and a $K$-corrected exclusivity
$EXCL_{KADJ}$ \cite{bischof2012exclusivity}; topics with non-finite scores are dropped, so the
effective topic count is recorded next to every score.

"""


def main() -> None:
    path = PAPER + r"\main_en.tex"
    text = io.open(path, encoding="utf-8").read()

    i = text.index(r"\section{Related Work}")
    j = text.index(r"\section{Method}", i)
    text = text[:i] + RELATED + text[j:]

    i = text.index(r"\subsection{Baselines, training protocols and metrics}")
    j = text.index(r"\begin{table}", i)
    text = text[:i] + BASELINES + " " + text[j:]

    io.open(path, "w", encoding="utf-8").write(text)
    print("Related Work and baselines compressed")


if __name__ == "__main__":
    main()
