# -*- coding: utf-8 -*-
"""FC-NTM paper experiment configuration (independent from any thesis code)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"

CLEANED_CORPUS = Path(r"D:/bg_thesis/ctm_prodlda_gsdmm_convexnmf_lda_20260903_105739/cleaned_corpus.xlsx")
DOMAIN_USER_WORDS = Path(r"D:/bg_thesis/ctm_prodlda_gsdmm_convexnmf_lda_20260903_105739/jieba_domain_user_words.txt")

EMBEDDING_MODEL_NAME = "moka-ai/m3e-base"

TOPN = 10
CV_WINDOW = 110
NPMI_UCI_WINDOW = 100

DICT_NO_BELOW = 2
DICT_NO_ABOVE = 0.9
DICT_KEEP_N = 1500

SEEDS = [11, 23, 37, 53, 71]
MAIN_K = 10
K_SWEEP = [20, 30, 40]

# Neural training defaults.
# The paper reports "at most 700 epochs per run"; that budget is now the
# single default for *every* entry point (the main FC-NTM entry point used to
# pass it on the command line while the ablation / missing-field / baseline
# scripts silently fell back to 3000, so the runs were not comparable).
NUM_EPOCHS = 700
PAPER_EPOCHS = 700
PATIENCE = 150
LEARNING_RATE = 2e-3
BATCH_SIZE = 128
HIDDEN = 200
DROPOUT = 0.2
KL_WARMUP_EPOCHS = 20
# NOTE: every entry point that produces a paper number passes ``kl_max=0.05``
# explicitly (run_fcntm_train / run_public_missing_field / run_ablation* /
# run_small / run_variant_screen_public), so 0.02 below is only the fallback
# default of ``models.FCNTM`` and is *not* the value used in the paper.
KL_MAX_WEIGHT = 0.02
CONSISTENCY_WEIGHT = 0.5
# 10-seed paired evidence on the private corpus favours 2.0: cross-field
# imputation recall rises from 0.073 (no cross-field reconstruction) / 0.109
# (weight 0.6) to 0.141 (weight 2.0), p=0.002.
CAUSE_RECON_WEIGHT = 2.0
# Fraction of records whose cause field is dropped entirely during training,
# so the item stream is trained on the same "field is missing" condition it
# meets at inference time.  0.5 lifts global C_V by ~+0.02 (9/10 seeds) at no
# measurable cost in imputation recall.
FIELD_DROPOUT = 0.5
TOPIC_ORTH_WEIGHT = 0.2
EMB_DECODER_TEMPERATURE = 0.35
