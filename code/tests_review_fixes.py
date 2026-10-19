# -*- coding: utf-8 -*-
"""Fast numeric checks for the fixes made in response to the review.

    python tests_review_fixes.py

Every test either passes or raises.  Only the resume-identity test trains
a tiny synthetic model; nothing here touches the paper's corpora.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import baselines as BL
import metrics as M
import models as NN


def test_btm_conditional_matches_dirichlet_multinomial() -> None:
    """The Gibbs conditional must equal the closed-form Dirichlet expectation.

    For phi ~ Dir(alpha), E[prod_i phi_i^(a_i)] = prod_i Gamma(alpha_i + a_i)
    / Gamma(alpha_i) * Gamma(S) / Gamma(S + sum a).  With alpha = n_kw + eta
    this gives exactly the numerator/denominator pair used by the sampler.
    """
    V, K, eta, alpha_prior = 4, 3, 0.01, 50.0 / 3
    rng = np.random.RandomState(0)
    # a state that a real biterm sampler can produce: every biterm adds two
    # word tokens, so sum_v n_kw == 2 n_k
    n_wz = (2 * rng.randint(0, 3, size=(K, V))).astype(float)
    n_bz = n_wz.sum(axis=1) / 2.0

    def closed_form(n, w1, w2):
        a = n + eta
        S = a.sum()
        if w1 == w2:
            return a[w1] * (a[w1] + 1.0) / (S * (S + 1.0))
        return a[w1] * a[w2] / (S * (S + 1.0))

    for w1 in range(V):
        for w2 in range(V):
            logp = BL._btm_conditional(n_bz, n_wz, w1, w2, alpha_prior, eta, V)
            for k in range(K):
                want = (n_bz[k] + alpha_prior) * closed_form(n_wz[k], w1, w2)
                got = math.exp(logp[k])
                assert math.isclose(got, want, rel_tol=1e-9), (w1, w2, k, got, want)
    print("  ok  BTM conditional == Dirichlet-multinomial closed form")


def test_btm_same_word_terms_differ() -> None:
    """w1 == w2 must get the +1 term; omitting it was the reported bug."""
    V, K = 3, 2
    n_bz = np.array([4.0, 1.0])
    n_wz = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    same = BL._btm_conditional(n_bz, n_wz, 1, 1, 1.0, 0.01, V)
    diff = BL._btm_conditional(n_bz, n_wz, 1, 0, 1.0, 0.01, V)
    ratio = np.exp(same - diff)
    assert np.all(ratio > 0) and not np.allclose(ratio, 1.0)
    print("  ok  same-word indicator changes the conditional")


def test_irbo_range_and_bounds() -> None:
    a = [f"w{i}" for i in range(10)]
    disjoint = [f"x{i}" for i in range(10)]
    half = a[5:] + [f"y{i}" for i in range(5)]
    assert abs(M.rbo(a, a) - 1.0) < 1e-9, M.rbo(a, a)
    assert abs(M.rbo_ext(a, a) - 1.0) < 1e-9, M.rbo_ext(a, a)
    assert M.irbo_diversity([a, a]) < 1e-9
    assert abs(M.irbo_diversity([a, disjoint]) - 1.0) < 1e-9
    assert 0.0 < M.irbo_diversity([a, half]) < 1.0
    assert M.irbo_diversity([a, a], tail="trunc") > 0.3
    print("  ok  IRBO is 0 for identical topic lists, "
          f"extrapolated RBO(identical)={M.rbo_ext(a, a):.6f}")


def test_etm_decoder_is_a_word_probability_mixture() -> None:
    torch.manual_seed(0)
    V, K, D, B = 6, 3, 5, 4
    emb = torch.randn(V, D)
    dec = NN.EmbeddingDecoder(K, emb.numpy(), temperature=1.0)
    dec.eval()
    theta = torch.softmax(torch.randn(B, K), dim=-1)
    log_p = dec.log_prob(theta)
    beta = dec.topic_word_prob()
    manual = torch.log((theta @ beta).clamp_min(1e-12))
    assert torch.allclose(log_p, manual, atol=1e-6)
    assert torch.allclose(torch.exp(log_p).sum(-1), torch.ones(B), atol=1e-5)
    old = torch.log_softmax(dec(theta), dim=-1)
    assert not torch.allclose(log_p, old, atol=1e-3)
    print("  ok  ETM decoder = log sum_k theta_k softmax_v(alpha_k E^T)")


def test_fcntm_decodes_each_stream_once() -> None:
    """Two target fields of one stream must share one decoder call."""
    torch.manual_seed(0)
    model = NN.FCNTM(20, 3, consistency_weight=0.5, cross_reconstruction=True,
                     cause_dropout=0.0, item_dropout=0.0, field_dropout=0.0)
    model.train()
    calls = {"n": 0}
    original = model.decode

    def counted(theta):
        calls["n"] += 1
        return original(theta)

    model.decode = counted
    x = torch.randint(0, 3, (8, 20)).float()
    model.train_step(x, x, x, x, torch.ones(8, dtype=torch.bool), 0.0)
    assert calls["n"] == 2, calls["n"]
    print("  ok  one decoder call per active stream (2 streams -> 2 calls)")


def test_rollback_clears_optimizer_state() -> None:
    torch.manual_seed(0)
    model = NN.ProdLDAModel(20, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(8, 20)
    loss, _ = model.train_step(x, x, 0.05)
    loss.backward()
    opt.step()
    assert opt.state, "optimizer state should be populated"
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    NN._rollback(model, opt, state)
    assert not opt.state, "rollback must clear the Adam moments"
    assert opt.param_groups[0]["lr"] < 1e-3
    print("  ok  NaN rollback clears weights + Adam moments and halves the LR")


def test_benchmark_resume_logic() -> None:
    import benchmark_public as BP

    ok = BP._row_is_complete({"error": np.nan, "CV": 0.4, "status": "ok"})
    ok_empty = BP._row_is_complete({"error": "", "CV": 0.4})
    bad = BP._row_is_complete({"error": "boom", "CV": np.nan})
    no_metric = BP._row_is_complete({"error": np.nan, "CV": np.nan})
    assert ok and ok_empty and not bad and not no_metric
    print("  ok  resume skips only rows with finite metrics and no error")


def test_single_active_field_does_not_break_batchnorm() -> None:
    """A field mask may leave exactly one active record (round-2 review 6.6).

    BatchNorm1d in training mode raises "Expected more than 1 value per
    channel" for a single sample; the decoder must fall back to the running
    statistics instead of failing.
    """
    torch.manual_seed(0)
    model = NN.FCNTM(20, 3, field_dropout=0.0)
    model.train()
    x = torch.ones(8, 20)
    has_cause = torch.tensor([True] + [False] * 7)
    loss, _ = model.train_step(x, x, x, x, has_cause, 0.0)
    assert torch.isfinite(loss)
    print("  ok  one active field record does not break BatchNorm1d")


def test_summary_groups_by_full_run_key() -> None:
    """Different K values must not be averaged into one summary row."""
    import benchmark_public as BP

    base = {"model": "X", "seed": 11, "topn": 10, "readout": "ctfidf",
            "UCI": 0.0, "UMass": 0.0, "EXCL": 0.0, "EXCL_KADJ": 0.0,
            "IRBO": 0.0, "status": "ok"}
    rows = [dict(base, K_target=20, topic_count=20, CV=0.5, NPMI=0.1),
            dict(base, K_target=30, topic_count=30, CV=0.6, NPMI=0.2)]
    out = BP.summarize_rows(rows)
    assert len(out) == 2, out
    assert sorted(out["K_target"].tolist()) == [20, 30]
    print("  ok  summary keeps K=20 and K=30 apart")


def test_btm_doc_topic_is_the_normalised_biterm_posterior() -> None:
    """The *predictive* variant of p(z|d) is a normalised biterm posterior.

    The paper's §3.3 equation is the point-estimate form, which is the default
    and is checked in ``test_btm_and_nmf_never_score_a_non_distribution``; this
    test covers the Dirichlet-integrated variant (the sensitivity check) that
    the sampler's conditional corresponds to.  The reference below is derived
    independently of the implementation, so it checks both the aggregation
    (each biterm weighted by p(b|d)) and the per-biterm posterior.
    """
    K, V = 3, 4
    n_wz = np.array([[3.0, 1.0, 0.0, 1.0],
                     [0.0, 2.0, 2.0, 1.0],
                     [1.0, 0.0, 1.0, 3.0]])
    n_bz = n_wz.sum(axis=1) / 2.0
    biterms = [(0, 1), (1, 2), (3, 3), (0, 0), (2, 3)]
    owners = [0, 0, 1, 1, 2]
    alpha, eta = 50.0 / K, 0.01

    theta = BL.btm_doc_topic(biterms, owners, 3, n_bz, n_wz, alpha, eta, V,
                             form="predictive")
    assert theta.shape == (3, K)
    assert np.allclose(theta.sum(axis=1), 1.0)

    def posterior(w1, w2):
        a = n_wz + eta                  # Dirichlet posterior of phi
        S = a.sum(axis=1)
        if w1 == w2:
            like = a[:, w1] * (a[:, w1] + 1.0)
        else:
            like = a[:, w1] * a[:, w2]
        want = (n_bz + alpha) * like / (S * (S + 1.0))
        return want / want.sum()

    for doc, own in enumerate([(0, 1), (2, 3), (4,)]):
        want = np.zeros(K)
        for b in own:
            want += posterior(*biterms[b])
        want /= len(own)
        assert np.allclose(theta[doc], want, atol=1e-12), (doc, theta[doc], want)
    print("  ok  BTM p(z|d) is the normalised biterm posterior")


def test_btm_inference_does_not_depend_on_the_batch() -> None:
    """The inference path must sample biterms per document.

    The review's counter-example: with one shared random stream, a long
    document got different biterms (and a different posterior, ~0.014 in the
    largest component) depending on which other documents shared the batch.
    The test goes through the real adapter (text -> pairs -> posterior), not
    just through ``btm_doc_topic``.
    """
    import occlusion_robustness as O

    class _Dict:
        token2id = {"w0": 0, "w1": 1, "w2": 2, "w3": 3}

    long_doc = ["w0"] * 60 + ["w1"] * 60
    other = ["w2"] * 60 + ["w3"] * 60
    rec = {
        "beta": np.full((2, 4), 0.25), "n_bz": np.array([2.0, 2.0]),
        "n_wz": np.array([[3.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0]]),
        "alpha": 25.0, "beta_prior": 0.01, "window": 12, "max_pairs": 150,
    }
    adapter = O.BTMAdapter({"vocab": ["w0", "w1", "w2", "w3"]}, 2, 11,
                           rec=rec)
    adapter.dictionary = _Dict()
    alone = adapter.theta([long_doc], [[]])
    batch = adapter.theta([other, long_doc], [[], []])
    assert np.allclose(alone[0], batch[1]), (alone[0], batch[1])
    # and the legacy shared stream really is order dependent
    args = ({}, _Dict())
    del args
    shared_a, _ = BL._biterms_from_tokens([long_doc], _Dict(), 12, 150, seed=11)
    shared_b, _ = BL._biterms_from_tokens([other, long_doc], _Dict(), 12, 150,
                                          seed=11)
    assert shared_a != shared_b[-len(shared_a):]
    print("  ok  BTM inference is batch invariant (per-document sampling)")


def test_nmf_adapter_zero_input_uses_a_prior() -> None:
    """A real (small) NMF fit must return a distribution for a zero bag."""
    import scipy.sparse as sp

    import occlusion_robustness as O

    rng = np.random.RandomState(0)
    counts = rng.randint(0, 4, size=(40, 6)).astype(np.float32)
    train = {"reason_bow": sp.csr_matrix(counts), "item_bow": sp.csr_matrix(
        np.zeros_like(counts)), "vocab": [f"w{i}" for i in range(6)]}
    adapter = O.NMFAdapter(train, 3, 11)
    adapter.set_word2id({f"w{i}": i for i in range(6)})
    theta = adapter.theta([[]], [[]])            # no visible evidence at all
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert np.all(theta >= 0)
    assert np.allclose(theta[0], adapter.prior)
    print("  ok  NMFAdapter returns the corpus prior for a zero input")


def test_run_model_rejects_an_invalid_adapter() -> None:
    """The scoring entry point itself must refuse a non-distribution."""
    import occlusion_robustness as O

    class Broken(O.Adapter):
        name = "broken"

        def theta(self, visible_A, field_B):
            return np.zeros((len(visible_A), 2))          # not a distribution

        def log_prob(self, theta):
            beta = np.full((2, 3), 1.0 / 3.0)
            return np.log(np.clip(theta @ beta, 1e-12, None))

        def set_word2id(self, word2id):
            return

    visible = {0.0: [["a", "b"]], 1.0: [[]]}
    hidden = {0.0: [[]], 1.0: [["a", "b"]]}
    word2id = {"a": 0, "b": 1}
    try:
        O.run_model(Broken(), visible, [[]], hidden, [0.0, 1.0], word2id,
                    "broken")
    except ValueError as exc:
        assert "p(z|d)" in str(exc) or "distribution" in str(exc), exc
    else:
        raise AssertionError("run_model accepted an all-zero topic distribution")
    print("  ok  run_model rejects a non-distribution instead of scoring it")


def test_resume_requires_complete_rows() -> None:
    """An identity match alone must not mark a partition as finished."""
    import occlusion_robustness as O
    import run_public_missing_field as R
    import pandas as pd

    occ_ok = {"dataset": "github", "setting": "fielddrop_on", "seed": "11",
              "full_CV": "0.7", "occ_CV": "0.7", "delta_CV": "0.01",
              "delta_NLL_token": "0.2", "top1_agreement": "0.3",
              "mean_js": "0.08"}
    bg = {"dataset": "github", "setting": "fielddrop_on", "seed": "11",
          "view": "background", "CV": "0.7", "NPMI": "0.2", "n_topics": "30"}
    sm = dict(bg, view="summary", n_topics="20")
    keys = [("fielddrop_on", 11)]
    assert R.complete_partitions([], [], "github", keys) == set()
    assert R.complete_partitions([occ_ok], [bg], "github", keys) == set()
    assert R.complete_partitions([dict(occ_ok, delta_CV="nan")], [bg, sm],
                                 "github", keys) == set()
    assert R.complete_partitions([occ_ok], [bg, sm], "github",
                                 keys) == {("fielddrop_on", 11)}

    rates = [0.0, 0.25, 0.5, 0.75, 1.0]

    def row(model, rate, nll="7.0"):
        return {"model": model, "rate": rate, "hidden_NLL": nll,
                "agreement": "0.4", "mean_js": "0.1"}

    full = pd.DataFrame([row("LDA", r) for r in rates])
    assert O.complete_models(full, rates) == {"LDA"}
    only_last = pd.DataFrame([row("LDA", 1.0)])
    assert O.complete_models(only_last, rates) == set()
    missing_rate = pd.DataFrame([row("LDA", r) for r in rates[:-1]])
    assert O.complete_models(missing_rate, rates) == set()
    nan_row = pd.DataFrame([row("LDA", r, "nan" if r == 0.75 else "7.0")
                            for r in rates])
    assert O.complete_models(nan_row, rates) == set()
    duplicated = pd.DataFrame([row("LDA", r) for r in rates + [0.5]])
    assert O.complete_models(duplicated, rates) == set()
    print("  ok  resume needs every rate / both views, not just an identity")


def test_paper_checker_notices_a_tampered_table_value() -> None:
    """The review's counter-example: editing a printed number must fail.

    ``check_paper_numbers.py`` parses the LaTeX, so a value changed on the
    paper side has to be caught; ``--self-test`` tampers with a copy of
    main.tex and requires a non-zero exit code.
    """
    import check_paper_numbers as C

    rc = C.self_test()
    assert rc == 0, "the checker did not notice a tampered table value"
    print("  ok  tampering with a printed table value is detected")


def test_btm_and_nmf_never_score_a_non_distribution() -> None:
    """The two probability-boundary bugs of the third review round.

    A BTM document without biterms used to keep an all-zero topic row (every
    word probability clipped to 1e-12, NLL = 27.63) and an NMF document with no
    visible evidence did the same.  Both must fall back to a proper
    distribution, and single-document inference must not depend on the batch.
    """
    K, V = 2, 4
    alpha, beta = 50.0 / K, 0.01
    n_wz = np.array([[5.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
    n_bz = n_wz.sum(axis=1) / 2.0
    biterms, owners = [(0, 1)], [0]

    theta = BL.btm_doc_topic(biterms, owners, 2, n_bz, n_wz, alpha, beta, V)
    assert np.allclose(theta.sum(axis=1), 1.0), theta
    assert np.all(theta >= 0)
    # the document without a biterm gets the corpus topic prior
    prior = (n_bz + alpha) / np.sum(n_bz + alpha)
    assert np.allclose(theta[1], prior)
    # batch splitting must not change a single document's posterior
    alone = BL.btm_doc_topic(biterms, owners, 1, n_bz, n_wz, alpha, beta, V)
    assert np.allclose(alone[0], theta[0])
    # no biterm at all -> every row is still a distribution
    empty = BL.btm_doc_topic([], [], 3, n_bz, n_wz, alpha, beta, V)
    assert np.allclose(empty.sum(axis=1), 1.0)

    # the two posterior forms must be distinguishable on asymmetric counts
    point = BL.btm_doc_topic(biterms, owners, 1, n_bz, n_wz, alpha, beta, V,
                             form="point")
    pred = BL.btm_doc_topic(biterms, owners, 1, n_bz, n_wz, alpha, beta, V,
                            form="predictive")
    assert not np.allclose(point, pred, atol=1e-9), (point, pred)
    # the point form is the paper's phi_kw = (n_kw + beta)/(2 n_k + V beta)
    phi = (n_wz + beta) / (2.0 * n_bz[:, None] + V * beta)
    ref = (n_bz + alpha) * phi[:, 0] * phi[:, 1]
    assert np.allclose(point[0], ref / ref.sum())

    # a zero row anywhere is an error, not something to score
    try:
        BL.check_simplex(np.array([[0.5, 0.5], [0.0, 0.0]]), "test")
    except ValueError:
        pass
    else:
        raise AssertionError("check_simplex accepted an all-zero row")
    print("  ok  BTM/NMF fall back to a proper distribution (batch invariant)")


def test_readout_column_says_what_was_really_scored() -> None:
    """``actual_readout`` must differ from ``readout`` where it really does.

    ``--readout`` only re-reads the two non-parametric baselines (they have no
    topic-word matrix of their own); every other model is scored on its own
    beta matrix no matter what was requested.  The result rows now record both,
    plus a digest of the scored state.
    """
    import benchmark_public as BP
    import result_identity as RI

    for model in ("BERTopic", "Top2Vec"):
        assert BP.actual_readout(model, "ctfidf") == "ctfidf"
        assert BP.actual_readout(model, "native") == "native"
    for model in ("LDA", "BTM", "NMF", "ProdLDA", "ETM", "CTMLite",
                  "CombinedTM", "FC-NTM"):
        assert BP.actual_readout(model, "ctfidf") == "beta"
        assert BP.actual_readout(model, "native") == "beta"

    beta = np.arange(6, dtype=float).reshape(2, 3)
    theta = np.linspace(0.0, 1.0, 8).reshape(4, 2)
    assert RI.array_digest(beta, theta) == RI.array_digest(beta.copy(), theta)
    other = beta.copy()
    other[0, 0] += 1e-9
    assert RI.array_digest(other, theta) != RI.array_digest(beta, theta)
    assert RI.array_digest(None) == RI.array_digest(None)
    assert len(RI.array_digest(beta, theta)) == 16
    print("  ok  actual read-out recorded; state digest tracks the scored state")


def test_result_identity_refuses_unpinned_results() -> None:
    """A stored result is only reused when its configuration is pinned."""
    import tempfile
    from pathlib import Path

    import result_identity as RI

    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "rows.csv"
        result.write_text("model,seed\nFC-NTM,11\n", encoding="utf-8")
        ident = RI.identity({"script": "x", "k": 30, "seeds": [11, 23, 37]})

        # no sidecar yet: the file exists but is not pinned to anything
        assert not RI.matches(result, ident)
        assert "no identity" in RI.why_not(result, ident)

        RI.write(result, ident)
        assert RI.matches(result, ident)

        # same file, different configuration -> must not be reused
        other = RI.identity({"script": "x", "k": 31, "seeds": [11, 23, 37]})
        assert not RI.matches(result, other)
        assert "k" in RI.why_not(result, other)

        # and a changed code digest changes the identity as well
        assert RI.code_version(("config.py",)) != RI.code_version(("metrics.py",))

        # an unverified result is moved aside, never deleted
        stash = RI.stash_unpinned(result)
        assert stash is not None and stash.exists() and not result.exists()
        assert RI.stash_unpinned(result) is None
    print("  ok  result identity refuses unpinned / mismatched results")


def test_resume_identity_guard() -> None:
    """A resume state may only continue when its experiment identity matches.

    Reproduces the 2026-10-07 review counter-example: the same resume path
    is reused after the training target changes.  The guard must archive the
    stale state and start fresh instead of silently continuing it.
    """
    import tempfile
    from pathlib import Path

    import result_identity as RI

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.zeros(1))

    def make_loss(model, target):
        grid = torch.linspace(0.0, 1.0, 16)

        def loss_fn(idx, kl_weight):
            x = grid[idx]
            pred = model.scale.expand_as(x)
            return ((pred - target * x) ** 2).mean(), {}

        return loss_fn

    def run(path, payload, target, epochs):
        model = Tiny()
        ident = RI.identity(payload)
        info = NN.train_topic_vae_valid(
            model, make_loss(model, target), make_loss(model, target),
            n_docs=16, seed=11, val_fraction=0.25, warmup_epochs=1,
            kl_max=0.0, patience=1000, num_epochs=epochs, lr=1e-2,
            resume_path=str(path), resume_every=1, allow_resume=True,
            resume_identity=ident)
        return info, ident

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.pt"
        payload_a = {"script": "resume-guard", "target": 0.0, "lr": 1e-2}
        payload_b = {"script": "resume-guard", "target": 100.0, "lr": 1e-2}

        run(path, payload_a, 0.0, epochs=4)
        stored = torch.load(path, map_location="cpu", weights_only=False)
        assert stored["identity_id"] == RI.identity(payload_a)["id"]
        assert int(stored["epoch"]) == 4

        # changed objective, same path: archive and restart, never continue
        info_b, ident_b = run(path, payload_b, 100.0, epochs=1)
        assert int(info_b["epochs_run"]) == 1, info_b["epochs_run"]
        stored = torch.load(path, map_location="cpu", weights_only=False)
        assert stored["identity_id"] == ident_b["id"]
        assert list(Path(tmp).glob("state.pt.stale*")), "stale state not archived"

        # a matching identity resumes when the budget grows: after the
        # 1-epoch fresh run above, asking for 4 epochs continues to 4
        info_resume, _ = run(path, payload_b, 100.0, epochs=4)
        assert int(info_resume["epochs_run"]) == 4, info_resume["epochs_run"]

        # a smaller budget must not inherit the 4-epoch state: it archives it
        # and restarts from scratch at 2 (2026-10-11 review, item 5)
        info_short, _ = run(path, payload_b, 100.0, epochs=2)
        assert int(info_short["epochs_run"]) == 2, info_short["epochs_run"]
        assert list(Path(tmp).glob("state.pt.stale*")), \
            "stale state not archived for the smaller budget"
    print("  ok  resume identity guard archives changed experiments")


def test_ablation_resume_identity() -> None:
    """Row reuse must verify dataset, K, epoch budget and config hash."""
    import run_ablation_public as A

    expected = {"no-cross":
                A.config_hash({**A.FULL_CONFIG, **A.CONFIGS["no-cross"]})}

    def row(**kw):
        base = {"config": "no-cross", "seed": "11", "readout": "beta",
                "CV": "0.5", "NPMI": "0.1", "dataset": "github",
                "K_used": "30", "epochs_budget": "700",
                "config_hash": expected["no-cross"]}
        base.update(kw)
        return base

    kept, done = A.complete_pairs([row(), row(readout="cTFIDF")], expected,
                                  "github", 30, 700, ["no-cross"])
    assert len(kept) == 2 and done == {("no-cross", 11)}
    for field, value in (("dataset", "agnews"), ("K_used", "20"),
                         ("epochs_budget", "1"),
                         ("config_hash", "deadbeef")):
        kept, done = A.complete_pairs([row(**{field: value})], expected,
                                      "github", 30, 700, ["no-cross"])
        assert not kept and not done, field
    kept, done = A.complete_pairs([row(config="full")], expected,
                                  "github", 30, 700, ["no-cross"])
    assert not kept and not done, "unrequested config must be dropped"
    print("  ok  ablation resume verifies dataset/K/budget/config hash per row")


def test_embedding_cache_identity_and_windows() -> None:
    """Cache key covers revision/length/pooling; windows lose no token."""
    import embed_public as EP

    old_len, old_rev = EP.EMB_MAX_LENGTH, EP.EMB_REVISION
    try:
        base = EP._cache_path("doc", ["hello world"], False)
        EP.EMB_MAX_LENGTH = old_len + 128
        assert EP._cache_path("doc", ["hello world"], False) != base
        EP.EMB_MAX_LENGTH = old_len
        EP.EMB_REVISION = "deadbeef"
        assert EP._cache_path("doc", ["hello world"], False) != base
        EP.EMB_REVISION = old_rev
        assert EP._cache_path("doc", ["hello world"], True) != base
        for n in (0, 1, 126, 127, 253, 1000):
            ids = list(range(n))
            windows = EP.token_windows(ids, 128, 2)
            assert [i for w in windows for i in w] == ids, n
            assert all(len(w) <= 126 for w in windows), n
    finally:
        EP.EMB_MAX_LENGTH, EP.EMB_REVISION = old_len, old_rev
    print("  ok  embedding cache key covers revision/length/pooling; "
          "token windows lose no token")


def test_civil_ladder_keeps_full_ids() -> None:
    """The three ladder splits must cover the same documents."""
    import pandas as pd
    from pathlib import Path

    data = Path(__file__).resolve().parent / "data" / "civil_openalex"
    base = {name: pd.read_csv(data / f"{name}_full.csv")
            for name in ("train", "test")}
    for variant in ("abs60", "abs30"):
        for name, full in base.items():
            path = data / f"{name}_{variant}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            assert list(df["openalex_id"]) == list(full["openalex_id"]), \
                (variant, name)
    print("  ok  civil ladder keeps the full split's document IDs")


def test_reuse_details_identity() -> None:
    """A reused full-config dump must match the complete fitting identity."""
    import run_ablation_public as A
    import result_identity as RI

    want = {"k": 3, "n_vocab": 4, "n_docs": 6, "code": "abc", "seed": 11}
    base = {"model": "FC-NTM", "seed": 11, "config_hash": "h",
            "beta": [[0.0] * 4 for _ in range(2)],
            "theta": [[0.5, 0.5]] * 6}
    # the review's counter-example: 2 topics accepted for a K=3 request
    _beta, _theta, why = A.reusable_full_record(base, want, "h", 3, 4, 6)
    assert any("fit identity" in w for w in why), why
    assert any("beta shape" in w for w in why), why

    rec = dict(base)
    rec["beta"] = [[0.0] * 4 for _ in range(3)]
    rec["theta"] = [[0.5, 0.5, 0.0]] * 6
    rec["fit_identity"] = RI.identity(want)
    _beta, _theta, why = A.reusable_full_record(rec, want, "h", 3, 4, 6)
    assert why == [], why
    # a changed protocol field is refused even with matching shapes
    _beta, _theta, why = A.reusable_full_record(
        rec, dict(want, code="other"), "h", 3, 4, 6)
    assert any("code" in w for w in why), why
    # the outer record seed must equal the seed inside the fit identity
    wrong_seed = dict(rec)
    wrong_seed["seed"] = 12
    _beta, _theta, why = A.reusable_full_record(
        wrong_seed, want, "h", 3, 4, 6)
    assert any("seed" in w for w in why), why
    print("  ok  details reuse requires the complete fitting identity")


def test_ablation_entry_smoke() -> None:
    """run_ablation_public.main() must run for all three reuse branches."""
    import json
    import sys
    import tempfile
    from pathlib import Path

    import result_identity as RI
    import run_ablation_public as A

    vocab = ["w0", "w1", "w2", "w3"]
    n_docs, k = 6, 3

    def fake_corpus(n):
        toks = [[vocab[(i + j) % 4] for j in range(2)] for i in range(n)]
        return {
            "n_vocab": 4, "n_docs": n, "vocab": vocab,
            "reason_bow": np.ones((n, 4), dtype=np.float32),
            "item_bow": np.ones((n, 4), dtype=np.float32),
            "reason_tokens": toks, "item_tokens": toks,
            "doc_tokens": [a + b for a, b in zip(toks, toks)],
        }

    def fake_load(name, abstract_words=0):
        return {"train": fake_corpus(n_docs), "test": fake_corpus(2)}

    trained = {"flag": False}

    def fake_train(train, K, seed, overrides, lr, max_epochs=700):
        trained["flag"] = True
        return (np.ones((k, 4), dtype=float),
                np.full((n_docs, k), 1.0 / k))

    def fake_evaluate(topics, mat, corpus):
        return {"CV": 0.5, "CV_n": 1.0, "NPMI": 0.1, "NPMI_n": 1.0,
                "UCI": 0.0, "UCI_n": 1.0, "UMass": 0.0, "UMass_n": 1.0,
                "EXCL": 0.0, "EXCL_KADJ": 0.0, "IRBO": 0.0}

    def details_record(lr):
        train = fake_corpus(n_docs)
        import hashlib
        payload = {
            "k": k, "n_vocab": 4, "n_docs": n_docs, "seed": 11,
            "vocab_digest": hashlib.sha1(
                "\x1f".join(vocab).encode("utf-8")).hexdigest()[:16],
            "data": RI.text_digest([train["doc_tokens"]]),
            "fields": RI.text_digest([train["reason_tokens"],
                                      train["item_tokens"]]),
            "code": RI.code_version(),
            "field_dropout": float(A.FULL_CONFIG["field_dropout"]),
            "fusion": "gate",
            "fusion_norm": A.FULL_CONFIG["fused_recon_norm"],
            "fusion_weight": float(A.FULL_CONFIG["fused_recon_weight"]),
            "lr": lr, "epochs_budget": 2,
        }
        return {"model": "FC-NTM", "seed": 11,
                "config_hash": A.config_hash(dict(A.FULL_CONFIG)),
                "beta": np.ones((k, 4)).tolist(),
                "theta": np.full((n_docs, k), 1.0 / k).tolist(),
                "fit_identity": RI.identity(payload)}

    old = (A.load_dataset, A.train_model, A.evaluate)
    argv = sys.argv[:]
    A.load_dataset, A.train_model, A.evaluate = (
        fake_load, fake_train, fake_evaluate)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # (a) no details: trains
            trained["flag"] = False
            sys.argv = ["run_ablation_public.py", "--dataset", "github",
                        "--seeds", "11", "--k", str(k),
                        "--configs", "no-cross", "--max-epochs", "2",
                        "--out", str(tmp / "nodetails.csv")]
            A.main()
            assert trained["flag"], "no-details branch did not train"
            # (b) valid details for the full config: reused, no training
            valid = tmp / "valid.json"
            valid.write_text(json.dumps([details_record(5e-4)]),
                             encoding="utf-8")
            trained["flag"] = False
            sys.argv = ["run_ablation_public.py", "--dataset", "github",
                        "--seeds", "11", "--k", str(k), "--lr", "5e-4",
                        "--configs", "full", "--max-epochs", "2",
                        "--reuse-details", str(valid),
                        "--out", str(tmp / "valid.csv")]
            A.main()
            assert not trained["flag"], "valid details were not reused"
            # (c) rejected details (wrong lr): trains fresh
            bad = tmp / "bad.json"
            bad.write_text(json.dumps([details_record(0.123)]),
                           encoding="utf-8")
            trained["flag"] = False
            sys.argv = ["run_ablation_public.py", "--dataset", "github",
                        "--seeds", "11", "--k", str(k), "--lr", "5e-4",
                        "--configs", "full", "--max-epochs", "2",
                        "--reuse-details", str(bad),
                        "--out", str(tmp / "bad.csv")]
            A.main()
            assert trained["flag"], "bad details were not rejected"
    finally:
        A.load_dataset, A.train_model, A.evaluate = old
        sys.argv = argv
    print("  ok  ablation main() runs for no/valid/rejected details")


def test_eval_identity_covers_test_split() -> None:
    """The evaluation identity must move with the test side (C4)."""
    import seed_identity as SI

    def side(tokens):
        return {"vocab": ["a", "b", "c"], "doc_tokens": tokens,
                "reason_tokens": tokens, "item_tokens": tokens}

    train = side([["a", "b"], ["c", "a"]])
    test_a = side([["a", "c"], ["b", "c"]])
    test_b = side([["a", "c"], ["b", "b"]])
    seeds = [11, 23]
    base = SI.corpus_identity("unit", "github", train, test_a, seeds, 3)
    assert base["payload"]["rates"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert base["payload"]["mask"], "mask protocol missing"
    assert base["payload"]["test_data"], "test split missing"
    assert SI.corpus_identity("unit", "github", train, test_b, seeds,
                              3)["id"] != base["id"], "test split not covered"
    assert SI.corpus_identity("unit", "github", train, test_a, [11],
                              3)["id"] != base["id"], "seed list not covered"
    occ = SI.corpus_identity("unit", "github", train, test_a, seeds, 3,
                             extra={"state_digest": "deadbeef"})
    other = SI.corpus_identity("unit", "github", train, test_a, seeds, 3,
                               extra={"state_digest": "cafebabe"})
    assert occ["id"] != other["id"], "checkpoint digest not covered"
    print("  ok  evaluation identity covers test split, seeds and checkpoint")


def test_state_dir_identity_uses_the_actual_checkpoint() -> None:
    """--state-dir must be the file whose digest lands in the identity (v6)."""
    import sys
    import tempfile
    from pathlib import Path

    import extend_seeds_fcntm_occlusion as E
    import result_identity as RI

    class FakeDict(dict):
        token2id = {"w": 0}

    def fake_corpus():
        return {"vocab": ["w"], "n_vocab": 1, "n_docs": 2,
                "doc_tokens": [["w"], ["w"]],
                "reason_tokens": [["w"], ["w"]],
                "item_tokens": [[], []], "dictionary": FakeDict()}

    captured = {}

    class FakeAdapter:
        def __init__(self, path, vocab, k, seed=None):
            captured["state"] = Path(path)

    def fake_merge(path, rows, ident, keys):
        captured["ident"] = ident
        captured["rows"] = list(rows)

    old = (E.O.load_corpus, E.O.FCNTMAdapter, E.O.run_model, E.masks_for,
           E.merge_pinned, E.verify_checkpoint, E.state_path)
    checked = {}
    E.O.load_corpus = lambda corpus: (fake_corpus(), fake_corpus())
    E.O.FCNTMAdapter = FakeAdapter
    E.O.run_model = lambda *a, **k: [{"model": "FC-NTM", "rate": 0.25,
                                      "hidden_NLL": 1.0, "n_docs": 1}]
    E.masks_for = lambda test, seed: ({0.25: [["w"]]}, {0.25: [["w"]]})
    E.merge_pinned = fake_merge
    E.verify_checkpoint = lambda path, train, seeds, k: checked.update(
        path=Path(path), seeds=list(seeds), k=k)
    argv = sys.argv[:]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            default = tmp / "default.pt"
            alt = tmp / "alt" / "fcntm_main_state_github.pt"
            alt.parent.mkdir()
            default.write_bytes(b"default-state")
            alt.write_bytes(b"alternate-state")
            E.state_path = lambda corpus: default
            sys.argv = ["extend_seeds_fcntm_occlusion.py", "--corpora",
                        "github", "--seeds", "53", "--state-dir",
                        str(alt.parent)]
            E.main()
            alt_digest = RI.file_digest(alt)
            default_digest = RI.file_digest(default)
            assert alt_digest != default_digest
    finally:
        (E.O.load_corpus, E.O.FCNTMAdapter, E.O.run_model, E.masks_for,
         E.merge_pinned, E.verify_checkpoint, E.state_path) = old
        sys.argv = argv
    assert checked.get("path") == alt, checked
    assert captured.get("state") == alt, captured.get("state")
    payload = captured["ident"]["payload"]
    assert payload["state_digest"] == alt_digest, payload
    assert payload["state_digest"] != default_digest, payload
    print("  ok  --state-dir digest follows the file that was scored")


def test_concat_validation_is_deterministic() -> None:
    """2026-10-14 review C1: validation must not consume sampling noise.

    FCNTMConcat.train_step used to force ``sample=True`` regardless of the
    module mode, so the validation loss of the concatenated-field control
    fluctuated between repeated calls and early stopping saw a different
    protocol than the main FC-NTM.
    """
    torch.manual_seed(0)
    model = NN.FCNTMConcat(48, 6, cause_dropout=0.3, item_dropout=0.4,
                           field_dropout=0.5)
    a = NN._normalize_rows((torch.rand(16, 48) < 0.2).float())
    b = NN._normalize_rows((torch.rand(16, 48) < 0.2).float())
    ac = (torch.rand(16, 48) < 0.2).float()
    bc = (torch.rand(16, 48) < 0.2).float()
    hc = torch.ones(16, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        losses = [float(model.train_step(a, b, ac, bc, hc, 0.05)[0])
                  for _ in range(5)]
    assert max(losses) == min(losses), losses
    model.train()
    seen = set()
    for _ in range(20):
        seen.add(round(float(model.train_step(a, b, ac, bc, hc, 0.05)[0]), 6))
    assert len(seen) > 1, "training step is unexpectedly deterministic"
    print("  ok  concat validation is deterministic, training still samples")


def test_concat_reuse_requires_matching_identity() -> None:
    """2026-10-15 v2/v3 review: fit identity, eval identity and row checks."""
    import run_concat_control as RC

    fit = {"corpus": "civil", "vocab_digest": "abcd1234", "n_vocab": 100,
           "k": 30, "code": "cafe", "lr": 5e-4, "num_epochs": 700,
           "field_dropout": 0.5, "data": "d1", "fields": "f1"}
    ev = {"corpus": "civil", "test_data": "t1", "test_fields": "tf1",
          "readouts": ["beta", "ctfidf"], "k": 30}
    row = dict(seed=11, CV=0.5, NPMI=0.1, UMass=-2.0, EXCL_KADJ=0.1,
               IRBO=0.9)
    rows = [dict(row, readout="beta"), dict(row, readout="ctfidf")]
    rec = {"identity": dict(fit), "eval_identity": dict(ev), "rows": rows}
    assert RC.reusable_rows(rec, dict(fit), dict(ev), 11) == rows
    assert RC.reusable_rows(rec, dict(fit, k=31), dict(ev), 11) is None
    assert RC.reusable_rows(rec, dict(fit, corpus="github"), dict(ev),
                            11) is None
    assert RC.reusable_rows(rec, dict(fit, vocab_digest="9999"), dict(ev),
                            11) is None
    assert RC.reusable_rows(rec, dict(fit, code="beef"), dict(ev),
                            11) is None
    assert RC.reusable_rows(rec, dict(fit), dict(ev, test_data="t2"),
                            11) is None
    assert RC.reusable_rows(rec, dict(fit), dict(ev), 23) is None
    # legacy records missing newer fit fields must not be reused
    for missing in ("lr", "num_epochs", "field_dropout", "data", "fields"):
        broken = dict(fit)
        broken.pop(missing)
        assert RC.reusable_rows(
            {"identity": broken, "eval_identity": dict(ev), "rows": rows},
            dict(fit), dict(ev), 11) is None, missing
    # invalid rows: non-finite NPMI / mixed seeds / single read-out
    bad = [dict(row, readout="beta", NPMI=float("nan")),
           dict(row, readout="ctfidf")]
    assert RC.reusable_rows(
        {"identity": dict(fit), "eval_identity": dict(ev), "rows": bad},
        dict(fit), dict(ev), 11) is None
    mixed = [dict(row, seed=11, readout="beta"),
             dict(row, seed=23, readout="ctfidf")]
    assert RC.reusable_rows(
        {"identity": dict(fit), "eval_identity": dict(ev), "rows": mixed},
        dict(fit), dict(ev), 11) is None
    single = [dict(row, readout="beta")]
    assert RC.reusable_rows(
        {"identity": dict(fit), "eval_identity": dict(ev), "rows": single},
        dict(fit), dict(ev), 11) is None
    # 2026-10-15 v4 review R2: rows without the metric columns are not a
    # valid cache entry
    stripped = [{"seed": 11, "readout": "beta"},
                {"seed": 11, "readout": "ctfidf"}]
    assert RC.reusable_rows(
        {"identity": dict(fit), "eval_identity": dict(ev),
         "rows": stripped}, dict(fit), dict(ev), 11) is None
    print("  ok  concat reuse checks fit/eval identity and row validity")


def test_completed_resume_does_not_train() -> None:
    """2026-10-15 v3 review C2: a finished state must not train further."""
    import tempfile
    from pathlib import Path as _P

    torch.manual_seed(0)
    calls = {"n": 0}

    def make_loss(model):
        def loss_fn(idx, kl_weight):
            calls["n"] += 1
            x = torch.zeros(len(idx), 16)
            x[:, 0] = 1.0
            hc = torch.ones(len(idx), dtype=torch.bool)
            return model.train_step(x, x, x, x, hc, kl_weight)
        return loss_fn

    with tempfile.TemporaryDirectory() as td:
        rp = str(_P(td) / "resume.pt")
        ident = {"id": "TESTID", "payload": {}}
        model = NN.FCNTM(16, 3, field_dropout=0.0)
        fn = make_loss(model)
        info = NN.train_topic_vae_valid(
            model, fn, fn, 24, 11, val_fraction=0.25, warmup_epochs=1,
            patience=1, num_epochs=2, resume_path=rp, resume_every=1,
            resume_identity=ident)
        assert not info["early_stopped"]
        calls["n"] = 0
        info2 = NN.train_topic_vae_valid(
            model, fn, fn, 24, 11, val_fraction=0.25, warmup_epochs=1,
            patience=1, num_epochs=2, resume_path=rp, resume_every=1,
            resume_identity=ident)
        assert calls["n"] == 0, calls
        assert info2["epochs_run"] == info["epochs_run"]
        assert info2["best_epoch"] == info["best_epoch"]
        assert info2["early_stopped"] is False, "budget resume mislabelled"
        # crafted early-stopped state: also a no-op on resume
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        ck["completed"] = True
        ck["stop_reason"] = "early_stopped"
        ck["patience_counter"] = 1
        ck["identity_id"] = "X2"
        ck["identity"] = {"id": "X2"}
        rp2 = str(_P(td) / "resume2.pt")
        torch.save(ck, rp2)
        calls["n"] = 0
        model3 = NN.FCNTM(16, 3, field_dropout=0.0)
        fn3 = make_loss(model3)
        info3 = NN.train_topic_vae_valid(
            model3, fn3, fn3, 24, 11, val_fraction=0.25, warmup_epochs=1,
            patience=1, num_epochs=10, resume_path=rp2, resume_every=1,
            resume_identity={"id": "X2"})
        assert calls["n"] == 0, calls
        assert info3["early_stopped"]
    print("  ok  completed or early-stopped resumes perform no training")


def test_token_recon_keeps_field_weights() -> None:
    """2026-10-15 v3 review C3: per-token mode must keep field weights."""
    torch.manual_seed(0)
    model = NN.FCNTM(12, 3, field_dropout=0.0, recon_norm="token")
    model.eval()
    x = (torch.rand(6, 12) < 0.3).float()
    hc = torch.ones(6, dtype=torch.bool)

    def loss():
        with torch.no_grad():
            return float(model.train_step(x, x, x, x, hc, 0.0)[0])

    model.field_weights = (1.0, 1.0)
    a = loss()
    model.field_weights = (9.0, 1.0)
    b = loss()
    assert abs(a - b) > 1e-6, (a, b)
    print("  ok  per-token reconstruction keeps the field weights")


def test_cached_split_size_mismatch_raises() -> None:
    """2026-10-15 v3 review C4: request/cached size mismatch must fail."""
    import tempfile
    from pathlib import Path as _P

    import public_data as PD

    with tempfile.TemporaryDirectory() as td:
        old = PD.PUBLIC_DIR
        PD.PUBLIC_DIR = _P(td)
        try:
            hdr = "label,title,description\n"
            (PD.PUBLIC_DIR / "train.csv").write_text(hdr, encoding="utf-8")
            (PD.PUBLIC_DIR / "test.csv").write_text(hdr, encoding="utf-8")
            try:
                PD.prepare_splits(train_per_class=2, test_per_class=1)
            except ValueError:
                pass
            else:
                raise AssertionError("size mismatch passed silently")
        finally:
            PD.PUBLIC_DIR = old
    print("  ok  cached split size mismatch is rejected")


def test_ladder_branch_size_mismatch_raises() -> None:
    """2026-10-15 v4 review R1: the ladder keeps the fixed base IDs."""
    import tempfile
    from pathlib import Path as _P

    import public_data_civil as PC

    with tempfile.TemporaryDirectory() as td:
        hdr = "openalex_id,title,abstract,abstract_full\n"
        rows = "".join(f"W{i},title text,abstract text,abstract text\n"
                       for i in range(20))
        for name in ("train_full.csv", "test_full.csv"):
            (_P(td) / name).write_text(hdr + rows, encoding="utf-8")
        old = PC.CIVIL_DIR
        PC.CIVIL_DIR = _P(td)
        try:
            try:
                PC.prepare_civil_splits(train_n=12, test_n=4,
                                        max_abstract_words=30)
            except ValueError:
                pass
            else:
                raise AssertionError("ladder size mismatch passed silently")
        finally:
            PC.CIVIL_DIR = old
    print("  ok  truncation ladder rejects a mismatched request size")


def test_summary_conflict_rows_are_rejected() -> None:
    """2026-10-15 review: conflicting same-key rows fail before de-dup."""
    import tempfile
    from pathlib import Path as _P

    import summarize_civil_extension as SCE

    header = "model,seed,readout,CV,NPMI\n"
    with tempfile.TemporaryDirectory() as td:
        a = _P(td) / "a.csv"
        b = _P(td) / "b.csv"
        a.write_text(header + "BTM,11,beta,0.5,0.1\n", encoding="utf-8")
        b.write_text(header + "BTM,11,beta,0.123,0.1\n", encoding="utf-8")
        problems = []
        SCE._conflict_rows([a, b], ["model", "seed", "readout"],
                           "BTM sources", problems)
        assert problems, "conflicting duplicate passed silently"
        c = _P(td) / "c.csv"
        c.write_text(header + "BTM,11,beta,0.5,0.1\n", encoding="utf-8")
        problems = []
        SCE._conflict_rows([a, c], ["model", "seed", "readout"],
                           "BTM sources", problems)
        assert not problems, problems
    print("  ok  conflicting duplicates are rejected before de-duplication")


def main() -> None:
    print("review-fix checks")
    test_btm_conditional_matches_dirichlet_multinomial()
    test_btm_same_word_terms_differ()
    test_btm_doc_topic_is_the_normalised_biterm_posterior()
    test_btm_and_nmf_never_score_a_non_distribution()
    test_btm_inference_does_not_depend_on_the_batch()
    test_nmf_adapter_zero_input_uses_a_prior()
    test_run_model_rejects_an_invalid_adapter()
    test_resume_requires_complete_rows()
    test_resume_identity_guard()
    test_ablation_resume_identity()
    test_reuse_details_identity()
    test_ablation_entry_smoke()
    test_eval_identity_covers_test_split()
    test_state_dir_identity_uses_the_actual_checkpoint()
    test_embedding_cache_identity_and_windows()
    test_civil_ladder_keeps_full_ids()
    test_paper_checker_notices_a_tampered_table_value()
    test_readout_column_says_what_was_really_scored()
    test_result_identity_refuses_unpinned_results()
    test_irbo_range_and_bounds()
    test_etm_decoder_is_a_word_probability_mixture()
    test_fcntm_decodes_each_stream_once()
    test_rollback_clears_optimizer_state()
    test_benchmark_resume_logic()
    test_single_active_field_does_not_break_batchnorm()
    test_summary_groups_by_full_run_key()
    test_concat_validation_is_deterministic()
    test_concat_reuse_requires_matching_identity()
    test_summary_conflict_rows_are_rejected()
    test_completed_resume_does_not_train()
    test_token_recon_keeps_field_weights()
    test_cached_split_size_mismatch_raises()
    test_ladder_branch_size_mismatch_raises()
    print("all checks passed")


if __name__ == "__main__":
    main()
