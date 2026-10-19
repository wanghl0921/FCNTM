# -*- coding: utf-8 -*-
"""Partial field occlusion: what happens when part of field A disappears?

The whole-field occlusion experiment (``run_public_missing_field.py``) only
looks at the two extremes (all of A present vs all of A gone) and measures
topic-word coherence, which turned out to be insensitive.  This script measures
the behaviour that the field-collaborative design actually targets:

    mask a fraction p of field A (background) *tokens*,
    infer the topic mixture from what is left (visible A + field B),
    and ask how well the model can still predict the masked-out tokens.

Two curves are produced per model:

  hidden_NLL(p)   negative log-likelihood of the occluded A tokens under the
                  topic mixture inferred from the visible input.  Lower is
                  better; a model that exploits field B to compensate for the
                  missing part of A keeps this curve flat.
  agreement(p)    fraction of test documents whose most probable topic is
                  unchanged relative to the unoccluded inference, plus the
                  mean JS divergence between the two mixtures (stability).

Every model is scored with its *own* inference path and its own word
distribution, so no read-out convention is imposed across models:

  FC-NTM      q(z | x_A^visible, x_B)  (the model is trained with field dropout)
  ProdLDA     q(z | [x_A^visible ; x_B])
  ETM         q(z | [x_A^visible ; x_B])
  LDA         posterior of the fitted gensim model on the masked bag
  NMF         projection of the masked count vector on the fitted components

Usage:
    python occlusion_robustness.py --corpus github --seed 11 \
        --models FC-NTM,LDA,NMF,ProdLDA,ETM --rates 0,0.25,0.5,0.75,1.0
    python occlusion_robustness.py --corpus github --seed 11 --models LDA,NMF
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

import baselines as BL
import metrics as M
import result_identity as RI
from config import OUT_DIR, TOPN
from run_public_missing_field import final_model_kwargs


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def load_corpus(name: str):
    if name == "agnews":
        import public_data as PD
        data = PD.load_agnews(1500, 1900, need_doc_bow=False)
    elif name == "civil":
        import public_data_civil as PC
        data = PC.load_civil(6000, 2000, need_doc_bow=False)
    else:
        import public_data_github as PGH
        data = PGH.load_github(1500, 500, need_doc_bow=False)
    return data["train"], data["test"]


def mask_tokens(tokens, rate, rng):
    """Return (visible, hidden) token lists for an occlusion rate."""
    if not tokens:
        return [], []
    n = len(tokens)
    n_hidden = int(round(rate * n))
    if n_hidden <= 0:
        return list(tokens), []
    if n_hidden >= n:
        return [], list(tokens)
    pos = rng.choice(n, size=n_hidden, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[pos] = True
    visible = [t for i, t in enumerate(tokens) if not mask[i]]
    hidden = [t for i, t in enumerate(tokens) if mask[i]]
    return visible, hidden


# ---------------------------------------------------------------------------
# model adapters: each returns theta (n_docs, K) for a masked test corpus plus
# a callable giving log p(w | theta) for a batch of documents
# ---------------------------------------------------------------------------


class Adapter:
    name = "?"

    def theta(self, visible_A, field_B):
        raise NotImplementedError

    def log_prob(self, theta):
        raise NotImplementedError


class FCNTMAdapter(Adapter):
    name = "FC-NTM"

    def __init__(self, state_path, vocab_size, K, seed=None, device="cpu"):
        import models as NN
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        key = f"fielddrop_on:{seed}" if seed is not None else None
        if key not in ck:
            raise KeyError(
                f"{state_path} has no trained state for seed {seed} "
                f"(available: {sorted(ck)})")
        rec = ck[key]
        kwargs = rec.get("kwargs") or final_model_kwargs(0.5, "gate",
                                                         "balanced", 0.5)
        self.model = NN.FCNTM(vocab_size, K, **kwargs)
        self.model.load_state_dict(rec["model_state"])
        self.model.eval()
        self.key = key

    def theta(self, visible_A, field_B):
        import models as NN
        A = _bow(visible_A, self.model.vocab_size, self.word2id)
        B = _bow(field_B, self.model.vocab_size, self.word2id)
        A = torch.from_numpy(A)
        B = torch.from_numpy(B)
        has_cause = torch.tensor(
            [len(t) > 0 for t in visible_A], dtype=torch.bool)
        with torch.no_grad():
            fused, _ = self.model.encode_doc_theta(
                NN._normalize_rows(A), NN._normalize_rows(B), has_cause)
        return fused.numpy()

    def log_prob(self, theta):
        with torch.no_grad():
            logits = self.model.decode(torch.from_numpy(theta).float())
            return torch.log_softmax(logits, dim=-1).numpy()

    def set_word2id(self, word2id):
        self.word2id = word2id


class ProdLDAAdapter(Adapter):
    name = "ProdLDA"

    def __init__(self, train, K, seed, doc_bow):
        import experiment as EX
        rec = EX._run_prodlda(train, K, seed, doc_bow)
        self.model = rec["model"]
        self.model.eval()

    def theta(self, visible_A, field_B):
        import models as NN
        x = _bow([a + b for a, b in zip(visible_A, field_B)],
                 self.model.vocab_size, self.word2id)
        x = torch.from_numpy(x)
        with torch.no_grad():
            theta, _, _ = self.model.encode_theta(
                NN._normalize_rows(x), sample=False)
        return theta.numpy()

    def log_prob(self, theta):
        with torch.no_grad():
            logits = self.model.decoder(torch.from_numpy(theta).float())
            return torch.log_softmax(logits, dim=-1).numpy()

    def set_word2id(self, word2id):
        self.word2id = word2id


class ETMAdapter(ProdLDAAdapter):
    name = "ETM"

    def __init__(self, train, K, seed, word_emb, doc_bow):
        import experiment as EX
        rec = EX._run_entm(train, K, seed, doc_bow, word_emb)
        self.model = rec["model"]
        self.model.eval()

    def log_prob(self, theta):
        with torch.no_grad():
            return self.model.decoder.log_prob(
                torch.from_numpy(theta).float()).numpy()


class LDAAdapter(Adapter):
    name = "LDA"

    def __init__(self, train, K, seed):
        import baselines as BL
        rec = BL.run_lda(train["corpus"], train["dictionary"], train["vocab"],
                         K, seed)
        self.model = rec["model"] if "model" in rec else None
        self.beta = rec["beta"]

    def theta(self, visible_A, field_B):
        out = np.zeros((len(visible_A), self.beta.shape[0]))
        for i, (a, b) in enumerate(zip(visible_A, field_B)):
            bow = self.dictionary.doc2bow(a + b)
            for k, p in self.model.get_document_topics(
                    bow, minimum_probability=0.0):
                out[i, k] = p
        return out

    def log_prob(self, theta):
        p = np.clip(theta @ self.beta, 1e-12, None)
        return np.log(p)

    def set_word2id(self, word2id):
        # LDA works on the gensim dictionary, which the caller injects
        # separately (see ``adapter.dictionary`` in main)
        return


class BTMAdapter(Adapter):
    """Biterm topic model: refit the biterms on the surviving text and reuse
    the fitted Gibbs counts to recompute ``p(z|d)`` (Yan et al. §3.3)."""

    name = "BTM"

    def __init__(self, train, K, seed, rec=None):
        import baselines as BL
        if rec is None:
            rec = BL.run_btm(train["doc_tokens"], train["dictionary"],
                             train["vocab"], K, seed)
        self.beta = rec["beta"]
        self.n_bz = rec["n_bz"]
        self.n_wz = rec["n_wz"]
        self.alpha = rec["alpha"]
        self.beta_prior = rec["beta_prior"]
        self.window = rec["window"]
        self.max_pairs = rec["max_pairs"]
        self.seed = seed
        self.K = int(self.n_bz.shape[0])
        self.V = len(train["vocab"])

    def theta(self, visible_A, field_B):
        import baselines as BL
        docs = [a + b for a, b in zip(visible_A, field_B)]
        pairs, owners = BL._biterms_from_tokens(
            docs, self.dictionary, self.window,
            max_pairs=self.max_pairs, seed=self.seed,
            per_document_rng=True)
        # documents (or whole batches) without a biterm fall back to the
        # corpus-level topic prior inside ``btm_doc_topic``, so the result no
        # longer depends on which other documents share the batch
        return BL.btm_doc_topic(pairs, owners, len(docs), self.n_bz, self.n_wz,
                                self.alpha, self.beta_prior, self.V)

    def log_prob(self, theta):
        p = np.clip(theta @ self.beta, 1e-12, None)
        return np.log(p)

    def set_word2id(self, word2id):
        # BTM works on the gensim dictionary, injected by the caller
        return


class NMFAdapter(Adapter):
    name = "NMF"

    def __init__(self, train, K, seed):
        import scipy.sparse as sp
        from sklearn.decomposition import NMF
        # sparse sum: adding the two dense (docs x V) matrices would need an
        # extra 118 MB temporary on top of the two field matrices
        counts = (sp.csr_matrix(train["reason_bow"])
                  + sp.csr_matrix(train["item_bow"])).astype(np.float32)
        self.model = NMF(n_components=K, init="nndsvda",
                         beta_loss="kullback-leibler", solver="mu",
                         max_iter=600, random_state=seed, tol=1e-4)
        self.model.fit(counts)
        H = np.asarray(self.model.components_, dtype=float)
        # topic-word distribution and the per-topic total mass of H.  The plain
        # ``normalize(W) @ normalize(H)`` this adapter used to compute is *not*
        # the NMF reconstruction: rescaling one column of W and the matching
        # row of H leaves W H unchanged but changes that product completely.
        # Keeping the topic masses lets us rebuild normalize(W H) exactly (see
        # ``theta``), which is the quantity the model actually predicts.
        self.beta = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-12)
        self.topic_mass = H.sum(axis=1)
        # corpus-level topic prior: the mean mixture of the *training* documents
        # (in the same mass-weighted scale as ``theta`` below).  A visible bag
        # with no evidence at all projects to W = 0, which would otherwise make
        # the row all-zero and every word probability 1e-12.
        w_train = np.clip(np.asarray(self.model.transform(counts)), 0, None)
        train_mix = w_train * self.topic_mass[None, :]
        total = train_mix.sum(axis=0)
        self.prior = (total / total.sum() if total.sum() > 0
                      else np.full(H.shape[0], 1.0 / H.shape[0]))

    def theta(self, visible_A, field_B):
        import baselines as BL
        import scipy.sparse as sp
        m = _bow([a + b for a, b in zip(visible_A, field_B)],
                 self.beta.shape[1], self.word2id)
        w = self.model.transform(sp.csr_matrix(m.astype(np.float32)))
        w = np.clip(w, 0, None)
        # the document mixture is proportional to W_dk * sum_w H_kw, so that
        # ``theta @ beta`` equals normalize(W H) row-wise
        mix = w * self.topic_mass[None, :]
        s = mix.sum(axis=1, keepdims=True)
        theta = mix / np.maximum(s, 1e-12)
        empty = s[:, 0] <= 0
        if np.any(empty):
            # no visible evidence: fall back to the training topic prior, the
            # same convention BTM uses for documents without biterms
            theta[empty] = self.prior
        return BL.check_simplex(theta, "NMF theta")

    def log_prob(self, theta):
        p = np.clip(theta @ self.beta, 1e-12, None)
        return np.log(p)

    def set_word2id(self, word2id):
        self.word2id = word2id


def _bow(token_lists, n_vocab, word2id):
    m = np.zeros((len(token_lists), n_vocab), dtype=np.float32)
    for d, ts in enumerate(token_lists):
        for w in ts:
            j = word2id.get(w)
            if j is not None:
                m[d, j] += 1.0
    return m


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------


def top1(theta):
    return np.argmax(theta, axis=1)


def sym_js_np(a, b):
    m = 0.5 * (a + b)
    out = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        pa, pb, pm = a[i], b[i], m[i]
        kl = 0.0
        for p, q in ((pa, pm), (pb, pm)):
            mask = p > 1e-12
            kl += 0.5 * float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
        out[i] = kl
    return out


def complete_models(frame, rates) -> set:
    """Models whose stored curve really covers every requested rate.

    The review showed that the old check ("the largest rate in the file reaches
    the target" plus "the rows that exist are finite") skipped a model whose
    file held a single 100% row: the missing 0/25/50/75% points were never
    computed.  A model now counts as finished only when

      * every requested rate has exactly one row,
      * the scored rates (rate > 0) have finite NLL, agreement and JS.

    Rate 0 has no hidden token by construction, so its NLL is empty by design.
    """
    import pandas as pd

    if not {"model", "rate"} <= set(frame.columns):
        return set()
    want = {round(float(r), 6) for r in rates}
    done = set()
    for name, group in frame.groupby("model"):
        seen = [round(float(r), 6) for r in group["rate"].tolist()]
        if len(seen) != len(set(seen)) or not want <= set(seen):
            continue
        indexed = group.set_index(group["rate"].round(6))
        ok = True
        for rate in want:
            if rate == 0.0:
                continue
            row = indexed.loc[rate]
            for column in ("hidden_NLL", "agreement", "mean_js"):
                if column not in group.columns:
                    continue
                value = pd.to_numeric(pd.Series([row[column]]),
                                      errors="coerce").iloc[0]
                if not np.isfinite(value):
                    ok = False
        if ok:
            done.add(name)
    return done


def run_model(adapter, visible_A, field_B, hidden_A, rates, word2id, label):
    adapter.set_word2id(word2id)
    rows = []
    base_theta = None
    for rate in rates:
        A = visible_A[rate]
        B = field_B
        theta = BL.check_simplex(np.asarray(adapter.theta(A, B), dtype=float),
                                 f"{label} p(z|d) @ rate={rate:g}")
        if rate == 0.0:
            base_theta = theta
        lp = np.asarray(adapter.log_prob(theta), dtype=float)
        # the scored quantity is exp(log_prob): it has to be a finite,
        # non-negative, normalised word distribution, otherwise a broken
        # adapter silently turns every token into -log(1e-12) = 27.63
        probs = np.exp(np.clip(lp, -700, 0))
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError(f"{label}: invalid word probabilities at "
                             f"rate={rate:g}")
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            bad = int(np.argmax(np.abs(row_sums - 1.0)))
            raise ValueError(f"{label}: word distribution of document {bad} "
                             f"sums to {row_sums[bad]:.6g} at rate={rate:g}")
        nll, n_docs = 0.0, 0
        for i, hid in enumerate(hidden_A[rate]):
            if not hid:
                continue
            idx = [word2id[w] for w in hid if w in word2id]
            if not idx:
                continue
            nll += float(-lp[i, idx].sum() / len(idx))
            n_docs += 1
        row = {"model": label, "rate": rate,
               "hidden_NLL": (nll / n_docs) if n_docs else float("nan"),
               "n_docs": n_docs}
        if base_theta is not None:
            row["agreement"] = float(np.mean(
                top1(theta) == top1(base_theta)))
            row["mean_js"] = float(sym_js_np(theta, base_theta).mean())
        rows.append(row)
        print(f"  {label} rate={rate:g} hidden_NLL={row['hidden_NLL']:.4f} "
              f"agree={row.get('agreement', float('nan')):.3f}", flush=True)
        del theta, lp
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--models", default="FC-NTM,LDA,NMF,ProdLDA,ETM")
    ap.add_argument("--rates", default="0,0.25,0.5,0.75,1.0")
    # the FC-NTM state has to match the corpus (different vocabulary sizes);
    # an explicit path still overrides it (see the resolution right below)
    ap.add_argument("--fcntm-state", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.fcntm_state:
        args.fcntm_state = str(OUT_DIR / f"{args.corpus}_missing_state.pt")

    rates = [float(r) for r in args.rates.split(",")]
    train, test = load_corpus(args.corpus)
    vocab = test["vocab"]
    word2id = test["dictionary"].token2id
    n_vocab = len(vocab)
    models = [m for m in args.models.split(",") if m]
    rng = np.random.RandomState(args.seed + 7)

    # mask the background field once per rate, on the test split
    visible_A, hidden_A = {}, {}
    seeds = rng.randint(0, 10 ** 6, size=len(test["reason_tokens"]))
    for rate in rates:
        vis, hid = [], []
        for i, toks in enumerate(test["reason_tokens"]):
            v, h = mask_tokens(toks, rate, np.random.RandomState(seeds[i]))
            vis.append(v)
            hid.append(h)
        visible_A[rate] = vis
        hidden_A[rate] = hid
    field_B = test["item_tokens"]

    rows = []
    out_path = OUT_DIR / (args.out or
                          f"occlusion_robustness_{args.corpus}_seed{args.seed}.csv")
    ident = RI.identity({
        "script": "occlusion_robustness.main",
        "corpus": args.corpus,
        "seed": int(args.seed),
        "k": int(args.k),
        "models": sorted(models),
        "rates": [float(r) for r in rates],
        # the *content* of the checkpoint, not its file name: replacing the
        # state file with another model used to keep the identity unchanged
        "fcntm_state": RI.file_digest(args.fcntm_state),
        "code": RI.code_version(),
        "data": RI.text_digest([test["reason_tokens"], test["item_tokens"]]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens"),
                                  test["reason_tokens"], test["item_tokens"]]),
    })
    done_models = set()
    reuse_ok = out_path.exists() and RI.matches(out_path, ident)
    if out_path.exists() and out_path.stat().st_size and not reuse_ok:
        print(f"  stored curve {out_path.name} is not pinned to this "
              f"configuration ({RI.why_not(out_path, ident)}); recomputing "
              f"every model", flush=True)
        stash = RI.stash_unpinned(out_path)
        if stash:
            print(f"  old curve moved to {stash}", flush=True)
    if reuse_ok:
        try:
            import pandas as pd
            prev = pd.read_csv(out_path)
            done_models = complete_models(prev, rates)
            if done_models:
                rows = prev[prev["model"].isin(done_models)].to_dict("records")
                print("resuming, already done:", sorted(done_models), flush=True)
        except Exception:  # noqa: BLE001
            done_models, rows = set(), []
    doc_bow_tensor = None
    for name in models:
        if name in done_models:
            print(f"[{args.corpus}] {name}: already done, skipping", flush=True)
            continue
        print(f"[{args.corpus}] {name}", flush=True)
        if name == "FC-NTM":
            state = Path(args.fcntm_state)
            if not state.exists():
                print("  no saved FC-NTM state; skipping", flush=True)
                continue
            try:
                adapter = FCNTMAdapter(state, n_vocab, args.k,
                                       seed=args.seed)
            except KeyError as exc:
                # never fall back to another seed's model: that would silently
                # produce curves attributed to the wrong run
                print(f"  {exc}", flush=True)
                raise SystemExit(2)
            adapter.word2id = word2id
        elif name == "ProdLDA":
            if doc_bow_tensor is None:
                doc_bow_tensor = torch.from_numpy(
                    (train["reason_bow"] + train["item_bow"]).astype(np.float32))
            adapter = ProdLDAAdapter(train, args.k, args.seed, doc_bow_tensor)
            adapter.word2id = word2id
        elif name == "ETM":
            import embed_public as EP
            emb = EP.embed_words(vocab)
            emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
            if doc_bow_tensor is None:
                doc_bow_tensor = torch.from_numpy(
                    (train["reason_bow"] + train["item_bow"]).astype(np.float32))
            adapter = ETMAdapter(train, args.k, args.seed, emb, doc_bow_tensor)
            adapter.word2id = word2id
            del emb
        elif name == "LDA":
            adapter = LDAAdapter(train, args.k, args.seed)
            adapter.dictionary = test["dictionary"]
        elif name == "BTM":
            adapter = BTMAdapter(train, args.k, args.seed)
            adapter.dictionary = train["dictionary"]
        elif name == "NMF":
            adapter = NMFAdapter(train, args.k, args.seed)
            adapter.word2id = word2id
        else:
            raise ValueError(name)
        rows.extend(run_model(adapter, visible_A, field_B, hidden_A, rates,
                              word2id, name))
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_path, index=False,
                                  encoding="utf-8-sig")
        del adapter
        gc.collect()

    import pandas as pd
    frame = pd.DataFrame(rows)
    out_path = OUT_DIR / (args.out or
                          f"occlusion_robustness_{args.corpus}_seed{args.seed}.csv")
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    RI.write(out_path, ident)
    print(frame.round(4).to_string(index=False))
    print("saved", out_path)


if __name__ == "__main__":
    main()
