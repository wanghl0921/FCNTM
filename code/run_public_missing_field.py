# -*- coding: utf-8 -*-
"""Missing-field robustness on a public corpus, with a real occlusion test.

What the earlier revision did (and why it was wrong)
----------------------------------------------------
It scored ``C_V`` twice with two *different topic-word candidate sets and two
different reference corpora*: the "full" number used the whole vocabulary and
the full documents, the "occluded" number restricted beta to the summary-field
vocabulary and used the (very short) titles as the reference corpus.  Nothing
was occluded at inference time: beta is a fixed model parameter and does not
change per document, so the resulting "drop" measured the evaluation setup,
not information loss.

What this script does now
-------------------------
The background field is masked *at inference*, on the same documents, with the
same model, and the topic mixture is re-inferred from the remaining field:

    theta_full = q(z | x_cause, x_item)          (both fields observed)
    theta_occ  = q(z | x_item)                   (cause field set to missing)

Four quantities are reported per run:

  * delta_CV / delta_NPMI  - difference between the class-TF-IDF readout built
    from theta_full and the one built from theta_occ, both scored on the same
    reference corpus with the same 10-word read-out protocol;
  * delta_NLL_token        - increase in per-token negative log-likelihood of
    the held-out fields when the topics are inferred without the background;
  * top1_agreement / mean_js - how many documents keep their most probable
    topic, and how far the two posterior mixtures are from each other.

The field-view table (topic words restricted to one field's vocabulary) is
still written, but it is named for what it is: a field-restricted topic-word
read-out, not an occlusion experiment.

Usage
-----
python run_public_missing_field.py --dataset github --seeds 11,23,37 \
    --settings fielddrop_on,fielddrop_off --dump-state outputs/github_state.pt
python run_public_missing_field.py --dataset agnews --seeds 11 \
    --reuse-state outputs/agnews_state.pt
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import config as cfg
import result_identity as RI
import metrics as M
import models as NN
from config import TOPN
from run_fcntm_train import class_tfidf, top_words

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def field_vocab_ids(token_lists, vocab):
    present = set(w for ts in token_lists for w in ts)
    return [i for i, w in enumerate(vocab) if w in present]


def topics_in_field(beta, ids, vocab, topn=TOPN):
    sub = beta[:, ids]
    return M.topics_from_matrix(sub, [vocab[i] for i in ids], topn=topn)


def bow_from_tokens(token_lists, word2id, n_vocab):
    mat = np.zeros((len(token_lists), n_vocab), dtype=np.float32)
    for d, ts in enumerate(token_lists):
        for w in ts:
            j = word2id.get(w)
            if j is not None:
                mat[d, j] += 1.0
    return mat


def final_model_kwargs(field_dropout, fusion, fusion_norm, fusion_weight):
    """The configuration reported in the paper (see run_fcntm_train.py)."""
    kwargs = dict(
        consistency_weight=0.5,
        cross_reconstruction=True,
        cause_dropout=0.30,
        item_dropout=0.40,
        exclusivity_weight=0.0,
        field_dropout=float(field_dropout),
    )
    if fusion in ("fused", "gate", "poe"):
        kwargs["fused_recon"] = True
        kwargs["fused_recon_norm"] = fusion_norm
        kwargs["fused_recon_weight"] = float(fusion_weight)
    if fusion in ("gate", "poe"):
        kwargs["fusion_mode"] = fusion
    return kwargs


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def complete_partitions(occ_rows, fv_rows, dataset, keys) -> set:
    """(setting, seed) partitions whose stored rows are actually usable.

    ``keys`` is an iterable of ``(setting, seed)``.  A partition counts as
    complete only when its whole-field row exists with finite metrics *and*
    both field-view rows exist with a non-empty topic support.  An identity
    match on its own says nothing about the rows: the previous version skipped
    header-only files and NaN rows.
    """
    occ = {}
    for row in occ_rows:
        try:
            occ[(row.get("dataset"), row.get("setting"),
                 int(row.get("seed", -1)))] = row
        except (TypeError, ValueError):
            continue
    fv = {}
    for row in fv_rows:
        try:
            fv[(row.get("dataset"), row.get("setting"),
                int(row.get("seed", -1)), row.get("view"))] = row
        except (TypeError, ValueError):
            continue
    out = set()
    for setting, seed in keys:
        key = (dataset, setting, int(seed))
        row = occ.get(key)
        if row is None or not all(_finite(row.get(c)) for c in (
                "full_CV", "occ_CV", "delta_CV", "delta_NLL_token",
                "top1_agreement", "mean_js")):
            continue
        complete = True
        for view in ("background", "summary"):
            field_row = fv.get(key + (view,))
            if field_row is None or not (_finite(field_row.get("CV"))
                                         and _finite(field_row.get("NPMI"))):
                complete = False
                break
            try:
                if int(float(field_row.get("n_topics", 0))) < 1:
                    complete = False
                    break
            except (TypeError, ValueError):
                complete = False
                break
        if complete:
            out.add((setting, int(seed)))
    return out


def train_model(cause_bow, item_bow, has_cause, n_docs, n_vocab, K, seed,
                field_dropout, lr, fusion, fusion_norm, fusion_weight,
                max_epochs=cfg.PAPER_EPOCHS, checkpoint_path=None,
                checkpoint_meta=None, resume_path=None, resume_every=50,
                allow_resume=True):
    """Train FC-NTM on tensors that the caller already holds in memory.

    The encoder inputs are row-normalised per mini-batch instead of once for
    the whole corpus, which avoids two extra (docs x vocab) buffers.
    """
    kwargs = final_model_kwargs(field_dropout, fusion, fusion_norm,
                               fusion_weight)
    NN.set_torch_seed(seed)
    model = NN.FCNTM(n_vocab, K, **kwargs)

    def loss_fn(idx, kl_weight):
        cause_batch = cause_bow[idx]
        item_batch = item_bow[idx]
        return model.train_step(
            NN._normalize_rows(cause_batch), NN._normalize_rows(item_batch),
            cause_batch, item_batch, has_cause[idx], kl_weight)

    extra = {"recover_nan": True, "num_epochs": int(max_epochs)}
    if checkpoint_path:
        extra["checkpoint_path"] = checkpoint_path
        extra["checkpoint_meta"] = checkpoint_meta or {}
    if resume_path:
        extra["resume_path"] = resume_path
        extra["resume_every"] = int(resume_every)
        extra["allow_resume"] = bool(allow_resume)
        def _logger(ep, loss, klw):
            if ep % 25 == 0 or ep <= 5:
                print(f"    [train] epoch {ep} loss {loss:.3f} "
                      f"klw {klw:.4f}", flush=True)

        extra["logger"] = _logger
    if lr:
        extra["lr"] = float(lr)
    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, n_docs, seed,
        val_fraction=0.2, kl_max=0.05, **extra)
    model.eval()
    return model, info, kwargs


def infer(model, cause_bow, item_bow, has_cause, batch: int = 1024):
    """Batch-wise inference; normalisation happens per batch to keep memory low."""
    outs = []
    with torch.no_grad():
        for start in range(0, cause_bow.shape[0], batch):
            sl = slice(start, start + batch)
            c = cause_bow[sl]
            i = item_bow[sl]
            fused, _ = model.encode_doc_theta(
                NN._normalize_rows(c), NN._normalize_rows(i), has_cause[sl])
            outs.append(fused)
    return torch.cat(outs, dim=0).numpy()


def token_nll(model, thetas, counts, present):
    """Mean per-token NLL of the observed fields under the given mixtures."""
    with torch.no_grad():
        logits = model.decode(thetas)
        log_p = torch.log_softmax(logits, dim=-1).clamp(min=-1e4, max=0.0)
        nll = -(counts * log_p).sum(dim=-1)
        tokens = counts.sum(dim=-1).clamp_min(1.0)
        per_doc = nll / tokens
        per_doc = per_doc[present]
    return float(per_doc.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["agnews", "github", "civil"],
                    default="github")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--settings", default="fielddrop_on,fielddrop_off")
    ap.add_argument("--fusion", choices=["none", "fused", "gate", "poe"],
                    default="gate")
    ap.add_argument("--fusion-norm", choices=["joint", "balanced"],
                    default="balanced")
    ap.add_argument("--fusion-weight", type=float, default=0.5)
    ap.add_argument("--max-epochs", type=int, default=cfg.PAPER_EPOCHS)
    ap.add_argument("--resume-every", type=int, default=50,
                    help="write a full resumable training state every N epochs")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing resumable state and train afresh")
    ap.add_argument("--reuse-state", default="",
                    help="load trained models from this .pt instead of "
                         "training (keys: '<setting>:<seed>')")
    ap.add_argument("--dump-state", default="",
                    help="write every trained model state to this .pt so the "
                         "experiment can be re-scored without retraining")
    ap.add_argument("--out", default="outputs/public_missing_field_occlusion.csv")
    ap.add_argument("--field-view-out",
                    default="outputs/public_missing_field_fieldview.csv")
    args = ap.parse_args()

    if args.dataset == "agnews":
        import public_data as PD
        data = PD.load_agnews(1500, 1900, need_doc_bow=False)
    elif args.dataset == "civil":
        import public_data_civil as PC
        data = PC.load_civil(6000, 2000, need_doc_bow=False)
    else:
        import public_data_github as PGH
        data = PGH.load_github(1500, 500, need_doc_bow=False)
    train, test = data["train"], data["test"]
    del data
    # the (6000 x V) document-level count matrix is not used here; dropping it
    # frees ~120 MB, which matters on a memory-tight CPU box
    train["doc_bow"] = None
    test["doc_bow"] = None
    gc.collect()

    vocab = train["vocab"]
    word2id = train["dictionary"].token2id
    n_vocab = train["n_vocab"]
    cause_ids = field_vocab_ids(train["reason_tokens"], vocab)
    item_ids = field_vocab_ids(train["item_tokens"], vocab)
    print(f"[{args.dataset}] field vocab cause={len(cause_ids)} "
          f"item={len(item_ids)}", flush=True)

    # ---- test tensors (built with the training dictionary) ----------------
    # the tensors share memory with the numpy arrays (no second copy); the
    # numpy handles are re-bound below and freed once the tensors exist
    train_cause = torch.from_numpy(train["reason_bow"]).float()
    train_item = torch.from_numpy(train["item_bow"]).float()
    train["reason_bow"] = None
    train["item_bow"] = None
    gc.collect()
    train_has_cause = torch.tensor(
        [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
    train_none = torch.zeros(train_item.shape[0], dtype=torch.bool)
    test_cause_np = bow_from_tokens(test["reason_tokens"], word2id, n_vocab)
    test_item_np = bow_from_tokens(test["item_tokens"], word2id, n_vocab)
    test_cause = torch.from_numpy(test_cause_np).float()
    test_item = torch.from_numpy(test_item_np).float()
    test_cause_np = None
    test_item_np = None
    gc.collect()
    test_has_cause = torch.tensor(
        [len(t) > 0 for t in test["reason_tokens"]], dtype=torch.bool)
    test_none = torch.zeros(test_item.shape[0], dtype=torch.bool)
    print(f"[{args.dataset}] test docs={len(test['doc_tokens'])} "
          f"with background field={int(test_has_cause.sum())}", flush=True)

    states = {}
    resume_path = None
    if args.dump_state:
        resume_path = str(cfg.ROOT / args.dump_state) \
            if not Path(args.dump_state).is_absolute() else args.dump_state
        resume_path = resume_path + ".resume"
    if args.reuse_state:
        p = Path(args.reuse_state)
        if p.exists() and p.stat().st_size:
            states = torch.load(p, map_location="cpu", weights_only=False)
            print("loaded states for", sorted(states), flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    settings = [s for s in args.settings.split(",") if s]

    occ_fields = ["dataset", "setting", "seed", "K", "readout",
                  "full_CV", "full_NPMI", "full_UMass",
                  "occ_CV", "occ_NPMI", "occ_UMass",
                  "delta_CV", "delta_NPMI",
                  "full_NLL_token", "occ_NLL_token", "delta_NLL_token",
                  "delta_NLL_body", "top1_agreement", "mean_js",
                  "config_hash", "config_json", "nll_norm",
                  "occ_readout_counts", "epochs_budget", "best_epoch",
                  "seconds"]
    fv_fields = ["dataset", "setting", "seed", "K", "view",
                 "reference_corpus", "n_topics", "CV", "CV_n", "NPMI",
                 "NPMI_n", "UMass", "UMass_n", "seconds"]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = cfg.ROOT / out_path
    fv_path = Path(args.field_view_out)
    if not fv_path.is_absolute():
        fv_path = cfg.ROOT / fv_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Identity is tracked *per (setting, seed) partition*, not for the file as a
    # whole: the same table holds several corpora and settings, so a whole-file
    # identity made a second corpus look like a configuration change and moved
    # the first corpus' rows aside.  Each partition gets its own hash of
    # (script, dataset, setting, seed, hyper-parameters, data, code).
    base_ident = {
        "script": "run_public_missing_field.main",
        "dataset": args.dataset,
        "k": int(args.k),
        "fusion": args.fusion,
        "fusion_norm": args.fusion_norm,
        "fusion_weight": float(args.fusion_weight),
        "max_epochs": int(args.max_epochs),
        "lr": float(args.lr),
        "no_resume": bool(args.no_resume),
        "code": RI.code_version(),
        "data": RI.text_digest([test.get("doc_tokens"),
                                test.get("reason_tokens"),
                                test.get("item_tokens")]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens"),
                                  test.get("reason_tokens"),
                                  test.get("item_tokens")]),
    }
    # the key carries the dataset: GitHub and AG News can use the same
    # setting/seed and must not share one partition identity
    def partition_key(setting, seed) -> str:
        return f"{args.dataset}|{setting}|{int(seed)}"

    wanted = {}
    for setting in settings:
        for seed in seeds:
            wanted[partition_key(setting, int(seed))] = RI.identity(
                dict(base_ident, setting=setting, seed=int(seed)))["id"]
    stored_partitions = {}
    sidecar = RI.sidecar_path(out_path)
    if sidecar.exists():
        try:
            stored_partitions = json.loads(
                sidecar.read_text(encoding="utf-8")).get("partitions", {})
        except (ValueError, OSError):
            stored_partitions = {}

    def partition_ok(setting, seed) -> bool:
        # ``wanted`` only holds the keys requested *now*: an existing row for a
        # seed that is not part of this run must not raise a KeyError
        key = partition_key(setting, seed)
        return key in wanted and stored_partitions.get(key) == wanted[key]

    unmatched = [k for k, v in wanted.items()
                 if stored_partitions.get(k) != v]
    if stored_partitions and unmatched:
        print(f"{len(unmatched)} of {len(wanted)} partitions of "
              f"{out_path.name} are not pinned to this configuration; "
              f"those rows will be recomputed", flush=True)
    elif not stored_partitions and out_path.exists() and out_path.stat().st_size:
        print(f"{out_path.name} carries no partition identities yet; every "
              f"listed key will be recomputed", flush=True)

    # A partition may only be skipped when its identity matches *and* both
    # tables really hold usable rows for it: an identity alone said nothing
    # about the rows (header-only files and NaN rows were being skipped).
    def read_rows(path):
        if not path.exists() or not path.stat().st_size:
            return []
        try:
            with open(path, encoding="utf-8-sig", newline="") as fh:
                return list(csv.DictReader(fh))
        except (OSError, ValueError):
            return []

    complete = complete_partitions(
        read_rows(out_path), read_rows(fv_path), args.dataset,
        [(setting, int(seed)) for setting in settings for seed in seeds])
    reusable = {(setting, seed) for (setting, seed) in complete
                if partition_ok(setting, seed)}
    print(f"  {len(reusable)} of {len(settings) * len(seeds)} requested "
          f"partitions are complete and pinned (the rest will be computed)",
          flush=True)
    # The table always holds several partitions (this corpus/settings/seeds plus
    # whatever other corpora were run before), so it is rewritten as a merge:
    # rows whose partition identity matches are carried over, rows belonging to
    # a partition that is being recomputed are dropped, and the new rows are
    # appended as they are produced (flushed, so a crash keeps finished keys).
    recompute = {(args.dataset, setting, int(seed))
                 for setting in settings for seed in seeds
                 if not partition_ok(setting, int(seed))}

    def carry_over(path, fields):
        if not path.exists() or not path.stat().st_size:
            return []
        try:
            with open(path, encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh), [])
                fh.seek(0)
                rows = list(csv.DictReader(fh))
        except (StopIteration, OSError, ValueError):
            return []
        if header and header != list(fields):
            print(f"  {path.name} has a different schema "
                  f"({len(header)} vs {len(fields)} columns); its rows cannot "
                  f"be carried over and were left in the backup", flush=True)
            return []
        keep = []
        for row in rows:
            try:
                key = (row.get("dataset"), row.get("setting"),
                       int(row.get("seed", -1)))
            except (TypeError, ValueError):
                continue
            if key not in recompute:
                keep.append(row)
        return keep

    carried_occ = carry_over(out_path, occ_fields)
    carried_fv = carry_over(fv_path, fv_fields)
    if carried_occ:
        print(f"  carrying over {len(carried_occ)} rows of other partitions",
              flush=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh_occ, \
            open(fv_path, "w", encoding="utf-8-sig", newline="") as fh_fv:
        occ_writer = csv.DictWriter(fh_occ, fieldnames=occ_fields,
                                    extrasaction="ignore")
        fv_writer = csv.DictWriter(fh_fv, fieldnames=fv_fields,
                                   extrasaction="ignore")
        occ_writer.writeheader()
        fv_writer.writeheader()
        for row in carried_occ:
            occ_writer.writerow(row)
        for row in carried_fv:
            fv_writer.writerow(row)
        for setting in settings:
            field_dropout = 0.5 if setting == "fielddrop_on" else 0.0
            for seed in seeds:
                if (setting, int(seed)) in reusable:
                    print(f"  skip {args.dataset}/{setting}/seed={seed}: "
                          f"already in the table (identity "
                          f"{wanted[partition_key(setting, int(seed))]})",
                          flush=True)
                    continue
                key = f"{setting}:{seed}"
                t0 = time.time()
                run_kwargs = final_model_kwargs(
                    field_dropout, args.fusion, args.fusion_norm,
                    args.fusion_weight)
                if key in states:
                    # use the configuration the weights were trained with, not
                    # the current command line (fusion_mode / fused_recon and
                    # the other non-parameter options would silently differ)
                    stored = states[key]
                    run_kwargs = stored.get("kwargs") or run_kwargs
                    NN.set_torch_seed(seed)
                    model = NN.FCNTM(n_vocab, args.k, **run_kwargs)
                    model.load_state_dict(stored["model_state"])
                    model.eval()
                    info = {"best_epoch": stored.get("best_epoch", -1)}
                else:
                    model, info, run_kwargs = train_model(
                        train_cause, train_item, train_has_cause,
                        train["n_docs"], n_vocab, args.k, seed,
                        field_dropout, args.lr, args.fusion, args.fusion_norm,
                        args.fusion_weight, args.max_epochs,
                        checkpoint_path=resume_path,
                        checkpoint_meta={"key": key, "kwargs": run_kwargs},
                        resume_path=(f"{resume_path}.{key.replace(':', '_')}"
                                     if resume_path else None),
                        resume_every=args.resume_every,
                        allow_resume=not args.no_resume)
                    if args.dump_state:
                        states[key] = {
                            "model_state": {k: v.detach().cpu().clone()
                                            for k, v in
                                            model.state_dict().items()},
                            "kwargs": run_kwargs,
                            "best_epoch": int(info["best_epoch"]),
                        }
                        sp = Path(args.dump_state)
                        if not sp.is_absolute():
                            sp = cfg.ROOT / sp
                        sp.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(states, sp)

                # ---- full vs occluded inference, same documents ----------
                # read-out: class-TF-IDF over the *training* posterior, exactly
                # as in run_fcntm_train.py
                theta_full = infer(model, train_cause, train_item,
                                   train_has_cause)
                theta_occ = infer(model, train_cause, train_item, train_none)
                beta = np.asarray(model.get_beta(), dtype=float)
                # aggregate without materialising a (docs x vocab) copy of the
                # combined field counts
                cause_np = train_cause.numpy()
                item_np = train_item.numpy()
                counts_full = theta_full.T @ cause_np + theta_full.T @ item_np
                counts_occ = theta_occ.T @ cause_np + theta_occ.T @ item_np
                scores_full = class_tfidf(counts_full)
                scores_occ = class_tfidf(counts_occ)
                topics_full = top_words(scores_full, vocab)
                topics_occ = top_words(scores_occ, vocab)

                # document-level statistics use the held-out test documents
                theta_full_test = infer(model, test_cause, test_item,
                                        test_has_cause)
                theta_occ_test = infer(model, test_cause, test_item, test_none)

                row = {"dataset": args.dataset, "setting": setting,
                       "seed": seed, "K": int(beta.shape[0]),
                       "readout": "cTFIDF",
                       "config_hash": hashlib.sha1(
                           json.dumps(run_kwargs, sort_keys=True).encode()
                       ).hexdigest()[:12],
                       "config_json": json.dumps(run_kwargs, sort_keys=True),
                       "nll_norm": "per_document_token_average",
                       "occ_readout_counts": "full_training_counts",
                       "epochs_budget": int(args.max_epochs),
                       "best_epoch": int(info.get("best_epoch", -1))}
                for readout, topics in (("full", topics_full),
                                        ("occ", topics_occ)):
                    ev = M.coherence_scores(
                        topics, test["doc_tokens"], test["corpus"],
                        test["dictionary"], topn=TOPN)
                    for metric in ("CV", "NPMI", "UMass"):
                        row[f"{readout}_{metric}"] = round(
                            float(ev[metric]), 6)
                row["delta_CV"] = round(row["full_CV"] - row["occ_CV"], 6)
                row["delta_NPMI"] = round(row["full_NPMI"] - row["occ_NPMI"], 6)

                with torch.no_grad():
                    th_full = torch.from_numpy(theta_full_test).float()
                    th_occ = torch.from_numpy(theta_occ_test).float()
                    counts_all = test_cause + test_item
                    row["full_NLL_token"] = round(
                        token_nll(model, th_full, counts_all, test_has_cause),
                        6)
                    row["occ_NLL_token"] = round(
                        token_nll(model, th_occ, counts_all, test_has_cause),
                        6)
                    row["delta_NLL_token"] = round(
                        row["occ_NLL_token"] - row["full_NLL_token"], 6)
                    body = test_cause
                    row["delta_NLL_body"] = round(
                        token_nll(model, th_occ, body, test_has_cause)
                        - token_nll(model, th_full, body, test_has_cause), 6)
                    argmax_full = th_full.argmax(dim=-1)
                    argmax_occ = th_occ.argmax(dim=-1)
                    row["top1_agreement"] = round(float(
                        (argmax_full[test_has_cause]
                         == argmax_occ[test_has_cause]).float().mean()), 6)
                    js = NN.sym_js(th_full[test_has_cause],
                                   th_occ[test_has_cause])
                    row["mean_js"] = round(float(js.mean()), 6)
                row["seconds"] = round(time.time() - t0, 2)
                occ_writer.writerow(row)
                fh_occ.flush()

                # field-restricted read-out, scored against the *field's own*
                # documents (the field-view definition of the paper,
                # tab:fieldview):
                # topic words are truncated to the field vocabulary and the
                # reference corpus is that field's text, not the full document
                for view, ids, field_tokens in (
                        ("background", cause_ids,
                         [t for t in test["reason_tokens"] if t]),
                        ("summary", item_ids,
                         [t for t in test["item_tokens"] if t])):
                    field_bows = [test["dictionary"].doc2bow(t)
                                  for t in field_tokens]
                    topics = topics_in_field(beta, ids, vocab)
                    ev = M.coherence_scores(
                        topics, field_tokens, field_bows,
                        test["dictionary"], topn=TOPN)
                    fv_writer.writerow({
                        "dataset": args.dataset, "setting": setting,
                        "seed": seed, "K": int(beta.shape[0]), "view": view,
                        "reference_corpus": view,
                        # the field's own test text cannot define every topic:
                        # words that the field never uses give an all-zero
                        # context vector, so gensim's c_v is undefined for that
                        # topic and it is dropped from the mean.  Record the
                        # support so a thin cell cannot be read as 30 topics.
                        "n_topics": int(ev["n_topics"]),
                        "CV": round(float(ev["CV"]), 6),
                        "CV_n": int(ev["CV_n"]),
                        "NPMI": round(float(ev["NPMI"]), 6),
                        "NPMI_n": int(ev["NPMI_n"]),
                        "UMass": round(float(ev["UMass"]), 6),
                        "UMass_n": int(ev["UMass_n"]),
                        "seconds": row["seconds"],
                    })
                fh_fv.flush()
                # this partition is now pinned to the configuration that
                # produced it (written immediately, so a crash keeps it)
                # commit the identity only now: both rows (occlusion and both
                # field views) have been written and flushed
                stored_partitions[partition_key(setting, int(seed))] = wanted[
                    partition_key(setting, int(seed))]
                RI.write(out_path, {"kind": "partitioned",
                                    "config": RI.identity(base_ident)["id"],
                                    "partitions": stored_partitions})
                print("  ", setting, "seed", seed,
                      {k: row[k] for k in
                       ("full_CV", "occ_CV", "delta_CV", "delta_NLL_token",
                        "top1_agreement")}, flush=True)
                del model
    RI.write(out_path, {"kind": "partitioned",
                        "config": RI.identity(base_ident)["id"],
                        "partitions": stored_partitions})
    print("saved", out_path, "|", fv_path)


if __name__ == "__main__":
    main()
