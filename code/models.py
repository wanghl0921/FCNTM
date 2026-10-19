# -*- coding: utf-8 -*-
"""Self-contained PyTorch implementations used in the FC-NTM paper.

Models implemented from the published descriptions:
  - ProdLDA (Srivastava & Sutton, 2017): linear decoder VAE topic model.
  - ENTM: embedding-decoder neural topic model (ETM-style decoder, Dieng et al. 2020).
  - CTMLite: contextual-embedding topic model (CombinedTM-style encoder, Bianchi et al. 2021).
  - FC-NTM: proposed field-collaborative neural topic model.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    BATCH_SIZE,
    DROPOUT,
    EMB_DECODER_TEMPERATURE,
    HIDDEN,
    KL_WARMUP_EPOCHS,
    KL_MAX_WEIGHT,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    CAUSE_RECON_WEIGHT,
    TOPIC_ORTH_WEIGHT,
)


def set_torch_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    length = x.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return x / length


def _softplus() -> nn.Module:
    return nn.Softplus()


class MLPEncoder(nn.Module):
    """BoW / concatenated-feature encoder producing Gaussian posterior params."""

    def __init__(
        self,
        input_dim: int,
        num_topics: int,
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            _softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            _softplus(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(hidden, num_topics)
        self.logvar_layer = nn.Linear(hidden, num_topics)
        self.mu_bn = nn.BatchNorm1d(num_topics, affine=False) if use_bn else nn.Identity()
        self.logvar_bn = nn.BatchNorm1d(num_topics, affine=False) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_bn(self.mu_layer(h))
        logvar = self.logvar_bn(self.logvar_layer(h))
        logvar = logvar.clamp(max=2.0)
        return mu, logvar


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: Optional[torch.Tensor] = None,
    prior_logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if prior_mu is None:
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    else:
        var = logvar.exp()
        prior_var = prior_logvar.exp()
        kl = 0.5 * torch.sum(
            (var + (mu - prior_mu).pow(2)) / prior_var - 1.0 + prior_logvar - logvar,
            dim=-1,
        )
    return kl.clamp_min(0.0)


def sym_js(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence between probability rows."""
    m = 0.5 * (a + b)
    kl_am = (a * (a / m.clamp_min(eps)).log().clamp_min(-100.0)).sum(dim=-1)
    kl_bm = (b * (b / m.clamp_min(eps)).log().clamp_min(-100.0)).sum(dim=-1)
    return 0.5 * (kl_am + kl_bm)


def reconstruction_loss_from_counts(
    word_logits: torch.Tensor,
    counts: torch.Tensor,
    reduce: bool = True,
    per_token: bool = False,
) -> torch.Tensor:
    """Per-document negative log-likelihood (counts are raw)."""
    # clamp keeps -inf log-probs from turning zero-count entries into NaN.
    log_p = torch.log_softmax(word_logits, dim=-1).clamp(min=-1e4, max=0.0)
    return _nll_from_log_prob(log_p, counts, reduce=reduce, per_token=per_token)


def _nll_from_log_prob(
    log_p: torch.Tensor,
    counts: torch.Tensor,
    reduce: bool = True,
    per_token: bool = False,
) -> torch.Tensor:
    """Negative log-likelihood for an already normalised ``log p(w | d)``."""
    log_p = log_p.clamp(min=-1e4, max=0.0)
    loss = -(counts * log_p).sum(dim=-1)
    if per_token:
        token_count = counts.sum(dim=-1).clamp_min(1.0)
        loss = loss / token_count
    return loss.mean() if reduce else loss


class LinearDecoder(nn.Module):
    """ProdLDA-style linear topic decoder."""

    def __init__(self, num_topics: int, vocab_size: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.empty(num_topics, vocab_size))
        nn.init.xavier_uniform_(self.beta)
        self.beta_bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop_theta = nn.Dropout(dropout)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(self.drop_theta(theta), self.beta)
        if self.training and logits.shape[0] < 2:
            # BatchNorm1d cannot estimate batch statistics from a single
            # sample ("Expected more than 1 value per channel when training").
            # This happens whenever a field-level mask leaves exactly one
            # active record (small batches, high natural missing rate), so the
            # call falls back to the running statistics for that step.
            was_training = self.beta_bn.training
            self.beta_bn.eval()
            try:
                return self.beta_bn(logits)
            finally:
                self.beta_bn.train(was_training)
        return self.beta_bn(logits)

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.beta.detach().cpu().numpy().copy()

    def beta_tensor(self) -> torch.Tensor:
        return self.beta

    def beta_prob_tensor(self, temperature: float = 1.0) -> torch.Tensor:
        """Row-normalized topic-word distribution used by the biterm term."""
        return torch.softmax(self.beta / temperature, dim=-1)


def topic_orthogonality_penalty(rows: torch.Tensor) -> torch.Tensor:
    """Mean |cosine| among topic parameter rows (pushes topics apart)."""
    if rows.shape[0] <= 1:
        return torch.zeros((), device=rows.device)
    norms = rows.norm(dim=-1).clamp_min(1e-8)
    normalized = rows / norms.unsqueeze(-1)
    gram = torch.abs(torch.mm(normalized, normalized.t()))
    n = rows.shape[0]
    off_diag = gram - torch.eye(n, device=rows.device)
    return off_diag.sum() / (n * (n - 1))


class EmbeddingDecoder(nn.Module):
    """Embedding decoder with two distinct mixture conventions.

    * ``topic_word_prob()`` implements the ETM decoder of Dieng et al. (2020),
      eq. (5)-(6): each topic is normalised into a word distribution first,

          beta_k = softmax_v(alpha_k E^T),   p(w | d) = sum_k theta_dk beta_kw.

      ``log_prob()`` returns log p(w | d) under that convention and must be
      used with ``_nll_from_log_prob`` (the two-step softmax cannot be written
      as a single softmax over logits).

    * ``forward()/_logits()`` keep the older logit-space mixture
      ``softmax_v(theta_d alpha E^T)`` that an earlier revision used for the
      "ETM" baseline.  That is a ProdLDA-style decoder, *not* ETM; it is kept
      only so the historical FC-NTM embedding variant still runs, and it is no
      longer used by the ETM baseline.
    """

    def __init__(
        self,
        num_topics: int,
        word_embeddings: np.ndarray,
        trainable_embeddings: bool = False,
        dropout: float = DROPOUT,
        temperature: float = EMB_DECODER_TEMPERATURE,
        use_word_bias: bool = False,
    ) -> None:
        super().__init__()
        vocab_size, dim = word_embeddings.shape
        self.temperature = temperature
        # ``torch.from_numpy`` on a float32 array shares memory with the caller
        # and ``.float()`` keeps sharing it.  When the embeddings are trainable
        # (ETM) the optimizer then writes into the caller's array, so every
        # later seed inherits the previous seed's updates even though the torch
        # seed is reset.  Clone so each model owns an immutable input snapshot.
        emb = torch.from_numpy(
            np.ascontiguousarray(word_embeddings, dtype=np.float32)).clone()
        self.use_word_bias = use_word_bias
        self.register_parameter(
            "word_embed",
            nn.Parameter(emb, requires_grad=trainable_embeddings),
        )
        self.alpha = nn.Parameter(torch.randn(num_topics, dim) * 0.08)
        self.word_bias = (
            nn.Parameter(torch.zeros(vocab_size)) if use_word_bias else None
        )
        self.drop_theta = nn.Dropout(dropout)

    def _logits(self, theta: torch.Tensor) -> torch.Tensor:
        # doc context in embedding space, then project to vocabulary
        context = torch.matmul(self.drop_theta(theta), self.alpha)  # (B, D)
        logits = torch.matmul(context, self.word_embed.t()) / self.temperature
        if self.use_word_bias:
            logits = logits + self.word_bias
        return logits

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self._logits(theta)

    def topic_word_prob(self) -> torch.Tensor:
        """beta (K, V) = softmax_v(alpha_k E^T) -- the ETM topic-word matrix."""
        logits = torch.matmul(self.alpha, self.word_embed.t()) / self.temperature
        if self.use_word_bias:
            logits = logits + self.word_bias
        return torch.softmax(logits, dim=-1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """log p(w | d) = log sum_k theta_dk beta_kw (standard ETM mixture)."""
        beta = self.topic_word_prob()
        mixture = torch.matmul(theta, beta).clamp_min(1e-12)
        return torch.log(mixture)

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.topic_word_prob().detach().cpu().numpy().copy()

    def topic_parameter(self) -> torch.Tensor:
        return self.alpha

    def beta_prob_tensor(self, temperature: float = 1.0) -> torch.Tensor:
        logits = torch.matmul(self.alpha, self.word_embed.t()) / self.temperature
        if self.use_word_bias:
            logits = logits + self.word_bias
        return torch.softmax(logits / temperature, dim=-1)


def train_topic_vae(
    model: nn.Module,
    loss_fn,
    n_docs: int,
    seed: int,
    logger=None,
) -> Dict[str, object]:
    """Generic training loop with KL warmup, early stopping and loss history."""
    set_torch_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    best_loss = float("inf")
    best_state: Dict[str, object] = {}
    patience = 0
    loss_history: List[float] = []
    epochs_run = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        permutation = torch.randperm(n_docs)
        kl_weight = min(1.0, epoch / KL_WARMUP_EPOCHS) * KL_MAX_WEIGHT
        for start in range(0, n_docs, BATCH_SIZE):
            idx = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            loss, _ = loss_fn(idx, kl_weight=kl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("loss is not finite")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("no batches")
        else:
            mean_loss = epoch_loss / n_batches
        loss_history.append(mean_loss)
        epochs_run = epoch
        if logger is not None:
            logger(epoch, mean_loss, kl_weight)

        if epoch <= KL_WARMUP_EPOCHS:
            # during KL warm-up the total loss is dominated by the annealing term;
            # only start selecting the best state after the target prior weight is reached
            continue

        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            best_state = {
                "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "loss": mean_loss,
            }
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if not best_state:
        best_state = {
            "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "epoch": epochs_run,
            "loss": loss_history[-1],
        }
    model.load_state_dict(best_state["model_state"])
    return {
        "model": model,
        "best_epoch": best_state["epoch"],
        "epochs_run": epochs_run,
        "train_loss": best_state["loss"],
        "loss_history": loss_history,
        "early_stopped": epochs_run < NUM_EPOCHS,
    }


def _rollback(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Dict[str, object],
) -> None:
    """Recover from a non-finite step without leaving stale Adam moments.

    Restoring only ``model.state_dict()`` (as an earlier revision did) keeps
    the Adam first/second moments of the diverged step, so the next step can
    immediately diverge again.  The moments are cleared together with the
    weights and the learning rate is halved.
    """
    model.load_state_dict(state)
    optimizer.state.clear()
    for grp in optimizer.param_groups:
        grp["lr"] = max(grp["lr"] * 0.5, 1e-6)


def train_topic_vae_valid(
    model: nn.Module,
    train_loss_fn,
    val_loss_fn,
    n_docs: int,
    seed: int,
    val_fraction: float = 0.2,
    kl_max: float = KL_MAX_WEIGHT,
    warmup_epochs: int = KL_WARMUP_EPOCHS,
    patience: int = PATIENCE,
    num_epochs: int = NUM_EPOCHS,
    logger=None,
    recover_nan: bool = False,
    lr: float = LEARNING_RATE,
    checkpoint_path: Optional[str] = None,
    checkpoint_meta: Optional[Dict[str, object]] = None,
    resume_path: Optional[str] = None,
    resume_every: int = 50,
    allow_resume: bool = True,
    resume_identity: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Train a neural topic VAE selecting the best state by held-out loss.

    Unlike ``train_topic_vae`` (which selects by training loss), this routine
    splits the corpus into train/validation, evaluates the validation loss at
    the end of every epoch after KL warm-up, and keeps the state with the
    lowest validation loss.  This avoids rewarding memorization of the tiny
    training corpus.

    ``checkpoint_path`` writes the best state to disk whenever it improves
    (throttled to every 5 epochs), so a machine that dies mid-run does not
    lose the whole training.  The file is a ``{key: {model_state, kwargs,
    best_epoch}}`` mapping that ``run_public_missing_field.py --reuse-state``
    can read directly.

    ``resume_path`` writes the *full* training state (weights, optimizer
    moments, epoch counter, histories and RNG states) every ``resume_every``
    epochs and, if the file already exists, continues from it instead of
    starting over.  This makes a long run interruptible: each invocation
    advances the run by as many epochs as the machine allows.

    A stored state is only reused when ``resume_identity`` (a
    ``result_identity.identity`` payload describing the data, fields, model
    configuration, learning rate and code version) matches the identity
    recorded inside the file.  A mismatch, a corrupt file or a legacy state
    without an identity is moved aside (``*.stale<timestamp>``) and training
    starts fresh, so a changed objective can never silently continue an old
    run.
    """
    set_torch_seed(seed)
    rng = np.random.RandomState(seed + 12345)
    perm = rng.permutation(n_docs)
    n_val = max(1, int(round(n_docs * val_fraction)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999)
    )
    best_val = float("inf")
    best_state: Dict[str, object] = {}
    patience_counter = 0
    loss_history: List[float] = []
    loss_by_epoch: Dict[int, float] = {}
    val_history: List[float] = []
    epochs_run = 0
    start_epoch = 1

    if resume_path and allow_resume and Path(resume_path).exists():
        ck = None
        try:
            ck = torch.load(resume_path, map_location="cpu",
                            weights_only=False)
        except Exception as exc:  # noqa: BLE001 - a corrupt state must not
            print(f"    [resume] cannot read {resume_path}: {exc}",
                  flush=True)  # crash, it is archived below
        want_id = (resume_identity or {}).get("id")
        stored_id = (ck or {}).get("identity_id")
        if ck is not None and not want_id:
            print("    [resume] no resume_identity was supplied; refusing "
                  "to reuse an unidentified state", flush=True)
            ck = None
        elif ck is not None and stored_id != want_id:
            print(f"    [resume] identity mismatch (stored {stored_id!r}, "
                  f"current {want_id!r}); the old state cannot be reused",
                  flush=True)
            ck = None
        elif ck is not None and int(ck.get("epoch", 0)) > int(num_epochs):
            # a smaller budget must not silently inherit a longer run
            # (2026-10-11 review, item 5)
            print(f"    [resume] requested budget ({num_epochs}) is smaller "
                  f"than the stored epoch ({ck.get('epoch')}); restarting "
                  f"from scratch", flush=True)
            ck = None
        if ck is None:
            archive = f"{resume_path}.stale{time.strftime('%Y%m%d%H%M%S')}"
            os.replace(resume_path, archive)
            print(f"    [resume] old state moved to {archive}; "
                  f"starting fresh", flush=True)
        else:
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            epochs_run = int(ck["epoch"])
            start_epoch = epochs_run + 1
            best_val = float(ck["best_val"])
            best_state = ck["best_state"]
            patience_counter = int(ck.get("patience_counter", 0))
            loss_history = list(ck.get("loss_history", []))
            loss_by_epoch = {int(k): float(v)
                             for k, v in ck.get("loss_by_epoch", {}).items()}
            val_history = list(ck.get("val_history", []))
            if ck.get("torch_rng") is not None:
                torch.set_rng_state(ck["torch_rng"])
            if ck.get("numpy_rng") is not None:
                np.random.set_state(ck["numpy_rng"])
            print(f"    [resume] continuing from epoch {epochs_run} "
                  f"(identity {stored_id})", flush=True)
        if ck is not None:
            # 2026-10-15 v3 review C2: a run that already finished (early
            # stop or budget) must return its saved best model instead of
            # silently training extra epochs.  Legacy files without the
            # completion flag are recognised through patience_counter.
            stop_reason = ck.get("stop_reason")
            legacy_finished = (
                not ck.get("completed")
                and int(ck.get("patience_counter", 0)) >= patience)
            if ck.get("completed") or legacy_finished:
                if stop_reason == "budget" and int(num_epochs) > epochs_run:
                    print("    [resume] the budget was extended; continuing",
                          flush=True)
                else:
                    model.load_state_dict(best_state["model_state"])
                    # 2026-10-15 v4 review R3: report the real stop reason;
                    # a budget-completed run is not an early stop
                    finished_early = stop_reason != "budget"
                    print(
                        "    [resume] the stored run already finished "
                        f"({stop_reason or 'legacy early-stopped state'}); "
                        "returning the saved best model without training",
                        flush=True)
                    return {
                        "model": model,
                        "best_epoch": best_state["epoch"],
                        "epochs_run": epochs_run,
                        "train_loss": float(
                            loss_by_epoch.get(
                                int(best_state["epoch"]),
                                loss_by_epoch[max(loss_by_epoch)]
                                if loss_by_epoch else float("nan"))),
                        "loss_history": loss_history,
                        "val_history": val_history,
                        "early_stopped": finished_early,
                        "nan_events": 0,
                        "num_epochs": int(num_epochs),
                        "patience": int(patience),
                    }

    nan_events = 0
    consecutive_diverged = 0
    last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def save_resume_state(epoch_index: int, completed: bool = False,
                          stop_reason: Optional[str] = None) -> None:
        torch.save({
            "model_state": {k: v.detach().clone()
                            for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch_index),
            "best_val": float(best_val),
            "best_state": best_state,
            "patience_counter": int(patience_counter),
            "loss_history": loss_history,
            "loss_by_epoch": {int(k): float(v)
                              for k, v in loss_by_epoch.items()},
            "val_history": val_history,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "identity_id": (resume_identity or {}).get("id"),
            "identity": resume_identity,
            # 2026-10-15 v3 review C2: record whether the run finished so a
            # later invocation can return the saved model instead of
            # training extra epochs on a completed state
            "completed": bool(completed),
            "stop_reason": stop_reason,
        }, resume_path)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        diverged = False
        batch_perm = torch.randperm(len(train_idx))
        kl_weight = min(1.0, epoch / warmup_epochs) * kl_max
        for start in range(0, len(train_idx), BATCH_SIZE):
            local = batch_perm[start : start + BATCH_SIZE]
            idx = train_idx[local]
            optimizer.zero_grad()
            loss, _ = train_loss_fn(idx, kl_weight=kl_weight)
            if not torch.isfinite(loss):
                if not recover_nan:
                    raise FloatingPointError("loss is not finite")
                nan_events += 1
                _rollback(model, optimizer, last_good)
                diverged = True
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(grad_norm):
                if not recover_nan:
                    raise FloatingPointError("gradient is not finite")
                nan_events += 1
                _rollback(model, optimizer, last_good)
                diverged = True
                break
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1
        if diverged:
            consecutive_diverged += 1
            if consecutive_diverged > 8:
                raise RuntimeError(
                    "training diverged; nan-recovery could not restore it")
            print(f"    [nan-recovery] epoch {epoch} diverged "
                  f"(attempt {consecutive_diverged}); rolled back to the last "
                  f"clean epoch, lr -> "
                  f"{optimizer.param_groups[0]['lr']:.2e}", flush=True)
            continue
        consecutive_diverged = 0
        if n_batches == 0:
            raise RuntimeError("no batches")
        mean_loss = epoch_loss / n_batches
        loss_history.append(mean_loss)
        loss_by_epoch[epoch] = mean_loss
        epochs_run = epoch
        if logger is not None:
            logger(epoch, mean_loss, kl_weight)
        epoch_snapshot = ({k: v.detach().clone()
                           for k, v in model.state_dict().items()}
                          if recover_nan else None)

        if epoch <= warmup_epochs:
            # do not select a state while the KL weight is still increasing
            if epoch_snapshot is not None:
                last_good = epoch_snapshot
            continue

        model.eval()
        with torch.no_grad():
            val_loss, _ = val_loss_fn(
                torch.from_numpy(np.asarray(val_idx, dtype=np.int64)), kl_weight=kl_max
            )
        model.train()
        val_loss_value = float(val_loss.detach().item())
        if not np.isfinite(val_loss_value) and recover_nan:
            _rollback(model, optimizer, last_good)
            print(f"    [nan-recovery] epoch {epoch} non-finite val loss; "
                  f"restored last clean epoch, lr -> "
                  f"{optimizer.param_groups[0]['lr']:.2e}", flush=True)
            continue
        if epoch_snapshot is not None:
            last_good = epoch_snapshot
        val_history.append(val_loss_value)
        if val_loss_value < best_val - 1e-4:
            best_val = val_loss_value
            best_state = {
                "model_state": {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                },
                "epoch": epoch,
                "loss": val_loss_value,
            }
            if checkpoint_path and (epoch % 5 == 0 or checkpoint_meta is None):
                key = (checkpoint_meta or {}).get("key", "model")
                payload = {}
                p = Path(checkpoint_path)
                if p.exists() and p.stat().st_size:
                    try:
                        payload = torch.load(p, map_location="cpu",
                                             weights_only=False)
                    except Exception:  # noqa: BLE001
                        payload = {}
                payload[key] = {
                    "model_state": best_state["model_state"],
                    "kwargs": (checkpoint_meta or {}).get("kwargs", {}),
                    "best_epoch": int(epoch),
                    "val_loss": float(val_loss_value),
                }
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, p)
            patience_counter = 0
        else:
            patience_counter += 1
        if resume_path and epoch % max(1, resume_every) == 0:
            save_resume_state(epoch)
        if patience_counter >= patience:
            break

    if resume_path:
        finished_reason = ("early_stopped" if patience_counter >= patience
                           else "budget")
        save_resume_state(epochs_run, completed=True,
                          stop_reason=finished_reason)

    if not best_state:
        best_state = {
            "model_state": {
                k: v.detach().clone() for k, v in model.state_dict().items()
            },
            "epoch": epochs_run,
            "loss": loss_history[-1],
        }
    model.load_state_dict(best_state["model_state"])
    return {
        "model": model,
        "best_epoch": best_state["epoch"],
        "epochs_run": epochs_run,
        # ``loss_history`` skips epochs that were rolled back, so it must be
        # indexed by epoch number rather than by position in the list.
        "train_loss": float(
            loss_by_epoch.get(
                int(best_state["epoch"]),
                loss_by_epoch[max(loss_by_epoch)] if loss_by_epoch else float("nan"),
            )
        ),
        "loss_history": loss_history,
        "val_history": val_history,
        "early_stopped": epochs_run < num_epochs,
        "nan_events": nan_events,
        "num_epochs": int(num_epochs),
        "patience": int(patience),
    }


# ---------------------------------------------------------------------------
# single-stream models (baselines)
# ---------------------------------------------------------------------------


class ProdLDAModel(nn.Module):
    def __init__(self, vocab_size: int, num_topics: int, encoder_input_dim: Optional[int] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        input_dim = encoder_input_dim or vocab_size
        self.encoder = MLPEncoder(input_dim, num_topics)
        self.decoder = LinearDecoder(num_topics, vocab_size)

    def encode_theta(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        z = _reparameterize(mu, logvar) if sample else mu
        return F.softmax(z, dim=-1), mu, logvar

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.decoder.get_beta()



    def train_step(
        self,
        x_norm: torch.Tensor,
        x_counts: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # sampling only while training: the validation loss is then a
        # deterministic function of the posterior mean instead of one noisy
        # Monte-Carlo draw per epoch
        theta, mu, logvar = self.encode_theta(x_norm, sample=self.training)
        logits = self.decoder(theta)
        recon = reconstruction_loss_from_counts(logits, x_counts, per_token=False)
        kl = gaussian_kl(mu, logvar).mean()
        loss = recon + kl_weight * kl + TOPIC_ORTH_WEIGHT * topic_orthogonality_penalty(
            self.decoder.beta_tensor()
        )
        return loss, {"recon": recon, "kl": kl}


class ENTMModel(nn.Module):
    """ETM (Dieng et al. 2020): embedding decoder with a word-probability mixture.

    Trains the standard ETM objective.  The decoder mixes *normalised* topic
    word distributions, ``p(w | d) = sum_k theta_dk softmax_v(alpha_k E^T)``,
    which is the objective of the original paper (eq. 5-6).  There is no
    batch-norm and no dropout on theta in this decoder, and the embedding
    temperature is 1.0, matching the published model.
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        word_embeddings: np.ndarray,
        trainable_embeddings: bool = False,
        emb_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = MLPEncoder(vocab_size, num_topics)
        self.decoder = EmbeddingDecoder(
            num_topics,
            word_embeddings,
            trainable_embeddings=trainable_embeddings,
            temperature=emb_temperature,
        )

    def encode_theta(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        z = _reparameterize(mu, logvar) if sample else mu
        return F.softmax(z, dim=-1), mu, logvar

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.decoder.get_beta()

    def train_step(
        self,
        x_norm: torch.Tensor,
        x_counts: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        theta, mu, logvar = self.encode_theta(x_norm, sample=self.training)
        log_p = self.decoder.log_prob(theta)
        recon = _nll_from_log_prob(log_p, x_counts, per_token=False)
        kl = gaussian_kl(mu, logvar).mean()
        loss = recon + kl_weight * kl + TOPIC_ORTH_WEIGHT * topic_orthogonality_penalty(
            self.decoder.topic_parameter()
        )
        return loss, {"recon": recon, "kl": kl}


class CTMLiteModel(nn.Module):
    """Contextual-topic-model-style encoder (BoW + projected sentence embedding)."""

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.ctx_proj = nn.Linear(context_dim, vocab_size, bias=False)
        input_dim = vocab_size * 2
        self.encoder = MLPEncoder(input_dim, num_topics)
        self.decoder = LinearDecoder(num_topics, vocab_size)

    def encode_theta(self, bow_norm: torch.Tensor, ctx: torch.Tensor, sample: bool = True) -> torch.Tensor:
        ctx_proj = self.ctx_proj(ctx)
        features = torch.cat([bow_norm, ctx_proj], dim=-1)
        mu, logvar = self.encoder(features)
        z = _reparameterize(mu, logvar) if sample else mu
        return F.softmax(z, dim=-1), mu, logvar

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.decoder.get_beta()

    def train_step(
        self,
        bow_norm: torch.Tensor,
        ctx: torch.Tensor,
        x_counts: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        theta, mu, logvar = self.encode_theta(bow_norm, ctx,
                                               sample=self.training)
        logits = self.decoder(theta)
        recon = reconstruction_loss_from_counts(logits, x_counts, per_token=False)
        kl = gaussian_kl(mu, logvar).mean()
        loss = recon + kl_weight * kl + TOPIC_ORTH_WEIGHT * topic_orthogonality_penalty(
            self.decoder.beta_tensor()
        )
        return loss, {"recon": recon, "kl": kl}


# ---------------------------------------------------------------------------
# proposed model
# ---------------------------------------------------------------------------


class FCNTM(nn.Module):
    """Field-collaborative neural topic model.

    A change-order document d=(x_cause, x_item) is represented by two field-conditioned
    variational streams q_cause(z|x_cause) and q_item(z|x_item) that share one embedding
    decoder over the global vocabulary.  Each present stream reconstructs *all* present
    fields (cross-field supervision); a symmetric Jensen-Shannon term aligns the two
    field-specific topic mixtures; when a field is missing its stream falls back to the
    prior, which makes the model robust to partial records.
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        word_embeddings: Optional[np.ndarray] = None,
        consistency_weight: float = 0.5,
        cross_reconstruction: bool = True,
        trainable_embeddings: bool = False,
        decoder_type: str = "linear",
        cause_recon_weight: float = CAUSE_RECON_WEIGHT,
        cause_dropout: float = 0.0,
        item_dropout: float = 0.0,
        exclusivity_weight: float = 0.0,
        biterm_weight: float = 0.0,
        biterm_samples: int = 4096,
        biterm_temperature: float = 1.0,
        force_encode_missing: bool = False,
        emb_temperature: float = EMB_DECODER_TEMPERATURE,
        field_dropout: float = 0.0,
        fusion_gate: bool = False,
        fused_recon: bool = False,
        fused_recon_norm: str = "joint",
        fused_recon_weight: float = 0.5,
        fusion_mode: Optional[str] = None,
        missing_prior: str = "zero",
        field_weights: Optional[tuple] = None,
        recon_norm: str = "count",
        recon_scale: float = 1.0,
        exclusivity_mode: str = "entropy",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.consistency_weight = consistency_weight
        self.cross_reconstruction = cross_reconstruction
        self.cause_recon_weight = cause_recon_weight
        self.decoder_type = decoder_type
        self.cause_dropout = cause_dropout
        self.item_dropout = item_dropout
        self.exclusivity_weight = exclusivity_weight
        self.biterm_weight = biterm_weight
        self.biterm_samples = biterm_samples
        self.biterm_temperature = biterm_temperature
        self.force_encode_missing = force_encode_missing
        self.field_dropout = field_dropout
        self.fusion_gate = fusion_gate
        self.fused_recon = fused_recon
        # How the extra fused reconstruction is balanced against the per-stream
        # terms:
        #   "joint"    - historical behaviour: the fused decoder counts as one
        #                more active stream, so *every* reconstruction and KL
        #                term is divided by (n_streams + 1).  This also rescales
        #                the stream terms, and since the consistency and
        #                orthogonality penalties are added after that division
        #                they become relatively 1.5x heavier.
        #   "balanced" - the per-stream terms keep exactly the scale they have
        #                without the fused decoder, and the fused reconstruction
        #                is added as its own term with ``fused_recon_weight``.
        if fused_recon_norm not in ("joint", "balanced"):
            raise ValueError(fused_recon_norm)
        self.fused_recon_norm = fused_recon_norm
        self.fused_recon_weight = float(fused_recon_weight)
        # ``fusion_mode`` selects how the two field mixtures are combined into
        # the document-level representation: "mean" (0.5/0.5 average, the
        # historical default), "gate" (learnable scalar gate) or "poe"
        # (precision-weighted product of experts, i.e. a Gaussian product).
        self.fusion_mode = fusion_mode or ("gate" if fusion_gate else "mean")
        # "zero"      : a missing field falls back to the N(0, I) prior, i.e.
        #               its stream contributes nothing to the fused mixture;
        # "learned"   : the fused mixture of a record whose cause field is
        #               missing adds a learnable corpus-level offset to the
        #               item stream, which is trained by fused reconstruction.
        self.missing_prior = missing_prior
        # Explicit (cause, item) reconstruction weights.  When given they
        # override ``cause_recon_weight`` so that the two fields can be
        # balanced by corpus statistics instead of a private-corpus constant.
        self.field_weights = field_weights
        # "count" : raw-count NLL (long fields dominate the loss);
        # "token" : per-token NLL, rescaled by ``recon_scale`` so the total
        #           magnitude stays comparable to the count-scale objective.
        self.recon_norm = recon_norm
        self.recon_scale = float(recon_scale)
        self.exclusivity_mode = exclusivity_mode
        if self.fusion_mode == "gate":
            self.fusion_logit = nn.Parameter(torch.zeros(1))
        if missing_prior == "learned":
            self.missing_bias = nn.Parameter(torch.zeros(num_topics))
        self.encoder_cause = MLPEncoder(vocab_size, num_topics)
        self.encoder_item = MLPEncoder(vocab_size, num_topics)
        if decoder_type == "linear":
            self.decoder = LinearDecoder(num_topics, vocab_size)
        elif decoder_type == "embedding":
            if word_embeddings is None:
                raise ValueError("embedding decoder requires word_embeddings")
            self.decoder = EmbeddingDecoder(
                num_topics,
                word_embeddings,
                trainable_embeddings=trainable_embeddings,
                temperature=emb_temperature,
            )
        else:
            raise ValueError(decoder_type)

    def decode(self, theta: torch.Tensor) -> torch.Tensor:
        return self.decoder(theta)

    @staticmethod
    def _poe(mu_a, logvar_a, mu_b, logvar_b):
        """Precision-weighted product of two diagonal Gaussians (product of experts)."""
        prec_a = torch.exp(-logvar_a)
        prec_b = torch.exp(-logvar_b)
        mu = (mu_a * prec_a + mu_b * prec_b) / (prec_a + prec_b + 1e-8)
        return F.softmax(mu, dim=-1)

    def _field_weight(self, name: str) -> float:
        if self.field_weights is not None:
            return float(self.field_weights[0] if name == "cause"
                         else self.field_weights[1])
        return float(self.cause_recon_weight if name == "cause" else 1.0)

    def _exclusivity_penalty(self) -> torch.Tensor:
        """Differentiable surrogate of the KADJ exclusivity metric.

        ``metrics.exclusivity`` keeps the top-n words of every row-normalised
        topic and averages their column-normalised mass.  ``argsort`` is not
        differentiable, so this surrogate scores every word by
        ``p_kw / sum_k p_kw`` instead: mass placed on a word that is already
        claimed by other topics is penalised.  Returns a *negative* score so it
        can be added to the loss.
        """
        beta = (self.decoder.beta_tensor() if self.decoder_type == "linear"
                else self.decoder.topic_parameter())
        p = torch.softmax(beta - beta.max(dim=1, keepdim=True).values, dim=1)
        p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
        col = p.sum(dim=0, keepdim=True)
        exclusive = (p * p / (col + 1e-12)).sum(dim=1)
        return -exclusive.mean()

    def fuse(self, theta_cause, theta_item, both, mu_c=None, logvar_c=None,
             mu_i=None, logvar_i=None):
        """Combine the two field mixtures into the document-level mixture.

        Records without a cause field fall back to the item stream; with
        ``missing_prior="learned"`` they instead use the item stream logits
        plus a learnable corpus-level offset, which gives the missing-field
        path its own trained representation.
        """
        fused = theta_item.clone()
        absent = ~both
        if self.missing_prior == "learned" and mu_i is not None and absent.any():
            fused[absent] = F.softmax(mu_i[absent] + self.missing_bias, dim=-1)
        if both.any():
            if self.fusion_mode == "poe" and mu_c is not None:
                fused[both] = self._poe(
                    mu_c[both], logvar_c[both], mu_i[both], logvar_i[both]
                )
            elif self.fusion_mode == "gate":
                g = torch.sigmoid(self.fusion_logit)
                fused[both] = g * theta_cause[both] + (1.0 - g) * theta_item[both]
            else:
                fused[both] = 0.5 * (theta_cause[both] + theta_item[both])
        return fused

    def set_biterms(self, w1, w2) -> None:
        """Register corpus biterms used by the optional co-occurrence term."""
        self.register_buffer("biterm_w1", torch.as_tensor(w1, dtype=torch.long))
        self.register_buffer("biterm_w2", torch.as_tensor(w2, dtype=torch.long))

    def stream_theta(
        self,
        encoder: MLPEncoder,
        x_norm: torch.Tensor,
        present: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Allocate the posterior tensors directly: slicing the first
        # ``num_topics`` columns of a normalised BoW is wrong in shape whenever
        # K > V and mixes up two unrelated quantities otherwise.
        mu = torch.zeros(
            (x_norm.shape[0], self.num_topics),
            dtype=x_norm.dtype,
            device=x_norm.device,
        )
        logvar = torch.zeros_like(mu)
        active = present.bool()
        if active.any():
            mu_a, logvar_a = encoder(x_norm[active])
            mu = mu.index_copy(0, active.nonzero(as_tuple=True)[0], mu_a)
            logvar = logvar.index_copy(0, active.nonzero(as_tuple=True)[0], logvar_a)
        z = _reparameterize(mu, logvar) if sample else mu
        theta = F.softmax(z, dim=-1)
        return theta, mu, logvar

    def train_step(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        cause_counts: torch.Tensor,
        item_counts: torch.Tensor,
        has_cause: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # full-field counts always serve as reconstruction targets; input dropout
        # only corrupts the encoder view (denoising objective).
        cause_targets = cause_counts
        item_targets = item_counts
        # word-level input dropout (denoising). Only active during training.
        # A masked field still uses its own encoder; targets are never masked,
        # so the model must reconstruct full evidence from a partial view.
        if self.training and self.cause_dropout > 0.0:
            keep_c = torch.bernoulli(
                (1.0 - self.cause_dropout) * torch.ones_like(cause_counts)
            )
            cause_counts = cause_counts * keep_c
            cause_norm = _normalize_rows(cause_counts)
        if self.training and self.item_dropout > 0.0:
            keep_i = torch.bernoulli(
                (1.0 - self.item_dropout) * torch.ones_like(item_counts)
            )
            item_counts = item_counts * keep_i
            item_norm = _normalize_rows(item_counts)
        batch = cause_norm.shape[0]
        # ``cause_observed`` marks records that really carry a cause field (the
        # reconstruction target); ``cause_present`` additionally reflects
        # field-level dropout, i.e. which records actually run the cause encoder.
        cause_observed = has_cause.bool() | self.force_encode_missing
        cause_present = cause_observed
        if self.training and self.field_dropout > 0.0:
            keep = torch.bernoulli(
                (1.0 - self.field_dropout)
                * torch.ones(batch, device=cause_norm.device)
            ).bool()
            cause_present = cause_present & keep
        present_fields = []
        theta_cause = theta_item = None
        mu_cause = logvar_cause = mu_item = logvar_item = None
        if cause_present.any():
            theta_cause, mu_c, logvar_c = self.stream_theta(
                self.encoder_cause, cause_norm, cause_present,
                sample=self.training
            )
            mu_cause, logvar_cause = mu_c, logvar_c
            present_fields.append(("cause", theta_cause, mu_c, logvar_c))
        theta_item, mu_i, logvar_i = self.stream_theta(
            self.encoder_item, item_norm, torch.ones(batch, dtype=torch.bool),
            sample=self.training
        )
        mu_item, logvar_item = mu_i, logvar_i
        present_fields.append(("item", theta_item, mu_i, logvar_i))

        total_loss = torch.zeros(batch, device=cause_norm.device)
        components = {}
        recon_sum = torch.zeros(batch, device=cause_norm.device)
        kl_sum = torch.zeros(batch, device=cause_norm.device)
        active_streams = torch.zeros(batch, device=cause_norm.device)

        for field_name, theta, mu, logvar in present_fields:
            active = (
                cause_present if field_name == "cause" else torch.ones(batch, dtype=torch.bool)
            )
            if not active.any():
                continue
            active_streams = active_streams + active.float()
            # Decode a stream exactly once.  Calling self.decode() inside the
            # target-field loop (as an earlier revision did) runs the decoder
            # twice per stream with different dropout masks and different
            # batch-norm batches, so the two reconstruction terms of one stream
            # did not come from the same function.
            idx_active = active.nonzero(as_tuple=True)[0]
            theta_active = theta[idx_active]
            logits_active = self.decode(theta_active)
            recon_active = torch.zeros(
                idx_active.shape[0], device=cause_norm.device)
            if self.cross_reconstruction:
                fields_to_decode = [("cause", cause_targets), ("item", item_targets)]
            else:
                fields_to_decode = [
                    (field_name, cause_targets if field_name == "cause" else item_targets)
                ]
            for g_name, g_counts in fields_to_decode:
                present_g_full = (
                    cause_observed if g_name == "cause"
                    else torch.ones(batch, dtype=torch.bool)
                )
                present_g = present_g_full[idx_active]
                if not present_g.any():
                    continue
                weight = self._field_weight(g_name)
                g_counts = g_counts[idx_active]
                g_counts_masked = torch.where(
                    present_g.unsqueeze(-1), g_counts, torch.zeros_like(g_counts)
                )
                per_doc = reconstruction_loss_from_counts(
                    logits_active, g_counts_masked, reduce=False,
                    per_token=self.recon_norm == "token",
                )
                # 2026-10-15 v3 review C3: apply the field weight after the
                # loss; weighting the counts first cancels the weight in the
                # per-token mode (count mode is linear in the counts, so the
                # value is unchanged there)
                recon_active = (recon_active
                                + self.recon_scale * weight * per_doc)
            # a stream that is inactive contributes nothing (its rows stay 0);
            # conversely an active stream's batch-norm statistics are computed
            # only over the records in which that stream is actually active.
            recon = torch.zeros(
                batch, device=cause_norm.device, dtype=recon_active.dtype
            ).index_put((idx_active,), recon_active)
            kl = gaussian_kl(mu, logvar)
            total_loss = total_loss + recon + kl_weight * kl
            recon_sum = recon_sum + recon
            kl_sum = kl_sum + kl
            components[f"recon_{field_name}"] = recon[active].mean()
            components[f"kl_{field_name}"] = kl[active].mean()

        if self.fused_recon:
            # The fused topic mixture is what the fused readout uses, but the
            # per-stream loop above never optimises it.  Decode from the fused
            # mixture as well so the consensus representation is actually trained.
            fused_theta = (
                self.fuse(theta_cause, theta_item, cause_present,
                          mu_cause, logvar_cause, mu_item, logvar_item)
                if theta_cause is not None
                else self.fuse(theta_cause, theta_item, cause_present,
                               None, None, mu_item, logvar_item)
            )
            fused_logits = self.decode(fused_theta)
            fused_extra = torch.zeros(batch, device=cause_norm.device)
            joint_norm = self.fused_recon_norm == "joint"
            for g_name, g_counts in (("cause", cause_targets),
                                     ("item", item_targets)):
                present_g = (
                    cause_observed if g_name == "cause"
                    else torch.ones(batch, dtype=torch.bool)
                )
                if not present_g.any():
                    continue
                weight = self._field_weight(g_name)
                g_counts_masked = torch.where(
                    present_g.unsqueeze(-1), g_counts,
                    torch.zeros_like(g_counts)
                )
                # 2026-10-15 v3 review C3: weight applied after the loss
                term = (self.recon_scale * weight
                        * reconstruction_loss_from_counts(
                            fused_logits, g_counts_masked, reduce=False,
                            per_token=self.recon_norm == "token",
                        ))
                if joint_norm:
                    total_loss = total_loss + term
                else:
                    fused_extra = fused_extra + term
            if joint_norm:
                active_streams = active_streams + 1.0

        total_loss = total_loss / active_streams.clamp_min(1.0)
        if self.fused_recon and self.fused_recon_norm != "joint":
            total_loss = total_loss + self.fused_recon_weight * fused_extra

        consistency = torch.zeros(batch, device=cause_norm.device)
        if self.consistency_weight > 0 and theta_cause is not None and cause_present.any():
            # Align the deterministic (posterior-mean) topic mixtures of the two
            # streams.  The posterior means were already produced by the stream
            # pass above, so re-use them instead of running the encoders twice
            # more (which also mixed a second dropout mask into the regulariser).
            theta_mean_c = F.softmax(mu_cause[cause_present], dim=-1)
            theta_mean_i = F.softmax(mu_item[cause_present], dim=-1)
            js = sym_js(theta_mean_c, theta_mean_i)
            consistency = consistency.index_put(
                (cause_present.nonzero(as_tuple=True)[0],), js, accumulate=False
            )
            components["consistency"] = js.mean()

        total_loss = total_loss + self.consistency_weight * consistency
        penalty = (
            topic_orthogonality_penalty(self.decoder.beta_tensor())
            if self.decoder_type == "linear"
            else topic_orthogonality_penalty(self.decoder.topic_parameter())
        )
        total = total_loss.mean() + TOPIC_ORTH_WEIGHT * penalty
        if self.exclusivity_weight > 0.0:
            if self.exclusivity_mode == "kadj":
                total = total + self.exclusivity_weight * self._exclusivity_penalty()
            else:
                # column-conditional entropy: push each word's probability mass
                # into as few topics as possible (differentiable proxy of
                # exclusivity).  Measured: its gradient on beta is ~1e-6 of the
                # reconstruction gradient, so this term is effectively inert.
                beta = self.decoder.beta_tensor()
                p_wk = torch.softmax(beta - beta.max(dim=0, keepdim=True).values, dim=1)
                p_kw = p_wk / (p_wk.sum(dim=0, keepdim=True) + 1e-12)
                col_ent = -(p_kw * torch.log(p_kw.clamp_min(1e-12))).sum(dim=0)
                total = total + self.exclusivity_weight * col_ent.mean()
        if self.biterm_weight > 0.0 and hasattr(self, "biterm_w1"):
            # Biterm likelihood: p(w, w') = sum_k pi_k * beta_kw * beta_kw'.
            # A stochastic mini-batch of corpus biterms is enough to regularize
            # the decoder toward word-pair co-occurrence (the NPMI/UMass signal).
            beta_prob = self.decoder.beta_prob_tensor(self.biterm_temperature)
            theta_all = theta_item.clone()
            if theta_cause is not None and cause_present.any():
                both = cause_present
                theta_all[both] = 0.5 * (theta_cause[both] + theta_item[both])
            pi = theta_all.mean(dim=0)
            n_bt = int(self.biterm_w1.shape[0])
            m = min(int(self.biterm_samples), n_bt)
            pick = torch.randint(0, n_bt, (m,), device=beta_prob.device)
            pw1 = beta_prob[:, self.biterm_w1[pick]]
            pw2 = beta_prob[:, self.biterm_w2[pick]]
            p_pair = (pi.unsqueeze(1) * pw1 * pw2).sum(dim=0).clamp_min(1e-12)
            loss_bt = -torch.log(p_pair).mean()
            total = total + self.biterm_weight * loss_bt
            components["biterm"] = loss_bt
        return total, components

    def encode_doc_theta(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        has_cause: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Deterministic topic mixtures from each stream and their fusion."""
        cause_present = has_cause.bool() | self.force_encode_missing
        theta_cause, mu_c, logvar_c = self.stream_theta(
            self.encoder_cause, cause_norm, cause_present, sample=False
        )
        theta_item, mu_i, logvar_i = self.stream_theta(
            self.encoder_item,
            item_norm,
            torch.ones(cause_norm.shape[0], dtype=torch.bool),
            sample=False,
        )
        both = cause_present
        fused = self.fuse(theta_cause, theta_item, both,
                          mu_c, logvar_c, mu_i, logvar_i)
        return fused, {
            "cause": theta_cause,
            "item": theta_item,
            "both": both.cpu().numpy(),
        }

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        if self.decoder_type == "embedding":
            return self.decoder.get_beta()
        return self.decoder.get_beta()


class FCNTMCtx(nn.Module):
    """FC-NTM with per-field contextual embeddings concatenated to BoW.

    Each field stream encodes ``[normalized BoW; contextual embedding]``.
    Everything else (shared linear decoder, cross-field reconstruction,
    JS consistency, missing-field prior fallback, word-level input masking)
    follows FC-NTM.
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        ctx_dim: int,
        consistency_weight: float = 0.5,
        cause_dropout: float = 0.0,
        item_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.consistency_weight = consistency_weight
        self.cause_dropout = cause_dropout
        self.item_dropout = item_dropout
        in_dim = vocab_size + ctx_dim
        self.encoder_cause = MLPEncoder(in_dim, num_topics)
        self.encoder_item = MLPEncoder(in_dim, num_topics)
        self.decoder = LinearDecoder(num_topics, vocab_size)

    def _cat(self, bow_norm: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return torch.cat([bow_norm, ctx], dim=-1)

    def stream_theta(
        self,
        encoder: MLPEncoder,
        x_cat: torch.Tensor,
        present: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = torch.zeros(x_cat.shape[0], self.num_topics, device=x_cat.device)
        logvar = torch.zeros_like(mu)
        active = present.bool()
        if active.any():
            idx = active.nonzero(as_tuple=True)[0]
            mu_a, logvar_a = encoder(x_cat[active])
            mu = mu.index_copy(0, idx, mu_a)
            logvar = logvar.index_copy(0, idx, logvar_a)
        z = _reparameterize(mu, logvar) if sample else mu
        return F.softmax(z, dim=-1), mu, logvar

    def train_step(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        cause_ctx: torch.Tensor,
        item_ctx: torch.Tensor,
        cause_counts: torch.Tensor,
        item_counts: torch.Tensor,
        has_cause: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cause_targets = cause_counts
        item_targets = item_counts
        if self.training and self.cause_dropout > 0.0:
            keep_c = torch.bernoulli(
                (1.0 - self.cause_dropout) * torch.ones_like(cause_counts)
            )
            cause_counts = cause_counts * keep_c
            cause_norm = _normalize_rows(cause_counts)
        if self.training and self.item_dropout > 0.0:
            keep_i = torch.bernoulli(
                (1.0 - self.item_dropout) * torch.ones_like(item_counts)
            )
            item_counts = item_counts * keep_i
            item_norm = _normalize_rows(item_counts)
        cause_cat = self._cat(cause_norm, cause_ctx)
        item_cat = self._cat(item_norm, item_ctx)

        batch = cause_norm.shape[0]
        theta_cause = theta_item = None
        present_fields = []
        if has_cause.any():
            theta_cause, mu_c, logvar_c = self.stream_theta(
                self.encoder_cause, cause_cat, has_cause, sample=self.training
            )
            present_fields.append(("cause", theta_cause, mu_c, logvar_c))
        theta_item, mu_i, logvar_i = self.stream_theta(
            self.encoder_item, item_cat, torch.ones(batch, dtype=torch.bool),
            sample=self.training
        )
        present_fields.append(("item", theta_item, mu_i, logvar_i))

        total_loss = torch.zeros(batch, device=cause_norm.device)
        components = {}
        active_streams = torch.zeros(batch, device=cause_norm.device)
        for field_name, theta, mu, logvar in present_fields:
            active = (
                has_cause.bool()
                if field_name == "cause"
                else torch.ones(batch, dtype=torch.bool)
            )
            if not active.any():
                continue
            active_streams += active.float()
            recon = torch.zeros(batch, device=cause_norm.device)
            fields_to_decode = [("cause", cause_targets), ("item", item_targets)]
            for g_name, g_counts in fields_to_decode:
                present_g = (
                    has_cause.bool()
                    if g_name == "cause"
                    else torch.ones(batch, dtype=torch.bool)
                )
                if not present_g.any():
                    continue
                weight = CAUSE_RECON_WEIGHT if g_name == "cause" else 1.0
                g_masked = torch.where(
                    present_g.unsqueeze(-1), g_counts * weight, torch.zeros_like(g_counts)
                )
                logits = self.decode(theta)
                recon = recon + reconstruction_loss_from_counts(
                    logits, g_masked, reduce=False, per_token=False
                )
            recon = torch.where(active, recon, torch.zeros_like(recon))
            kl = gaussian_kl(mu, logvar)
            total_loss = total_loss + recon + kl_weight * kl
            components[f"recon_{field_name}"] = recon[active].mean()
            components[f"kl_{field_name}"] = kl[active].mean()
        if self.fused_recon:
            # The fused topic mixture is what the fused readout uses, but the
            # per-stream loop above never optimises it.  Decode from the fused
            # mixture as well so the consensus representation is actually trained.
            fused_theta = (
                self.fuse(theta_cause, theta_item, cause_present)
                if theta_cause is not None
                else theta_item
            )
            fused_logits = self.decode(fused_theta)
            for g_name, g_counts in (("cause", cause_targets),
                                     ("item", item_targets)):
                present_g = (
                    cause_observed if g_name == "cause"
                    else torch.ones(batch, dtype=torch.bool)
                )
                if not present_g.any():
                    continue
                weight = (
                    self.cause_recon_weight if g_name == "cause" else 1.0
                )
                g_counts_masked = torch.where(
                    present_g.unsqueeze(-1), g_counts * weight,
                    torch.zeros_like(g_counts)
                )
                total_loss = total_loss + reconstruction_loss_from_counts(
                    fused_logits, g_counts_masked, reduce=False,
                    per_token=False
                )
            active_streams = active_streams + 1.0

        total_loss = total_loss / active_streams.clamp_min(1.0)

        consistency = torch.zeros(batch, device=cause_norm.device)
        if theta_cause is not None and has_cause.any():
            idx = has_cause.bool().nonzero(as_tuple=True)[0]
            mean_c, _ = self.encoder_cause(cause_cat[idx])
            mean_i, _ = self.encoder_item(item_cat[idx])
            js = sym_js(F.softmax(mean_c, dim=-1), F.softmax(mean_i, dim=-1))
            consistency = consistency.index_put((idx,), js, accumulate=False)
            components["consistency"] = js.mean()
        total_loss = total_loss + self.consistency_weight * consistency
        penalty = topic_orthogonality_penalty(self.decoder.beta_tensor())
        return total_loss.mean() + TOPIC_ORTH_WEIGHT * penalty, components

    def decode(self, theta: torch.Tensor) -> torch.Tensor:
        return self.decoder(theta)

    def encode_doc_theta(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        cause_ctx: torch.Tensor,
        item_ctx: torch.Tensor,
        has_cause: torch.Tensor,
    ):
        cause_cat = self._cat(cause_norm, cause_ctx)
        item_cat = self._cat(item_norm, item_ctx)
        theta_cause, _, _ = self.stream_theta(
            self.encoder_cause, cause_cat, has_cause, sample=False
        )
        theta_item, _, _ = self.stream_theta(
            self.encoder_item,
            item_cat,
            torch.ones(cause_norm.shape[0], dtype=torch.bool),
            sample=False,
        )
        fused = theta_item.clone()
        both = has_cause.bool()
        if both.any():
            fused[both] = 0.5 * (theta_cause[both] + theta_item[both])
        return fused

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.decoder.get_beta()



class FCNTMConcat(nn.Module):
    """Concatenated-field single-encoder control for FC-NTM.

    One encoder reads the concatenated two-field bag of words
    ``[x_bg; x_sm]`` and produces a single variational stream; the stream
    decodes both fields through the same shared linear decoder, with the same
    field weights, KL warm-up and topic-orthogonality penalty as FC-NTM.
    Whole-field dropout is emulated by zeroing the background half of the
    encoder input (the reconstruction targets stay complete), and the same
    per-field word-level dropout is applied to each half.  There is no JS
    consistency term, no fusion gate and no fused-reconstruction term: the
    single stream *is* the document representation.  The objective therefore
    also differs from FC-NTM (fusion reconstruction and consistency terms),
    so this control compares the complete configurations; it does not
    isolate the two-encoder architecture on its own (2026-10-15 v3 review).
    """

    def __init__(
        self,
        vocab_size: int,
        num_topics: int,
        cause_dropout: float = 0.30,
        item_dropout: float = 0.40,
        field_dropout: float = 0.0,
        cause_recon_weight: float = CAUSE_RECON_WEIGHT,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.cause_dropout = cause_dropout
        self.item_dropout = item_dropout
        self.field_dropout = field_dropout
        self.cause_recon_weight = float(cause_recon_weight)
        # the concatenated input doubles the first layer, so the single
        # encoder has (to first order) the same parameter count as the two
        # per-field encoders of FC-NTM
        self.encoder = MLPEncoder(2 * vocab_size, num_topics)
        self.decoder = LinearDecoder(num_topics, vocab_size)

    def decode(self, theta: torch.Tensor) -> torch.Tensor:
        return self.decoder(theta)

    def _encode(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        sample: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([cause_norm, item_norm], dim=-1)
        mu, logvar = self.encoder(x)
        z = _reparameterize(mu, logvar) if sample else mu
        return F.softmax(z, dim=-1), mu, logvar



    def train_step(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        cause_counts: torch.Tensor,
        item_counts: torch.Tensor,
        has_cause: torch.Tensor,
        kl_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cause_targets = cause_counts
        item_targets = item_counts
        cause_view = cause_norm
        item_view = item_norm
        if self.training and self.cause_dropout > 0.0:
            keep_c = torch.bernoulli(
                (1.0 - self.cause_dropout) * torch.ones_like(cause_counts)
            )
            cause_view = _normalize_rows(cause_counts * keep_c)
        if self.training and self.item_dropout > 0.0:
            keep_i = torch.bernoulli(
                (1.0 - self.item_dropout) * torch.ones_like(item_counts)
            )
            item_view = _normalize_rows(item_counts * keep_i)
        observed = has_cause.bool()
        # whole-field dropout: zero the background half of the input view for a
        # random subset of records (targets are never masked)
        if self.training and self.field_dropout > 0.0:
            keep = torch.bernoulli(
                (1.0 - self.field_dropout)
                * torch.ones(cause_view.shape[0], device=cause_view.device)
            ).bool()
            dropped = observed & ~keep
            if dropped.any():
                cause_view = cause_view.clone()
                cause_view[dropped] = 0.0
        # 2026-10-14 review C1: validation must be deterministic, exactly as
        # in the main FC-NTM (sample=self.training), otherwise the
        # reparameterisation noise leaks into early stopping and best-model
        # selection
        theta, mu, logvar = self._encode(cause_view, item_view,
                                          sample=self.training)
        logits = self.decode(theta)
        recon = self.recon_scale_term(logits, cause_targets, item_targets,
                                      observed)
        kl = gaussian_kl(mu, logvar)
        total = recon + kl_weight * kl
        batch_total = total.mean() + TOPIC_ORTH_WEIGHT * \
            topic_orthogonality_penalty(self.decoder.beta_tensor())
        components = {
            "recon": recon.mean(),
            "kl": kl.mean(),
        }
        return batch_total, components

    def recon_scale_term(
        self,
        logits: torch.Tensor,
        cause_targets: torch.Tensor,
        item_targets: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """Per-document reconstruction of both fields with FC-NTM's weights."""
        cause_masked = torch.where(
            observed.unsqueeze(-1),
            cause_targets * self.cause_recon_weight,
            torch.zeros_like(cause_targets),
        )
        recon = reconstruction_loss_from_counts(
            logits, cause_masked, reduce=False,
        ) + reconstruction_loss_from_counts(
            logits, item_targets, reduce=False,
        )
        return recon

    def encode_doc_theta(
        self,
        cause_norm: torch.Tensor,
        item_norm: torch.Tensor,
        has_cause: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Deterministic topic mixture; a missing background field leaves its
        half of the input at zero (the concatenated analogue of the
        missing-field fallback)."""
        observed = has_cause.bool()
        cause_view = cause_norm.clone()
        if not bool(observed.all()):
            cause_view[~observed] = 0.0
        theta, _, _ = self._encode(cause_view, item_norm, sample=False)
        return theta, {"both": observed.cpu().numpy()}

    @torch.no_grad()
    def get_beta(self) -> np.ndarray:
        return self.decoder.get_beta()
