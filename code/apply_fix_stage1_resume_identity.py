# -*- coding: utf-8 -*-
"""Stage 1 of the 2026-10-07 review fixes: tie resume states to the experiment.

models.py: train_topic_vae_valid() now stores a result_identity payload inside
the resume file and refuses to load it when the requested identity differs
(archiving the stale file instead).  run_fcntm_train.py passes that identity.
tests_review_fixes.py gets a regression test for the review's counter-example.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs):
    path = ROOT / name
    text = path.read_text(encoding="utf-8")
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(name) for f in fails):
        path.write_text(text, encoding="utf-8")
        print(f"  patched {name}")
    return path


# ------------------------------------------------------------------ models.py
m = [
    ("imports",
     "from pathlib import Path\nfrom typing import Dict, List, Optional, Sequence, Tuple",
     "import os\nimport time\nfrom pathlib import Path\nfrom typing import Dict, List, Optional, Sequence, Tuple",
     1),
    ("signature",
     "    resume_path: Optional[str] = None,\n    resume_every: int = 50,\n    allow_resume: bool = True,\n) -> Dict[str, object]:",
     "    resume_path: Optional[str] = None,\n    resume_every: int = 50,\n    allow_resume: bool = True,\n    resume_identity: Optional[Dict[str, object]] = None,\n) -> Dict[str, object]:",
     1),
    ("docstring",
     "    ``resume_path`` writes the *full* training state (weights, optimizer\n"
     "    moments, epoch counter, histories and RNG states) every ``resume_every``\n"
     "    epochs and, if the file already exists, continues from it instead of\n"
     "    starting over.  This makes a long run interruptible: each invocation\n"
     "    advances the run by as many epochs as the machine allows.\n",
     "    ``resume_path`` writes the *full* training state (weights, optimizer\n"
     "    moments, epoch counter, histories and RNG states) every ``resume_every``\n"
     "    epochs and, if the file already exists, continues from it instead of\n"
     "    starting over.  This makes a long run interruptible: each invocation\n"
     "    advances the run by as many epochs as the machine allows.\n"
     "\n"
     "    A stored state is only reused when ``resume_identity`` (a\n"
     "    ``result_identity.identity`` payload describing the data, fields, model\n"
     "    configuration, learning rate and code version) matches the identity\n"
     "    recorded inside the file.  A mismatch, a corrupt file or a legacy state\n"
     "    without an identity is moved aside (``*.stale<timestamp>``) and training\n"
     "    starts fresh, so a changed objective can never silently continue an old\n"
     "    run.\n",
     1),
    ("load block",
     "    if resume_path and allow_resume and Path(resume_path).exists():\n"
     "        ck = torch.load(resume_path, map_location=\"cpu\", weights_only=False)\n"
     "        model.load_state_dict(ck[\"model_state\"])\n"
     "        optimizer.load_state_dict(ck[\"optimizer_state\"])\n"
     "        epochs_run = int(ck[\"epoch\"])\n"
     "        start_epoch = epochs_run + 1\n"
     "        best_val = float(ck[\"best_val\"])\n"
     "        best_state = ck[\"best_state\"]\n"
     "        patience_counter = int(ck.get(\"patience_counter\", 0))\n"
     "        loss_history = list(ck.get(\"loss_history\", []))\n"
     "        loss_by_epoch = {int(k): float(v)\n"
     "                         for k, v in ck.get(\"loss_by_epoch\", {}).items()}\n"
     "        val_history = list(ck.get(\"val_history\", []))\n"
     "        if ck.get(\"torch_rng\") is not None:\n"
     "            torch.set_rng_state(ck[\"torch_rng\"])\n"
     "        if ck.get(\"numpy_rng\") is not None:\n"
     "            np.random.set_state(ck[\"numpy_rng\"])\n"
     "        print(f\"    [resume] continuing from epoch {epochs_run}\", flush=True)\n",
     "    if resume_path and allow_resume and Path(resume_path).exists():\n"
     "        ck = None\n"
     "        try:\n"
     "            ck = torch.load(resume_path, map_location=\"cpu\",\n"
     "                            weights_only=False)\n"
     "        except Exception as exc:  # noqa: BLE001 - a corrupt state must not\n"
     "            print(f\"    [resume] cannot read {resume_path}: {exc}\",\n"
     "                  flush=True)  # crash, it is archived below\n"
     "        want_id = (resume_identity or {}).get(\"id\")\n"
     "        stored_id = (ck or {}).get(\"identity_id\")\n"
     "        if ck is not None and not want_id:\n"
     "            print(\"    [resume] no resume_identity was supplied; refusing \"\n"
     "                  \"to reuse an unidentified state\", flush=True)\n"
     "            ck = None\n"
     "        elif ck is not None and stored_id != want_id:\n"
     "            print(f\"    [resume] identity mismatch (stored {stored_id!r}, \"\n"
     "                  f\"current {want_id!r}); the old state cannot be reused\",\n"
     "                  flush=True)\n"
     "            ck = None\n"
     "        if ck is None:\n"
     "            archive = f\"{resume_path}.stale{time.strftime('%Y%m%d%H%M%S')}\"\n"
     "            os.replace(resume_path, archive)\n"
     "            print(f\"    [resume] old state moved to {archive}; \"\n"
     "                  f\"starting fresh\", flush=True)\n"
     "        else:\n"
     "            model.load_state_dict(ck[\"model_state\"])\n"
     "            optimizer.load_state_dict(ck[\"optimizer_state\"])\n"
     "            epochs_run = int(ck[\"epoch\"])\n"
     "            start_epoch = epochs_run + 1\n"
     "            best_val = float(ck[\"best_val\"])\n"
     "            best_state = ck[\"best_state\"]\n"
     "            patience_counter = int(ck.get(\"patience_counter\", 0))\n"
     "            loss_history = list(ck.get(\"loss_history\", []))\n"
     "            loss_by_epoch = {int(k): float(v)\n"
     "                             for k, v in ck.get(\"loss_by_epoch\", {}).items()}\n"
     "            val_history = list(ck.get(\"val_history\", []))\n"
     "            if ck.get(\"torch_rng\") is not None:\n"
     "                torch.set_rng_state(ck[\"torch_rng\"])\n"
     "            if ck.get(\"numpy_rng\") is not None:\n"
     "                np.random.set_state(ck[\"numpy_rng\"])\n"
     "            print(f\"    [resume] continuing from epoch {epochs_run} \"\n"
     "                  f\"(identity {stored_id})\", flush=True)\n",
     1),
    ("save block",
     "            \"torch_rng\": torch.get_rng_state(),\n"
     "            \"numpy_rng\": np.random.get_state(),\n"
     "        }, resume_path)\n",
     "            \"torch_rng\": torch.get_rng_state(),\n"
     "            \"numpy_rng\": np.random.get_state(),\n"
     "            \"identity_id\": (resume_identity or {}).get(\"id\"),\n"
     "            \"identity\": resume_identity,\n"
     "        }, resume_path)\n",
     1),
]

# --------------------------------------------------------- run_fcntm_train.py
r = [
    ("resume identity",
     "            resume_file = None\n"
     "            if details_path:\n"
     "                resume_file = f\"{details_path}.resume.seed{int(seed)}\"\n"
     "                extra[\"resume_path\"] = resume_file\n"
     "                extra[\"resume_every\"] = int(resume_every)\n"
     "                extra[\"allow_resume\"] = bool(allow_resume)\n",
     "            resume_file = None\n"
     "            if details_path:\n"
     "                resume_file = f\"{details_path}.resume.seed{int(seed)}\"\n"
     "                # tie the resume state to the same experiment as the result\n"
     "                # CSV: resuming by path alone let a state trained under another\n"
     "                # objective, split or learning rate be reused silently\n"
     "                # (2026-10-07 review, item 1)\n"
     "                resume_identity = RI.identity({\n"
     "                    \"script\": \"run_fcntm_train.run_corpus.resume\",\n"
     "                    \"seed\": int(seed),\n"
     "                    \"n_vocab\": int(corpus_train[\"n_vocab\"]),\n"
     "                    \"n_docs\": int(corpus_train[\"n_docs\"]),\n"
     "                    \"k\": int(K),\n"
     "                    \"field_dropout\": float(field_dropout),\n"
     "                    \"fusion\": fusion,\n"
     "                    \"fusion_norm\": fusion_norm,\n"
     "                    \"fusion_weight\": float(fusion_weight),\n"
     "                    \"recover_nan\": bool(recover_nan),\n"
     "                    \"lr\": float(lr or 0.0),\n"
     "                    \"val_fraction\": 0.2,\n"
     "                    \"kl_max\": 0.05,\n"
     "                    \"code\": RI.code_version(),\n"
     "                    \"data\": RI.text_digest([corpus_train.get(\"doc_tokens\")]),\n"
     "                    \"fields\": RI.text_digest(\n"
     "                        [corpus_train.get(\"reason_tokens\"),\n"
     "                         corpus_train.get(\"item_tokens\")]),\n"
     "                })\n"
     "                extra[\"resume_path\"] = resume_file\n"
     "                extra[\"resume_identity\"] = resume_identity\n"
     "                extra[\"resume_every\"] = int(resume_every)\n"
     "                extra[\"allow_resume\"] = bool(allow_resume)\n",
     1),
]

# ----------------------------------------------------- tests_review_fixes.py
t = [
    ("docstring",
     "Every test either passes or raises.  No training is performed.",
     "Every test either passes or raises.  Only the resume-identity test trains\n"
     "a tiny synthetic model; nothing here touches the paper's corpora.",
     1),
    ("new test",
     "def main() -> None:\n    print(\"review-fix checks\")",
     "def test_resume_identity_guard() -> None:\n"
     "    \"\"\"A resume state may only continue when its experiment identity matches.\n"
     "\n"
     "    Reproduces the 2026-10-07 review counter-example: the same resume path\n"
     "    is reused after the training target changes.  The guard must archive the\n"
     "    stale state and start fresh instead of silently continuing it.\n"
     "    \"\"\"\n"
     "    import tempfile\n"
     "    from pathlib import Path\n"
     "\n"
     "    import result_identity as RI\n"
     "\n"
     "    class Tiny(nn.Module):\n"
     "        def __init__(self):\n"
     "            super().__init__()\n"
     "            self.scale = nn.Parameter(torch.zeros(1))\n"
     "\n"
     "    def make_loss(model, target):\n"
     "        grid = torch.linspace(0.0, 1.0, 16)\n"
     "\n"
     "        def loss_fn(idx, kl_weight):\n"
     "            x = grid[idx]\n"
     "            pred = model.scale.expand_as(x)\n"
     "            return ((pred - target * x) ** 2).mean(), {}\n"
     "\n"
     "        return loss_fn\n"
     "\n"
     "    def run(path, payload, target, epochs):\n"
     "        model = Tiny()\n"
     "        ident = RI.identity(payload)\n"
     "        info = NN.train_topic_vae_valid(\n"
     "            model, make_loss(model, target), make_loss(model, target),\n"
     "            n_docs=16, seed=11, val_fraction=0.25, warmup_epochs=1,\n"
     "            kl_max=0.0, patience=1000, num_epochs=epochs, lr=1e-2,\n"
     "            resume_path=str(path), resume_every=1, allow_resume=True,\n"
     "            resume_identity=ident)\n"
     "        return info, ident\n"
     "\n"
     "    with tempfile.TemporaryDirectory() as tmp:\n"
     "        path = Path(tmp) / \"state.pt\"\n"
     "        payload_a = {\"script\": \"resume-guard\", \"target\": 0.0, \"lr\": 1e-2}\n"
     "        payload_b = {\"script\": \"resume-guard\", \"target\": 100.0, \"lr\": 1e-2}\n"
     "\n"
     "        run(path, payload_a, 0.0, epochs=4)\n"
     "        stored = torch.load(path, map_location=\"cpu\", weights_only=False)\n"
     "        assert stored[\"identity_id\"] == RI.identity(payload_a)[\"id\"]\n"
     "        assert int(stored[\"epoch\"]) == 4\n"
     "\n"
     "        # changed objective, same path: archive and restart, never continue\n"
     "        info_b, ident_b = run(path, payload_b, 100.0, epochs=1)\n"
     "        assert int(info_b[\"epochs_run\"]) == 1, info_b[\"epochs_run\"]\n"
     "        stored = torch.load(path, map_location=\"cpu\", weights_only=False)\n"
     "        assert stored[\"identity_id\"] == ident_b[\"id\"]\n"
     "        assert list(Path(tmp).glob(\"state.pt.stale*\")), \"stale state not archived\"\n"
     "\n"
     "        # a matching identity does resume: asking for fewer epochs than the\n"
     "        # stored state reports it continued (a fresh run would report 2)\n"
     "        run(path, payload_b, 100.0, epochs=3)\n"
     "        info_resume, _ = run(path, payload_b, 100.0, epochs=2)\n"
     "        assert int(info_resume[\"epochs_run\"]) == 3, info_resume[\"epochs_run\"]\n"
     "    print(\"  ok  resume identity guard archives changed experiments\")\n"
     "\n"
     "\n"
     "def main() -> None:\n    print(\"review-fix checks\")",
     1),
    ("call",
     "    test_resume_requires_complete_rows()\n",
     "    test_resume_requires_complete_rows()\n    test_resume_identity_guard()\n",
     1),
]

patch("models.py", m)
patch("run_fcntm_train.py", r)
patch("tests_review_fixes.py", t)

if fails:
    print("FAILED:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("stage 1 patches applied")