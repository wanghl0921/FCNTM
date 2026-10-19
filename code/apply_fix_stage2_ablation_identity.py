# -*- coding: utf-8 -*-
"""Stage 2 of the 2026-10-07 review fixes: pin ablation reuse to the run.

run_ablation_public.py: --resume now requires a matching result_identity sidecar
(dataset, abstract-words, K, lr, epochs, seeds, configs, code and data digests);
rows are additionally re-verified per row (dataset / K / epochs_budget /
config_hash / finite metrics).  --reuse-details only accepts a "full" dump whose
config hash and shapes match the current corpus; unverified dumps are refused.
tests_review_fixes.py gets a row-filter regression test.
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


# ------------------------------------------------------- run_ablation_public.py
pairs = [
    ("import RI",
     "import metrics as M\nimport models as NN\nfrom run_fcntm_train import class_tfidf, evaluate, top_words",
     "import metrics as M\nimport models as NN\nimport result_identity as RI\nfrom run_fcntm_train import class_tfidf, evaluate, top_words",
     1),
    ("complete_pairs helper",
     "def config_hash(kwargs: dict) -> str:\n"
     "    payload = json.dumps(kwargs, sort_keys=True).encode(\"utf-8\")\n"
     "    import hashlib\n"
     "    return hashlib.sha1(payload).hexdigest()[:12]\n",
     "def config_hash(kwargs: dict) -> str:\n"
     "    payload = json.dumps(kwargs, sort_keys=True).encode(\"utf-8\")\n"
     "    import hashlib\n"
     "    return hashlib.sha1(payload).hexdigest()[:12]\n"
     "\n"
     "\n"
     "def complete_pairs(rows, expected_hash, dataset, k, max_epochs, configs):\n"
     "    \"\"\"Keep only rows that this exact run could have produced.\n"
     "\n"
     "    ``--resume`` used to skip on (config, seed) alone, so a file produced\n"
     "    with another dataset, K, learning rate or epoch budget was reused\n"
     "    (2026-10-07 review, item 2).  A row now has to name the same dataset,\n"
     "    K and epoch budget and carry the config hash of the requested config;\n"
     "    both read-outs must have finite metrics for the pair to be complete.\n"
     "    \"\"\"\n"
     "    kept, have = [], {}\n"
     "    for row in rows:\n"
     "        try:\n"
     "            current = (row[\"config\"] in configs\n"
     "                       and row[\"config\"] in expected_hash\n"
     "                       and row.get(\"dataset\") == dataset\n"
     "                       and int(row.get(\"K_used\", -1)) == k\n"
     "                       and int(row.get(\"epochs_budget\", -1)) == max_epochs\n"
     "                       and row.get(\"config_hash\")\n"
     "                       == expected_hash[row[\"config\"]]\n"
     "                       and math.isfinite(float(row[\"CV\"]))\n"
     "                       and math.isfinite(float(row[\"NPMI\"])))\n"
     "        except (KeyError, TypeError, ValueError):\n"
     "            current = False\n"
     "        if not current:\n"
     "            continue\n"
     "        kept.append(row)\n"
     "        have.setdefault((row[\"config\"], int(row[\"seed\"])), set()).add(\n"
     "            row.get(\"readout\"))\n"
     "    done = {key for key, readouts in have.items()\n"
     "            if {\"beta\", \"cTFIDF\"} <= readouts}\n"
     "    return kept, done\n",
     1),
    ("run identity",
     "    seeds = [int(s) for s in args.seeds.split(\",\")]\n"
     "    configs = [c for c in args.configs.split(\",\") if c]\n",
     "    seeds = [int(s) for s in args.seeds.split(\",\")]\n"
     "    configs = [c for c in args.configs.split(\",\") if c]\n"
     "    expected_hash = {name: config_hash({**FULL_CONFIG, **CONFIGS[name]})\n"
     "                     for name in configs}\n"
     "    ident = RI.identity({\n"
     "        \"script\": \"run_ablation_public.main\",\n"
     "        \"dataset\": args.dataset,\n"
     "        \"abstract_words\": int(args.abstract_words),\n"
     "        \"k\": int(args.k),\n"
     "        \"lr\": float(args.lr),\n"
     "        \"max_epochs\": int(args.max_epochs),\n"
     "        \"seeds\": [int(s) for s in seeds],\n"
     "        \"configs\": sorted(configs),\n"
     "        \"full_config\": FULL_CONFIG,\n"
     "        \"code\": RI.code_version(),\n"
     "        \"data\": RI.text_digest([train.get(\"doc_tokens\"),\n"
     "                                 test.get(\"doc_tokens\")]),\n"
     "        \"fields\": RI.text_digest([train.get(\"reason_tokens\"),\n"
     "                                  train.get(\"item_tokens\"),\n"
     "                                  test.get(\"reason_tokens\"),\n"
     "                                  test.get(\"item_tokens\")]),\n"
     "    })\n",
     1),
    ("reuse details",
     "    reused = {}\n"
     "    if args.reuse_details:\n"
     "        p = Path(args.reuse_details)\n"
     "        if p.exists() and p.stat().st_size:\n"
     "            for rec in json.loads(p.read_text(encoding=\"utf-8\")):\n"
     "                if rec[\"model\"] == \"FC-NTM\":\n"
     "                    reused[(int(rec[\"seed\"]))] = (\n"
     "                        np.asarray(rec[\"beta\"], dtype=float),\n"
     "                        np.asarray(rec[\"theta\"], dtype=float)\n"
     "                        if rec.get(\"theta\") is not None else None,\n"
     "                        rec.get(\"config_hash\"))\n"
     "            print(\"reused full-config beta for seeds\", sorted(reused), flush=True)\n"
     "            unverified = [s for s, v in reused.items() if not v[2]]\n"
     "            if unverified:\n"
     "                print(\"  WARNING: the reused details for seeds\", unverified,\n"
     "                      \"carry no configuration hash; they cannot be tied to \"\n"
     "                      \"the current FULL_CONFIG and are written with \"\n"
     "                      \"reuse_verified=False\", flush=True)\n",
     "    reused = {}\n"
     "    if args.reuse_details:\n"
     "        p = Path(args.reuse_details)\n"
     "        expected_full = expected_hash.get(\"full\", config_hash(dict(FULL_CONFIG)))\n"
     "        if p.exists() and p.stat().st_size:\n"
     "            refused = []\n"
     "            for rec in json.loads(p.read_text(encoding=\"utf-8\")):\n"
     "                if rec.get(\"model\") != \"FC-NTM\":\n"
     "                    continue\n"
     "                seed = int(rec[\"seed\"])\n"
     "                stored = rec.get(\"config_hash\")\n"
     "                beta = np.asarray(rec[\"beta\"], dtype=float)\n"
     "                theta = (np.asarray(rec[\"theta\"], dtype=float)\n"
     "                         if rec.get(\"theta\") is not None else None)\n"
     "                why = []\n"
     "                if stored != expected_full:\n"
     "                    why.append(\"configuration hash does not match the \"\n"
     "                               \"full configuration of this run\")\n"
     "                if beta.ndim != 2 or beta.shape[1] != train[\"n_vocab\"]:\n"
     "                    why.append(\"beta shape does not match the current \"\n"
     "                               \"vocabulary\")\n"
     "                if theta is not None and theta.shape[0] != train[\"n_docs\"]:\n"
     "                    why.append(\"theta does not match the current training \"\n"
     "                               \"documents\")\n"
     "                if why:\n"
     "                    refused.append((seed, \"; \".join(why)))\n"
     "                    continue\n"
     "                reused[seed] = (beta, theta, stored)\n"
     "            if reused:\n"
     "                print(\"reused full-config beta for seeds\", sorted(reused),\n"
     "                      flush=True)\n"
     "            for seed, why in refused:\n"
     "                print(f\"  refusing reused details for seed {seed}: {why}\",\n"
     "                      flush=True)\n"
     "            if refused:\n"
     "                print(\"  unverified full-config details are not used as \"\n"
     "                      \"the reference row; those seeds are trained\",\n"
     "                      flush=True)\n",
     1),
    ("resume block",
     "    kept_rows = []\n"
     "    done = set()\n"
     "    if args.resume and out_path.exists() and out_path.stat().st_size:\n"
     "        with open(out_path, \"r\", encoding=\"utf-8-sig\", newline=\"\") as fh:\n"
     "            kept_rows = [r for r in csv.DictReader(fh)\n"
     "                         if r.get(\"config\") and r.get(\"seed\")]\n"
     "        have: dict = {}\n"
     "        for row in kept_rows:\n"
     "            try:\n"
     "                finite = (math.isfinite(float(row[\"CV\"]))\n"
     "                          and math.isfinite(float(row[\"NPMI\"])))\n"
     "            except (KeyError, TypeError, ValueError):\n"
     "                finite = False\n"
     "            if finite:\n"
     "                have.setdefault((row[\"config\"], int(row[\"seed\"])), set()).add(\n"
     "                    row.get(\"readout\"))\n"
     "        done = {key for key, readouts in have.items()\n"
     "                if {\"beta\", \"cTFIDF\"} <= readouts}\n"
     "        print(f\"resume: {len(kept_rows)} row(s) on disk, {len(done)} complete \"\n"
     "              f\"(config, seed) pair(s)\", flush=True)\n",
     "    kept_rows = []\n"
     "    done = set()\n"
     "    if args.resume and out_path.exists() and out_path.stat().st_size:\n"
     "        if not RI.matches(out_path, ident):\n"
     "            print(f\"  stored rows in {out_path.name} are not pinned to \"\n"
     "                  f\"this configuration ({RI.why_not(out_path, ident)}); \"\n"
     "                  f\"they cannot be skipped or reused\", flush=True)\n"
     "            stash = RI.stash_unpinned(out_path)\n"
     "            if stash:\n"
     "                print(f\"  old rows moved to {stash}\", flush=True)\n"
     "        else:\n"
     "            with open(out_path, \"r\", encoding=\"utf-8-sig\",\n"
     "                      newline=\"\") as fh:\n"
     "                kept_raw = [r for r in csv.DictReader(fh)\n"
     "                            if r.get(\"config\") and r.get(\"seed\")]\n"
     "            kept_rows, done = complete_pairs(\n"
     "                kept_raw, expected_hash, args.dataset, args.k,\n"
     "                args.max_epochs, configs)\n"
     "            print(f\"resume: {len(kept_rows)} of {len(kept_raw)} row(s) \"\n"
     "                  f\"verified for this configuration, {len(done)} complete \"\n"
     "                  f\"(config, seed) pair(s)\", flush=True)\n",
     1),
    ("fieldnames",
     "    fieldnames = [\"dataset\", \"config\", \"seed\", \"K_used\", \"readout\",\n"
     "                  \"config_hash\", \"reuse_verified\", \"epochs_budget\", \"seconds\",\n",
     "    fieldnames = [\"dataset\", \"config\", \"seed\", \"K_used\", \"readout\",\n"
     "                  \"config_hash\", \"reuse_verified\", \"identity_id\",\n"
     "                  \"epochs_budget\", \"seconds\",\n",
     1),
    ("row identity",
     "                        \"config_hash\": row_hash,\n"
     "                        \"reuse_verified\": reuse_verified,\n"
     "                        \"epochs_budget\": int(args.max_epochs),\n",
     "                        \"config_hash\": row_hash,\n"
     "                        \"reuse_verified\": reuse_verified,\n"
     "                        \"identity_id\": ident[\"id\"],\n"
     "                        \"epochs_budget\": int(args.max_epochs),\n",
     1),
    ("write sidecar",
     "        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction=\"ignore\")\n"
     "        writer.writeheader()\n"
     "        for row in kept_rows:\n"
     "            writer.writerow(row)\n"
     "        fh.flush()\n",
     "        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction=\"ignore\")\n"
     "        writer.writeheader()\n"
     "        for row in kept_rows:\n"
     "            writer.writerow(row)\n"
     "        fh.flush()\n"
     "        # pin the file (even a partial one) as soon as this run owns it, so\n"
     "        # an interrupted run resumes against the same identity\n"
     "        RI.write(out_path, ident)\n",
     1),
]
patch("run_ablation_public.py", pairs)

# --------------------------------------------------------- tests_review_fixes.py
t = [
    ("new test",
     "def main() -> None:\n    print(\"review-fix checks\")",
     "def test_ablation_resume_identity() -> None:\n"
     "    \"\"\"Row reuse must verify dataset, K, epoch budget and config hash.\"\"\"\n"
     "    import run_ablation_public as A\n"
     "\n"
     "    expected = {\"no-cross\":\n"
     "                A.config_hash({**A.FULL_CONFIG, **A.CONFIGS[\"no-cross\"]})}\n"
     "\n"
     "    def row(**kw):\n"
     "        base = {\"config\": \"no-cross\", \"seed\": \"11\", \"readout\": \"beta\",\n"
     "                \"CV\": \"0.5\", \"NPMI\": \"0.1\", \"dataset\": \"github\",\n"
     "                \"K_used\": \"30\", \"epochs_budget\": \"700\",\n"
     "                \"config_hash\": expected[\"no-cross\"]}\n"
     "        base.update(kw)\n"
     "        return base\n"
     "\n"
     "    kept, done = A.complete_pairs([row(), row(readout=\"cTFIDF\")], expected,\n"
     "                                  \"github\", 30, 700, [\"no-cross\"])\n"
     "    assert len(kept) == 2 and done == {(\"no-cross\", 11)}\n"
     "    for field, value in ((\"dataset\", \"agnews\"), (\"K_used\", \"20\"),\n"
     "                         (\"epochs_budget\", \"1\"),\n"
     "                         (\"config_hash\", \"deadbeef\")):\n"
     "        kept, done = A.complete_pairs([row(**{field: value})], expected,\n"
     "                                      \"github\", 30, 700, [\"no-cross\"])\n"
     "        assert not kept and not done, field\n"
     "    kept, done = A.complete_pairs([row(config=\"full\")], expected,\n"
     "                                  \"github\", 30, 700, [\"no-cross\"])\n"
     "    assert not kept and not done, \"unrequested config must be dropped\"\n"
     "    print(\"  ok  ablation resume verifies dataset/K/budget/config hash per row\")\n"
     "\n"
     "\n"
     "def main() -> None:\n    print(\"review-fix checks\")",
     1),
    ("call",
     "    test_resume_identity_guard()\n",
     "    test_resume_identity_guard()\n    test_ablation_resume_identity()\n",
     1),
]
patch("tests_review_fixes.py", t)

if fails:
    print("FAILED:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("stage 2 patches applied")