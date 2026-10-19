# -*- coding: utf-8 -*-
"""Stage 5 of the 2026-10-07 review fixes: a fixed-ID, fixed-vocab ladder.

public_data_civil.py: when max_abstract_words > 0 the truncation variants are
built from the *full* split's document IDs (never re-sampled) and the
dictionary is built on the untruncated full training split, so full / abs60 /
abs30 share one vocabulary and one evaluation reference.  The regression test
checks the ID alignment of the three cached splits.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs):
    path = ROOT / name
    raw = path.read_text(encoding="utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n") if crlf else raw
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(name) for f in fails):
        out = text.replace("\n", "\r\n") if crlf else text
        path.write_text(out, encoding="utf-8")
        print(f"  patched {name}")


pairs = [
    ("prepare signature",
     "def prepare_civil_splits(\n"
     "    train_n: int = 6000,\n"
     "    test_n: int = 2000,\n"
     "    seed: int = 20260919,\n"
     "    max_abstract_words: int = 0,\n"
     "    tag: str = \"\",\n"
     ") -> Dict[str, object]:\n"
     "    \"\"\"Deterministic split; ``max_abstract_words`` builds a field-ratio variant.\"\"\"\n"
     "    CIVIL_DIR.mkdir(parents=True, exist_ok=True)\n"
     "    suffix = tag or (\"full\" if not max_abstract_words else f\"abs{max_abstract_words}\")\n"
     "    out_train = CIVIL_DIR / f\"train_{suffix}.csv\"\n"
     "    out_test = CIVIL_DIR / f\"test_{suffix}.csv\"\n"
     "    if out_train.exists() and out_test.exists():\n"
     "        return {\"train\": str(out_train), \"test\": str(out_test), \"cached\": True}\n",
     "def prepare_civil_splits(\n"
     "    train_n: int = 6000,\n"
     "    test_n: int = 2000,\n"
     "    seed: int = 20260919,\n"
     "    max_abstract_words: int = 0,\n"
     "    tag: str = \"\",\n"
     ") -> Dict[str, object]:\n"
     "    \"\"\"Deterministic split; ``max_abstract_words`` builds a field-ratio variant.\n"
     "\n"
     "    A truncation variant is derived from the *full* split's document IDs\n"
     "    instead of re-sampling: the earlier implementation filtered on the\n"
     "    truncated field first and then reshuffled, so abs60/abs30 shared only\n"
     "    61\\%/53\\% of the full training records and the ladder changed the\n"
     "    sample as well as the field length (2026-10-07 review, item 5).\n"
     "    \"\"\"\n"
     "    CIVIL_DIR.mkdir(parents=True, exist_ok=True)\n"
     "    suffix = tag or (\"full\" if not max_abstract_words else f\"abs{max_abstract_words}\")\n"
     "    out_train = CIVIL_DIR / f\"train_{suffix}.csv\"\n"
     "    out_test = CIVIL_DIR / f\"test_{suffix}.csv\"\n"
     "    if max_abstract_words:\n"
     "        base = {\"train\": CIVIL_DIR / \"train_full.csv\",\n"
     "                \"test\": CIVIL_DIR / \"test_full.csv\"}\n"
     "        if not all(p.exists() for p in base.values()):\n"
     "            raise FileNotFoundError(\n"
     "                \"the truncation ladder needs the full split first \"\n"
     "                \"(run load_civil() once with max_abstract_words=0)\")\n"
     "        frames = {}\n"
     "        for name, path in base.items():\n"
     "            df = pd.read_csv(path)\n"
     "            df[\"abstract\"] = df[\"abstract_full\"].astype(str).map(\n"
     "                lambda s: \" \".join(str(s).split()[:max_abstract_words]))\n"
     "            too_thin = int(sum(len(en_tokenize_civil(a)) < 10\n"
     "                               for a in df[\"abstract\"]))\n"
     "            if too_thin:\n"
     "                raise ValueError(\n"
     "                    f\"{path.name}: {too_thin} abstracts fall below 10 \"\n"
     "                    f\"tokens at {max_abstract_words} words; the fixed-ID \"\n"
     "                    f\"ladder cannot be built\")\n"
     "            frames[name] = df\n"
     "        frames[\"train\"].to_csv(out_train, index=False, encoding=\"utf-8\")\n"
     "        frames[\"test\"].to_csv(out_test, index=False, encoding=\"utf-8\")\n"
     "        print(f\"fixed-ID ladder {suffix}: {out_train.name} / \"\n"
     "              f\"{out_test.name} \"\n"
     "              f\"({len(frames['train'])}+{len(frames['test'])} docs from \"\n"
     "              f\"the full split)\", flush=True)\n"
     "        return {\"train\": str(out_train), \"test\": str(out_test),\n"
     "                \"cached\": False, \"fixed_ids\": True}\n"
     "    if out_train.exists() and out_test.exists():\n"
     "        return {\"train\": str(out_train), \"test\": str(out_test), \"cached\": True}\n",
     1),
    ("fixed vocab",
     "    suffix = \"full\" if not max_abstract_words else f\"abs{max_abstract_words}\"\n"
     "    train_df = pd.read_csv(CIVIL_DIR / f\"train_{suffix}.csv\")\n"
     "    test_df = pd.read_csv(CIVIL_DIR / f\"test_{suffix}.csv\")\n"
     "\n"
     "    all_tokens = [en_tokenize_civil(t) + en_tokenize_civil(a)\n"
     "                  for t, a in zip(train_df[\"title\"], train_df[\"abstract\"])]\n",
     "    suffix = \"full\" if not max_abstract_words else f\"abs{max_abstract_words}\"\n"
     "    train_df = pd.read_csv(CIVIL_DIR / f\"train_{suffix}.csv\")\n"
     "    test_df = pd.read_csv(CIVIL_DIR / f\"test_{suffix}.csv\")\n"
     "\n"
     "    # the ladder keeps one vocabulary across all three field-lengths: it is\n"
     "    # built on the untruncated full split, so only the field length changes\n"
     "    # between variants (2026-10-07 review, item 5)\n"
     "    vocab_df = (pd.read_csv(CIVIL_DIR / \"train_full.csv\")\n"
     "                if max_abstract_words else train_df)\n"
     "    all_tokens = [en_tokenize_civil(t) + en_tokenize_civil(a)\n"
     "                  for t, a in zip(vocab_df[\"title\"], vocab_df[\"abstract\"])]\n",
     1),
    ("meta flag",
     "            \"split_seed\": 20260919,\n"
     "            \"max_abstract_words\": int(max_abstract_words),\n",
     "            \"split_seed\": 20260919,\n"
     "            \"max_abstract_words\": int(max_abstract_words),\n"
     "            \"fixed_ids\": bool(meta.get(\"fixed_ids\", False)),\n"
     "            \"fixed_vocabulary\": bool(max_abstract_words),\n",
     1),
]
patch("public_data_civil.py", pairs)

t = [
    ("new test",
     "def main() -> None:\n    print(\"review-fix checks\")",
     "def test_civil_ladder_keeps_full_ids() -> None:\n"
     "    \"\"\"The three ladder splits must cover the same documents.\"\"\"\n"
     "    import pandas as pd\n"
     "    from pathlib import Path\n"
     "\n"
     "    data = Path(__file__).resolve().parent / \"data\" / \"civil_openalex\"\n"
     "    base = {name: pd.read_csv(data / f\"{name}_full.csv\")\n"
     "            for name in (\"train\", \"test\")}\n"
     "    for variant in (\"abs60\", \"abs30\"):\n"
     "        for name, full in base.items():\n"
     "            path = data / f\"{name}_{variant}.csv\"\n"
     "            if not path.exists():\n"
     "                continue\n"
     "            df = pd.read_csv(path)\n"
     "            assert list(df[\"openalex_id\"]) == list(full[\"openalex_id\"]), \\\n"
     "                (variant, name)\n"
     "    print(\"  ok  civil ladder keeps the full split's document IDs\")\n"
     "\n"
     "\n"
     "def main() -> None:\n    print(\"review-fix checks\")",
     1),
    ("call",
     "    test_embedding_cache_identity_and_windows()\n",
     "    test_embedding_cache_identity_and_windows()\n    test_civil_ladder_keeps_full_ids()\n",
     1),
]
patch("tests_review_fixes.py", t)

if fails:
    print("FAILED:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("stage 5 patches applied")

# regenerate the two truncation splits under the fixed-ID rule
sys.path.insert(0, str(ROOT))
import public_data_civil as PC  # noqa: E402
for words in (60, 30):
    PC.prepare_civil_splits(6000, 2000, max_abstract_words=words)
print("regenerated abs60 / abs30 splits")