# -*- coding: utf-8 -*-
"""Stage 3 of the 2026-10-07 review fixes: embedding cache + token windows.

embed_public.py: the cache key and identity sidecar now cover the full model
name, revision, maximum length, pooling method and algorithm version; the
chunked control tokenises with the model's own tokenizer and windows token ids
with room for the special tokens, so no document token is dropped.
tests_review_fixes.py gets a cache-key / window-coverage regression test.
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


pairs = [
    ("imports",
     "import hashlib\nimport os\nfrom pathlib import Path",
     "import hashlib\nimport json\nimport os\nimport time\nfrom pathlib import Path",
     1),
    ("algo constant",
     "EMB_CHUNK_LONG = os.environ.get(\n"
     "    \"FCNTM_EN_EMBEDDING_CHUNKED\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\")\n"
     "_MODEL = None",
     "EMB_CHUNK_LONG = os.environ.get(\n"
     "    \"FCNTM_EN_EMBEDDING_CHUNKED\", \"0\").strip().lower() in (\"1\", \"true\", \"yes\")\n"
     "# bump when the windowing / pooling implementation changes: the value is\n"
     "# part of both the cache key and the identity sidecar, so vectors produced\n"
     "# by another algorithm can never be reused (2026-10-07 review, items 3-4)\n"
     "EMB_ALGO = \"tokwin-v2\"\n"
     "_MODEL = None",
     1),
    ("encode windows",
     "            all_vectors.append((summed / counts).numpy())\n"
     "    return np.vstack(all_vectors).astype(np.float32)\n"
     "\n"
     "\n"
     "def _cache_path(kind: str, items: Sequence[str]) -> Path:\n"
     "    model_short = EMB_MODEL.split(\"/\")[-1]\n"
     "    payload = \"\\x1f\".join(items)\n"
     "    digest = hashlib.sha1(payload.encode(\"utf-8\")).hexdigest()[:16]\n"
     "    OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
     "    return OUT_DIR / f\"emb_{model_short}_{kind}_{digest}.npy\"\n",
     "            all_vectors.append((summed / counts).numpy())\n"
     "    return np.vstack(all_vectors).astype(np.float32)\n"
     "\n"
     "\n"
     "def _encode_windows(windows: List[List[int]],\n"
     "                    batch_size: int = 64) -> np.ndarray:\n"
     "    \"\"\"Encode pre-tokenised windows, adding the special tokens per window.\"\"\"\n"
     "    model = _get_model()\n"
     "    tokenizer = model[\"tokenizer\"]\n"
     "    encoder = model[\"model\"]\n"
     "    all_vectors = []\n"
     "    with torch.no_grad():\n"
     "        for start in range(0, len(windows), batch_size):\n"
     "            batch = windows[start : start + batch_size]\n"
     "            ids = [tokenizer.build_inputs_with_special_tokens(list(w))\n"
     "                   for w in batch]\n"
     "            encoded = tokenizer.pad({\"input_ids\": ids}, padding=True,\n"
     "                                    return_tensors=\"pt\")\n"
     "            out = encoder(**encoded)\n"
     "            mask = encoded[\"attention_mask\"].unsqueeze(-1).float()\n"
     "            summed = (out.last_hidden_state * mask).sum(dim=1)\n"
     "            counts = mask.sum(dim=1).clamp_min(1.0)\n"
     "            all_vectors.append((summed / counts).numpy())\n"
     "    return np.vstack(all_vectors).astype(np.float32)\n"
     "\n"
     "\n"
     "def token_windows(ids: Sequence[int], max_length: int,\n"
     "                  n_special: int) -> List[List[int]]:\n"
     "    \"\"\"Split token ids into windows that each fit the encoder's limit.\n"
     "\n"
     "    ``n_special`` is the number of tokens the tokenizer adds around an\n"
     "    input ([CLS]/[SEP] for BERT-family models), so each window carries at\n"
     "    most ``max_length - n_special`` ids.  Every input id lands in exactly\n"
     "    one window; an empty input still yields one (empty) window.\n"
     "    \"\"\"\n"
     "    window = max(1, int(max_length) - int(n_special))\n"
     "    parts = [list(ids[start : start + window])\n"
     "             for start in range(0, len(ids), window)]\n"
     "    return parts or [[]]\n"
     "\n"
     "\n"
     "def _encoder_identity(kind: str, chunk_long: bool) -> dict:\n"
     "    \"\"\"Everything that can change the vector for the same input text.\"\"\"\n"
     "    return {\n"
     "        \"algo\": EMB_ALGO,\n"
     "        \"model\": EMB_MODEL,\n"
     "        \"revision\": EMB_REVISION,\n"
     "        \"max_length\": int(EMB_MAX_LENGTH),\n"
     "        \"kind\": kind,\n"
     "        \"pooling\": \"token-window-mean\" if chunk_long else \"truncated-mean\",\n"
     "    }\n"
     "\n"
     "\n"
     "def _cache_path(kind: str, items: Sequence[str],\n"
     "                chunk_long: bool = False) -> Path:\n"
     "    model_short = EMB_MODEL.split(\"/\")[-1]\n"
     "    payload = \"\\x1f\".join(items)\n"
     "    digest = hashlib.sha1(payload.encode(\"utf-8\")).hexdigest()[:16]\n"
     "    ident = json.dumps(_encoder_identity(kind, chunk_long), sort_keys=True)\n"
     "    key = hashlib.sha1(ident.encode(\"utf-8\")).hexdigest()[:12]\n"
     "    OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
     "    return OUT_DIR / f\"emb_{model_short}_{kind}_{key}_{digest}.npy\"\n",
     1),
    ("chunk_and_pool",
     "def chunk_and_pool(texts: Sequence[str]) -> np.ndarray:\n"
     "    \"\"\"Mean-pool MiniLM embeddings over 128-token chunks of each document.\n"
     "\n"
     "    The fifth review pointed out that the contextual baselines only ever saw\n"
     "    the first ``max_length`` tokens while the bag-of-words models saw the whole\n"
     "    text - a real disadvantage on GitHub, whose issues can exceed 20k tokens.\n"
     "    This encoder splits every document into windows of ``EMB_MAX_LENGTH`` tokens\n"
     "    (on whitespace, which is how the corpus is tokenised) and averages the chunk\n"
     "    vectors, so long documents are no longer truncated.\n"
     "    \"\"\"\n"
     "    texts = list(texts)\n"
     "    token_lists = [t.split() for t in texts]\n"
     "    chunks, owners = [], []\n"
     "    for i, tokens in enumerate(token_lists):\n"
     "        if not tokens:\n"
     "            chunks.append(\"\")\n"
     "            owners.append(i)\n"
     "            continue\n"
     "        for start in range(0, len(tokens), EMB_MAX_LENGTH):\n"
     "            chunks.append(\" \".join(tokens[start:start + EMB_MAX_LENGTH]))\n"
     "            owners.append(i)\n"
     "    vectors = _encode(chunks)\n"
     "    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)\n"
     "    weights = np.zeros(len(texts), dtype=np.float32)\n"
     "    for vector, owner in zip(vectors, owners):\n"
     "        out[owner] += vector\n"
     "        weights[owner] += 1.0\n"
     "    return out / np.maximum(weights[:, None], 1.0)\n",
     "def chunk_and_pool(texts: Sequence[str]) -> np.ndarray:\n"
     "    \"\"\"Mean-pool MiniLM embeddings over every token window of each document.\n"
     "\n"
     "    The contextual baselines could only see the first ``max_length`` tokens\n"
     "    while the bag-of-words models saw the whole text - a real disadvantage on\n"
     "    GitHub, whose issues can exceed 20k tokens.  The earlier control split on\n"
     "    whitespace and counted special tokens against the window, so window tails\n"
     "    could still be truncated.  This version tokenises with the model's own\n"
     "    tokenizer, reserves room for the special tokens and windows the token ids,\n"
     "    so *every* token of the document enters exactly one window.\n"
     "    \"\"\"\n"
     "    texts = list(texts)\n"
     "    model = _get_model()\n"
     "    tokenizer = model[\"tokenizer\"]\n"
     "    n_special = int(tokenizer.num_special_tokens_to_add(pair=False))\n"
     "    chunks, owners = [], []\n"
     "    n_tokens = n_covered = 0\n"
     "    for i, text in enumerate(texts):\n"
     "        ids = list(tokenizer(text, add_special_tokens=False)[\"input_ids\"])\n"
     "        n_tokens += len(ids)\n"
     "        for part in token_windows(ids, EMB_MAX_LENGTH, n_special):\n"
     "            n_covered += len(part)\n"
     "            chunks.append(part)\n"
     "            owners.append(i)\n"
     "    if n_covered != n_tokens:\n"
     "        raise RuntimeError(\n"
     "            f\"token windows lost {n_tokens - n_covered} of {n_tokens} tokens\")\n"
     "    vectors = _encode_windows(chunks)\n"
     "    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)\n"
     "    weights = np.zeros(len(texts), dtype=np.float32)\n"
     "    for vector, owner in zip(vectors, owners):\n"
     "        out[owner] += vector\n"
     "        weights[owner] += 1.0\n"
     "    return out / np.maximum(weights[:, None], 1.0)\n",
     1),
    ("embed_texts",
     "def embed_texts(texts: Sequence[str], kind: str = \"doc\",\n"
     "                chunk_long: bool = False) -> np.ndarray:\n"
     "    \"\"\"Encode ``texts``; ``FCNTM_EN_EMBEDDING_CHUNKED=1`` forces window pooling.\"\"\"\n"
     "    texts = list(texts)\n"
     "    chunk_long = chunk_long or EMB_CHUNK_LONG\n"
     "    path = _cache_path(kind, texts)\n"
     "    if chunk_long:\n"
     "        path = path.with_name(path.stem + \"_chunked\" + path.suffix)\n"
     "    if path.exists():\n"
     "        return np.load(path)\n"
     "    arr = chunk_and_pool(texts) if chunk_long else _encode(texts)\n"
     "    np.save(path, arr)\n"
     "    return arr\n",
     "def embed_texts(texts: Sequence[str], kind: str = \"doc\",\n"
     "                chunk_long: bool = False) -> np.ndarray:\n"
     "    \"\"\"Encode ``texts``; ``FCNTM_EN_EMBEDDING_CHUNKED=1`` forces window pooling.\n"
     "\n"
     "    The cache key names the encoder, its revision, the maximum length, the\n"
     "    pooling method and the algorithm version; the sidecar repeats that\n"
     "    identity and the array shape.  An unusable cache is moved aside instead\n"
     "    of being reused (2026-10-07 review, item 4).\n"
     "    \"\"\"\n"
     "    texts = list(texts)\n"
     "    chunk_long = chunk_long or EMB_CHUNK_LONG\n"
     "    path = _cache_path(kind, texts, chunk_long)\n"
     "    sidecar = path.with_name(path.stem + \".identity.json\")\n"
     "    identity = {\"encoder\": _encoder_identity(kind, chunk_long),\n"
     "                \"n_items\": len(texts)}\n"
     "    if path.exists():\n"
     "        try:\n"
     "            arr = np.load(path)\n"
     "            meta = (json.loads(sidecar.read_text(encoding=\"utf-8\"))\n"
     "                    if sidecar.exists() else None)\n"
     "            if (meta is not None\n"
     "                    and meta.get(\"identity\") == identity\n"
     "                    and list(arr.shape) == list(meta.get(\"shape\", []))\n"
     "                    and arr.shape[0] == len(texts)):\n"
     "                return arr\n"
     "            print(f\"  cached embeddings {path.name} do not match this \"\n"
     "                  f\"encoder (sidecar missing or different); recomputing\",\n"
     "                  flush=True)\n"
     "        except Exception as exc:  # noqa: BLE001\n"
     "            print(f\"  cannot reuse cached embeddings {path.name}: {exc}; \"\n"
     "                  f\"recomputing\", flush=True)\n"
     "        stamp = time.strftime(\"%Y%m%d%H%M%S\")\n"
     "        for stale in (path, sidecar):\n"
     "            if stale.exists():\n"
     "                os.replace(stale, stale.with_name(\n"
     "                    stale.name + f\".stale{stamp}\"))\n"
     "    arr = chunk_and_pool(texts) if chunk_long else _encode(texts)\n"
     "    np.save(path, arr)\n"
     "    sidecar.write_text(json.dumps(\n"
     "        {\"identity\": identity, \"shape\": list(arr.shape)},\n"
     "        ensure_ascii=False, indent=1), encoding=\"utf-8\")\n"
     "    return arr\n",
     1),
]
patch("embed_public.py", pairs)

t = [
    ("new test",
     "def main() -> None:\n    print(\"review-fix checks\")",
     "def test_embedding_cache_identity_and_windows() -> None:\n"
     "    \"\"\"Cache key covers revision/length/pooling; windows lose no token.\"\"\"\n"
     "    import embed_public as EP\n"
     "\n"
     "    old_len, old_rev = EP.EMB_MAX_LENGTH, EP.EMB_REVISION\n"
     "    try:\n"
     "        base = EP._cache_path(\"doc\", [\"hello world\"], False)\n"
     "        EP.EMB_MAX_LENGTH = old_len + 128\n"
     "        assert EP._cache_path(\"doc\", [\"hello world\"], False) != base\n"
     "        EP.EMB_MAX_LENGTH = old_len\n"
     "        EP.EMB_REVISION = \"deadbeef\"\n"
     "        assert EP._cache_path(\"doc\", [\"hello world\"], False) != base\n"
     "        EP.EMB_REVISION = old_rev\n"
     "        assert EP._cache_path(\"doc\", [\"hello world\"], True) != base\n"
     "        for n in (0, 1, 126, 127, 253, 1000):\n"
     "            ids = list(range(n))\n"
     "            windows = EP.token_windows(ids, 128, 2)\n"
     "            assert [i for w in windows for i in w] == ids, n\n"
     "            assert all(len(w) <= 126 for w in windows), n\n"
     "    finally:\n"
     "        EP.EMB_MAX_LENGTH, EP.EMB_REVISION = old_len, old_rev\n"
     "    print(\"  ok  embedding cache key covers revision/length/pooling; \"\n"
     "          \"token windows lose no token\")\n"
     "\n"
     "\n"
     "def main() -> None:\n    print(\"review-fix checks\")",
     1),
    ("call",
     "    test_ablation_resume_identity()\n",
     "    test_ablation_resume_identity()\n    test_embedding_cache_identity_and_windows()\n",
     1),
]
patch("tests_review_fixes.py", t)

if fails:
    print("FAILED:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("stage 3 patches applied")