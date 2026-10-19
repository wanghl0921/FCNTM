# -*- coding: utf-8 -*-
"""Fetch OpenAlex primary topics for the civil-engineering corpus records.

The topic assignment comes from OpenAlex itself (an independent source that
never sees our models), and is cached to
``outputs/civil_openalex_topics.csv`` for the title-only archival task.

Usage:
    python fetch_openalex_topics.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import config

CIVIL_DIR = config.DATA_DIR / "civil_openalex"
CACHE = config.OUT_DIR / "civil_openalex_topics.csv"
API = "https://api.openalex.org/works"
MAILTO = "fcntm-paper@example.org"   # polite pool marker


def _get(url: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": f"fcntm ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after {exc}", flush=True)
            time.sleep(2 + 2 * attempt)
    return {}


def fetch_ids(ids) -> dict:
    out = {}
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        flt = "openalex_id:" + "|".join(chunk)
        url = (f"{API}?filter={urllib.parse.quote(flt)}"
               f"&per-page=50&select=id,primary_topic")
        data = _get(url)
        for w in data.get("results", []):
            key = w["id"].rsplit("/", 1)[-1]
            topic = (w.get("primary_topic") or {})
            out[key] = {
                "topic": topic.get("display_name", ""),
                "topic_id": topic.get("id", ""),
                "subfield": ((topic.get("subfield") or {}).get("display_name", "")),
                "field": ((topic.get("field") or {}).get("display_name", "")),
                "score": topic.get("score", None),
            }
        print(f"  fetched {min(start + 50, len(ids))}/{len(ids)}", flush=True)
        time.sleep(0.15)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if CACHE.exists() and not args.refresh:
        frame = pd.read_csv(CACHE)
        print(f"cache exists: {CACHE} ({len(frame)} rows)")
        return
    parts = []
    for split in ("train", "test"):
        path = CIVIL_DIR / f"{split}_full.csv"
        df = pd.read_csv(path)
        parts.append(df[["openalex_id"]])
    ids = sorted(set(pd.concat(parts)["openalex_id"].astype(str)))
    print(f"fetching topics for {len(ids)} records", flush=True)
    got = fetch_ids(ids)
    rows = []
    for rid in ids:
        rec = got.get(rid, {})
        rows.append({"openalex_id": rid, **{k: rec.get(k, "")
                                           for k in ("topic", "topic_id",
                                                     "subfield", "field",
                                                     "score")}})
    frame = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False, encoding="utf-8-sig")
    missing = int((frame["topic"] == "").sum())
    print(f"saved {CACHE}: {len(frame)} rows, {missing} without a topic")
    top = frame["topic"].value_counts().head(20)
    print(top.to_string())


if __name__ == "__main__":
    main()
