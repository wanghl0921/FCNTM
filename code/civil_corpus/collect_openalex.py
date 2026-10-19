# -*- coding: utf-8 -*-
"""Collect a public civil-engineering 两个字段 text corpus from OpenAlex.

Two-field structure used by FC-NTM:  item = 标题 (title), reason = 摘要 (abstract).
OpenAlex is CC0 and needs no API key (a ``mailto`` puts us in the polite pool).

    python collect_openalex.py --probe                 # resolve the subfield id
    python collect_openalex.py --max-works 6000        # collect + report

Outputs (in this directory):
    civil_openalex_raw.csv      every work kept after the length filters
    civil_openalex_stats.txt    length / ratio / duplicate statistics
    civil_openalex_samples.txt  readable samples for a manual sanity check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://api.openalex.org"
MAILTO = "fcnm-research@example.com"          # polite pool; change if you like
UA = "FCNTM-civil-corpus/1.0 (research; mailto:%s)" % MAILTO


def get(path: str, **params) -> dict:
    params.setdefault("mailto", MAILTO)
    url = f"{API}{path}?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and "select" in params:
                # the API rejects unknown/renamed ``select`` fields - just ask
                # for the full records instead of failing the whole harvest
                print("  select rejected by the API; retrying without it",
                      flush=True)
                params.pop("select")
                url = f"{API}{path}?" + urllib.parse.urlencode(params, doseq=True)
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                continue
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/5 after HTTP {exc.code} (sleep {wait}s)",
                  flush=True)
            time.sleep(wait)
        except Exception as exc:                       # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/5 after {type(exc).__name__}: {exc} "
                  f"(sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {url}")


def abstract_from_inverted(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in inv.items():
        for pos in spots:
            positions.append((int(pos), word))
    positions.sort()
    return " ".join(word for _, word in positions)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text or "")


EN_STOP = {"the", "of", "and", "to", "in", "is", "for", "with", "on", "are",
           "that", "this", "by", "as", "an", "be", "from", "was", "were",
           "which", "can", "we", "it", "or", "at", "have", "has", "not"}

_TRIM_HEAD = re.compile(r"^\s*(abstract|summary|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)"
                        r"\s*[:\-–—]?\s*", re.I)
_TRIM_TAIL = re.compile(r"\s*(copyright|©|\(c\)|all rights reserved)"
                        r"[^.]*\.?\s*$", re.I)


def clean_abstract(text: str) -> str:
    """Drop publisher scaffolding and the leading 'Abstract' label."""
    text = re.sub(r"\s+", " ", text or "").strip()
    for _ in range(2):
        text = _TRIM_HEAD.sub("", text).strip()
        text = _TRIM_TAIL.sub("", text).strip()
    return text


def looks_english(text: str) -> bool:
    toks = [w.lower() for w in words(text)]
    if len(toks) < 40:
        return False
    return sum(t in EN_STOP for t in toks) / len(toks) >= 0.04


def resolve_subfield(name: str) -> tuple[str, str]:
    """Find the OpenAlex subfield that matches ``name`` best."""
    data = get("/subfields", search=name, **{"per-page": 25})
    cands = data.get("results", [])
    if not cands:
        raise SystemExit(f"no subfield matched {name!r}")
    print(f"  subfield candidates for {name!r}:")
    for c in cands[:8]:
        print(f"    {c['id'].rsplit('/', 1)[-1]:>6}  {c['display_name']:<45} "
              f"{c.get('works_count', 0):,} works")
    best = cands[0]
    return best["id"], best["display_name"]


def probe() -> int:
    for name in ("civil engineering", "structural engineering",
                 "construction engineering", "geotechnical engineering"):
        sid, label = resolve_subfield(name)
        data = get("/works", filter=f"primary_topic.subfield.id:{sid.rsplit('/', 1)[-1]}"
                                    ",has_abstract:true,language:en", **{"per-page": 1})
        print(f"  -> {label}: {data['meta']['count']:,} English works with abstract\n")
    return 0


def collect(max_works: int, subfields: list[str], min_abs: int, max_abs: int,
            min_title: int, max_title: int, out_name: str,
            from_year: int, to_year: int,
            licenses: str = "cc-by|cc-by-sa|public-domain") -> int:
    ids, labels = [], []
    for name in subfields:
        sid, label = resolve_subfield(name)
        ids.append(sid.rsplit("/", 1)[-1])
        labels.append(label)
    sid_filter = "|".join(dict.fromkeys(ids))
    label = " + ".join(dict.fromkeys(labels))
    years = list(range(from_year, to_year + 1))
    per_year = max(1, -(-max_works // len(years)))          # ceil
    print(f"\ncollecting from {label} (subfields {sid_filter}), "
          f"years {from_year}-{to_year}, target {per_year}/year, "
          f"licences {licenses}", flush=True)
    kept, seen = [], set()
    drops = Counter()
    per_page = 200
    scanned = 0
    for year in years:
        cursor, got = "*", 0
        while got < per_year:
            data = get("/works",
                       filter=f"primary_topic.subfield.id:{sid_filter}"
                              ",has_abstract:true,language:en"
                              f",publication_year:{year}"
                              ",type:article|review|conference-paper"
                              + (f",primary_location.license:{licenses}"
                                 if licenses else ""),
                       **{"per-page": per_page, "cursor": cursor,
                          "sort": "publication_date:asc",
                          "select": "id,doi,title,abstract_inverted_index,"
                                    "publication_year,primary_location,type"})
            rows = data.get("results", [])
            if not rows:
                break
            for w in rows:
                scanned += 1
                title = (w.get("title") or "").strip()
                abstract = clean_abstract(
                    abstract_from_inverted(w.get("abstract_inverted_index")))
                t_words, a_words = words(title), words(abstract)
                if not (min_title <= len(t_words) <= max_title):
                    drops["title_length"] += 1
                    continue
                if not (min_abs <= len(a_words) <= max_abs):
                    drops["abstract_length"] += 1
                    continue
                if not looks_english(abstract):
                    drops["not_english"] += 1
                    continue
                key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
                if key in seen:
                    drops["duplicate_title"] += 1
                    continue
                seen.add(key)
                loc = (w.get("primary_location") or {}).get("source") or {}
                lic = ((w.get("primary_location") or {}).get("license") or "")
                kept.append({
                    "openalex_id": w["id"].rsplit("/", 1)[-1],
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "title": title,
                    "abstract": abstract,
                    "year": w.get("publication_year") or "",
                    "venue": (loc.get("display_name") or "")[:80],
                    "license": lic.rsplit("/", 1)[-1],
                    "type": w.get("type") or "",
                    "title_words": len(t_words),
                    "abstract_words": len(a_words),
                })
                got += 1
            cursor = data["meta"].get("next_cursor")
            if not cursor:
                break
        print(f"  {year}: +{got} (total {len(kept):,}, scanned {scanned:,}, "
              f"drops {dict(drops)})", flush=True)

    out = ROOT / out_name
    fields = list(kept[0].keys()) if kept else []
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nwrote {out} ({len(kept):,} records)")
    report(kept, label, ROOT / "civil_openalex_stats.txt",
           ROOT / "civil_openalex_samples.txt")
    return 0


def report(rows: list[dict], label: str, stats_path: Path,
           samples_path: Path) -> None:
    import statistics as st
    n = len(rows)
    tl = [r["title_words"] for r in rows]
    al = [r["abstract_words"] for r in rows]
    ratios = [r["abstract_words"] / max(r["title_words"], 1) for r in rows]
    years = Counter(r["year"] for r in rows)
    venues = Counter(r["venue"] for r in rows if r["venue"])
    vocab = Counter()
    for r in rows:
        vocab.update(w.lower() for w in words(r["title"] + " " + r["abstract"]))
    lines = [
        f"corpus: {label}",
        f"records: {n:,}",
        f"title words      : mean {st.mean(tl):.1f}  median {st.median(tl):.0f} "
        f" min {min(tl)}  max {max(tl)}",
        f"abstract words   : mean {st.mean(al):.1f}  median {st.median(al):.0f} "
        f" min {min(al)}  max {max(al)}",
        f"abstract/title   : mean {st.mean(ratios):.1f}  median "
        f"{st.median(ratios):.1f}   (GitHub 12.9 / AG News 3.9)",
        f"years            : {min(years)}--{max(years)}",
        f"top venues       :",
    ]
    for v, c in venues.most_common(10):
        lines.append(f"    {c:>5}  {v}")
    lines.append(f"vocabulary (raw) : {len(vocab):,} types, "
                 f"{sum(vocab.values()):,} tokens")
    lines.append("top 40 tokens    : " + ", ".join(w for w, _ in vocab.most_common(40)))
    text = "\n".join(lines)
    stats_path.write_text(text, encoding="utf-8")
    print("\n" + text)

    sample = []
    for r in rows[:6]:
        sample.append(f"--- {r['openalex_id']}  ({r['year']}, {r['venue']}) "
                      f"title={r['title_words']}w abstract={r['abstract_words']}w\n"
                      f"TITLE: {r['title']}\nABSTRACT: {r['abstract'][:700]}\n")
    samples_path.write_text("\n".join(sample), encoding="utf-8")
    print(f"\nwrote samples -> {samples_path}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--subfields",
                    default="civil engineering,building and construction")
    ap.add_argument("--max-works", type=int, default=6000)
    ap.add_argument("--min-abs", type=int, default=60)
    ap.add_argument("--max-abs", type=int, default=400)
    ap.add_argument("--min-title", type=int, default=4)
    ap.add_argument("--max-title", type=int, default=30)
    ap.add_argument("--out", default="civil_openalex_raw.csv")
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--to-year", type=int, default=2023)
    ap.add_argument("--licenses", default="cc-by|cc-by-sa|public-domain",
                    help="pipe-separated OpenAlex licence keys; '' = no filter")
    args = ap.parse_args()
    if args.probe:
        return probe()
    return collect(args.max_works, args.subfields.split(","),
                   args.min_abs, args.max_abs,
                   args.min_title, args.max_title, args.out,
                   args.from_year, args.to_year, args.licenses)


if __name__ == "__main__":
    raise SystemExit(main())
